import utils  # needs to be the first

import sys
import matplotlib
from dataclasses import dataclass
import tensorflow as tf
import keras
from pathlib import Path
from typing import Any, Callable, List, NamedTuple, Tuple, Union
from enum import Enum
from absl import logging, app, flags
import textwrap
from kaggle.api.kaggle_api_extended import KaggleApi
from siamese_first.siamese_lib.layers import (
    DistanceLayer,
    SiameseModel,
    visualize as siamese_visualize,
)
import matplotlib.pyplot as plt
import os
import json
import getpass
import random
import inspect
import shlex
from PIL import Image
import pprint
import numpy as np
import platform


@dataclass
class AppConfig:
    data_path: Path
    seed: int


class FlagHolderHelpPair(NamedTuple):
    holder: flags.FlagHolder[Any]
    help: str


class EArgsName(str, Enum):
    working_directory = "working-directory"
    action = "action"
    seed = "seed"


class EActions(str, Enum):
    help = "help"
    train = "train"


_FLAGS = flags.FLAGS
_flag_helps: dict[EArgsName, str] = {
    EArgsName.working_directory: "Directory Used for RW operations",
    EArgsName.action: "Action to be performed by the application",
    EArgsName.seed: "Override Random seed generation for determinism",
}
_flag_holders: list[FlagHolderHelpPair] = [
    FlagHolderHelpPair(
        flags.DEFINE_string(
            EArgsName.working_directory.value,
            str(Path.home() / ".siamese_data"),
            _flag_helps[EArgsName.working_directory],
        ),
        _flag_helps[EArgsName.working_directory],
    ),
    FlagHolderHelpPair(
        flags.DEFINE_enum(
            EArgsName.action.value,
            None,
            [EActions.help.value, EActions.train.value],
            _flag_helps[EArgsName.action],
        ),
        _flag_helps[EArgsName.action],
    ),
    FlagHolderHelpPair(
        flags.DEFINE_integer(EArgsName.seed.value, None, _flag_helps[EArgsName.seed]),
        _flag_helps[EArgsName.seed],
    ),
]


def print_filtered_help():
    print("Usage:")
    max_name_length = max(len(holder.name) for holder, _ in _flag_holders)
    indent = max_name_length + 8
    help_wrap_width = 120

    for holder, help_text in _flag_holders:
        first_line = f"    --{holder.name.ljust(max_name_length)}  "
        wrapped_help = textwrap.wrap(help_text, width=help_wrap_width - indent)

        print(first_line + wrapped_help[0])
        for line in wrapped_help[1:]:
            print(" " * indent + line)

    print(
        'to get all flags supported by this Abseil based application, use "--helpfull"'
    )


def get_kaggle_credentials() -> dict[str, str]:
    """Prompt user for Kaggle credentials and return as a dictionary."""
    print("Enter your Kaggle credentials:")
    username = input("Kaggle Username: ")
    api_key = getpass.getpass("Kaggle API Key: ")
    return {"username": username, "key": api_key}


def authenticate_kaggle() -> KaggleApi:
    """Authenticate with Kaggle using user-provided credentials."""
    kaggle_dir = os.path.expanduser("~/.kaggle")
    kaggle_json_path = os.path.join(kaggle_dir, "kaggle.json")

    # Check if credentials already exist
    if os.path.exists(kaggle_json_path):
        print("Using cached Kaggle credentials.")
    else:
        credentials = get_kaggle_credentials()

        # Ensure ~/.kaggle directory exists
        os.makedirs(kaggle_dir, exist_ok=True)

        # Save credentials to kaggle.json
        with open(kaggle_json_path, "w") as f:
            json.dump(credentials, f)

        # os.chmod(kaggle_json_path, 0o600)  # Secure the file

    # Authenticate with Kaggle API
    api = KaggleApi()
    api.authenticate()
    print("Kaggle authentication successful!")

    return api


def is_valid_image(file_path: Path) -> bool:
    """Check if a file is a valid image using Pillow."""
    if file_path is None or not file_path.is_file():
        return False
    try:
        with Image.open(file_path) as img:
            img.verify()  # Verify if the image is not corrupted
        return True
    except Exception:
        return False


def preprocess_image(file_path: tf.Tensor, target_size) -> tf.Tensor:
    if isinstance(file_path, str) or isinstance(file_path, Path):
        file_path = tf.convert_to_tensor(str(file_path), dtype=tf.string)
    img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(img, channels=3)  # Decode directly as RGB
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, target_size)  # Resize lazily
    return img


class ImageBag:
    def __init__(
        self, base_path: Path, target_size: tuple[int, int] | list[int], batch_size=32
    ):
        if not base_path.exists():
            raise ValueError(f"Path {base_path} doesn't exist")

        self._base_path = base_path
        self._target_size = target_size
        self._batch_size = batch_size
        self._class_to_image_dict: dict[str, list[Path]] = dict()
        self._image_to_class_dict: dict[Path, str] = dict()

        for path in base_path.rglob("*"):
            if not path.is_file():
                continue

            components = path.relative_to(base_path).parts[:-1]
            if len(components) != 1:
                logging.debug(
                    "Unexpected relative path. Expected a single folder (=class), got %s, ignoring path %s",
                    components,
                    path,
                )
                continue

            components = components[0]
            if is_valid_image(path):
                if components in self._class_to_image_dict:
                    self._class_to_image_dict[components].append(path)
                else:
                    self._class_to_image_dict[components] = [path]
                self._image_to_class_dict[path] = components

        if logging.get_verbosity() >= logging.DEBUG:
            pprint.pprint(self._class_to_image_dict)

    def _get_image_file_pattern(self, base_path: Path) -> str:
        """Generate an iterable of file patterns for image files."""
        image_extensions = ["png", "jpg", "jpeg"]
        return [str(base_path / f"*.{ext}") for ext in image_extensions]
        # return str(base_path / "*.jpeg")

    @staticmethod
    def _get_dataset_length(dataset: tf.data.Dataset) -> np.int64:
        c = tf.data.experimental.cardinality(dataset).numpy()
        if c == tf.data.experimental.INFINITE_CARDINALITY:
            raise ValueError("Infinite dataset")

        if c == tf.data.experimental.UNKNOWN_CARDINALITY:
            c = dataset.reduce(np.int64(0), lambda x, _: x + 1)

        c = np.int64(c)
        return c

    def get_datasets(
        self,
    ) -> Tuple[
        Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset],
        Tuple[np.int64, np.int64, np.int64],
    ]:
        """Load datasets using image_dataset_from_directory and prepare triplets."""

        # Store file paths, not images
        image_path_ds_dict: dict[str, tf.data.Dataset] = {
            path.name: tf.data.Dataset.list_files(
                self._get_image_file_pattern(path), shuffle=False
            )
            for path in self._base_path.iterdir()
            if path.is_dir()
        }

        datasets: list[tf.data.Dataset] = []
        num_triplets = np.int64(0)
        for name, dataset in image_path_ds_dict.items():
            negative_name = random.choice(
                [k for k in image_path_ds_dict.keys() if k != name]
            )

            # Split into anchors and positives
            dataset_len = ImageBag._get_dataset_length(dataset)
            negative_len = ImageBag._get_dataset_length(
                image_path_ds_dict[negative_name]
            )
            num_triplets += np.int64(min(negative_len, dataset_len) // 2)
            anchors_ds, positive_ds = (
                dataset.take(dataset_len // 2),
                dataset.skip(dataset_len // 2),
            )
            negative_ds = image_path_ds_dict[negative_name].take(negative_len // 2)

            # Zip and map lazily
            # Nota come `map`, con il caricamento effettivo del dataset, e' l'ultima operazione. Concatenare operazioni di trasformazioni causa
            # eager evaluation
            triplet_ds = tf.data.Dataset.zip((anchors_ds, positive_ds, negative_ds))
            triplet_ds = triplet_ds.map(
                lambda a, p, n: [
                    preprocess_image(a, self._target_size),
                    preprocess_image(p, self._target_size),
                    preprocess_image(n, self._target_size),
                ]
            )

            datasets.append(triplet_ds)

        # Concatenate datasets
        full_ds = datasets[0]
        for ds in datasets[1:]:
            full_ds = full_ds.concatenate(ds)

        # TODO controlla shuffle seed se deterministic
        full_ds.shuffle(1000)

        # Get total dataset size
        total_size = ImageBag._get_dataset_length(full_ds)
        test_size = total_size // 10  # 10% for testing
        test_triplets = np.int64(num_triplets // 10)

        # Split deterministically
        test_ds = full_ds.take(test_size)
        remaining_ds = full_ds.skip(test_size)

        # Shuffle remaining data
        shuffled_ds = remaining_ds.shuffle(1000)  # Ensures random train/val split

        # Train (80% of total) & Validation (10% of total)
        train_size = (total_size * 8) // 10
        train_ds = shuffled_ds.take(train_size)
        train_triplets = np.int64(num_triplets * 8 // 10)
        val_ds = shuffled_ds.skip(train_size)
        val_triplets = np.int64(num_triplets // 10)

        # Batch, prefetch for performance
        def prepare(ds):
            return ds.batch(self._batch_size).prefetch(tf.data.AUTOTUNE)

        return (prepare(train_ds), prepare(val_ds), prepare(test_ds)), (
            np.int64(train_triplets // self._batch_size),
            np.int64(val_triplets // self._batch_size),
            np.int64(test_triplets // self._batch_size),
        )

    @property
    def image_count(self):
        return sum(len(v) for v in self._class_to_image_dict.values())


class CommandProcessor:
    def __init__(self, command_not_found_callback: Callable[[list[str]], None]) -> None:
        self._commands: dict[Tuple, Callable] = {}
        self._commands_invalid_syntax: dict[Tuple, Callable] = {}
        self._command_not_found_callback = command_not_found_callback

    def register_command(
        self,
        command: Tuple,
        func: Callable,
        signature: Tuple = (),
        invalid_syntax_callback: Callable[[], None] = lambda: (),
    ) -> None:
        if not callable(func):
            raise ValueError(f"Passed function {func} is not a callable")

        if (
            not callable(invalid_syntax_callback)
            or len(inspect.signature(invalid_syntax_callback).parameters) != 0
        ):
            raise ValueError(
                f"Passed invalid callback {invalid_syntax_callback} is not a zero param callable"
            )

        func_sig = inspect.signature(func)
        if len(func_sig.parameters) != len(signature):
            raise ValueError(
                f"Different number of parameters found between the given callback and the passed signature: {len(func_sig.parameters)} vs {len(signature)}"
            )

        for param_type in signature:
            if param_type not in (str, int, float):
                raise ValueError(
                    "Commands can only accept positional arguments of type str, int, or float."
                )

        self._commands[command] = func
        self._commands_invalid_syntax[command] = invalid_syntax_callback

    @property
    def available_commands(self) -> list[Tuple]:
        return getattr(self, "_commands", {}).keys()

    def _convert_arguments(
        self, given_args: List[str], expected_types: List[Union[type, None]]
    ) -> List[Union[str, int, float]]:
        converted_args = []
        for arg, expected_type in zip(given_args, expected_types):
            if expected_type is int:
                try:
                    converted_args.append(int(arg))
                except ValueError:
                    return []
            elif expected_type is float:
                try:
                    converted_args.append(float(arg))
                except ValueError:
                    return []
            else:
                converted_args.append(arg)
        return converted_args

    def _check_signature_validity_and_call(
        self,
        given_signature: List[str],
        inspected_signature: inspect.Signature,
        func: Callable,
    ):
        expected_types = [
            param.annotation if param.annotation in (str, int, float) else str
            for param in inspected_signature.parameters.values()
        ]
        converted_args = self._convert_arguments(given_signature, expected_types)
        if len(converted_args) != len(given_signature):
            return False

        try:
            bound_args = inspected_signature.bind(*converted_args)
            bound_args.apply_defaults()
            func(*bound_args.args)
            return True
        except TypeError as e:
            logging.debug(f"Argument mismatch: {e}")
            return False

    def __call__(self, command_line: str) -> None:
        parts = shlex.split(command_line.strip())
        if not parts:
            return

        command, *args = parts
        func_list = [v for k, v in self._commands.items() if command in k]
        invalid_cb_list = [
            v for k, v in self._commands_invalid_syntax.items() if command in k
        ]
        if len(func_list) == 0:
            self._command_not_found_callback(self._commands.keys())
        else:
            func = func_list[0]
            invalid_cb = invalid_cb_list[0]
            func_sig = inspect.signature(func)
            if not self._check_signature_validity_and_call(args, func_sig, func):
                invalid_cb()


class SaveModelCallback(tf.keras.callbacks.Callback):
    def __init__(self, model: tf.keras.Model, base_path: Path, name_pattern: str):
        super().__init__()
        self._base_path = base_path
        self._name_pattern = name_pattern
        try:
            name_pattern.format(epoch=0)
        except KeyError:
            raise KeyError(
                f"name pattern {name_pattern} doesn't contain formatting key 'epoch'"
            )
        self._model = model

    def on_epoch_end(self, epoch, logs=None):
        path = self._base_path / self._name_pattern.format(epoch=epoch)
        logging.info("Saving model for epoch %d to %s", epoch, str(path))
        self._model.save(path, save_format="tf")


def binary_prompt(prompt: str = "Use existing? (Y/N): ") -> bool:
    use = ""
    while use not in ["y", "n", "yes", "no"]:
        use = input(prompt).lower()
    if use == "y":
        return True
    else:
        return False


def save_dict_to_json(data: dict[str, Any], filepath: str | Path) -> None:
    """Save a dictionary to a JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def load_dict_from_json(filepath: str | Path) -> dict[str, Any]:
    """Load a dictionary from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def ask_filenames(prompts: list[str]):
    directories = []
    if platform.system() != "Linux":
        from tkinter import filedialog, Tk

        root = Tk()
        for i in range(len(prompts)):
            dir_path = ""
            while not is_valid_image(Path(dir_path)):
                dir_path = filedialog.askopenfilename(title=prompts[i])
            directories.append(Path(dir_path))
        root.withdraw()
    else:  # PyQt6 alternative for Linux
        from PyQt6.QtWidgets import QApplication, QFileDialog
        from PyQt6.QtCore import QTimer

        app = QApplication([])

        for prompt in prompts:
            dir_path = ""
            if prompt == prompts[-1]:
                logging.debug("setQuitOnLastWindowClosed")
                app.setQuitOnLastWindowClosed(True)

            while not is_valid_image(Path(dir_path)):
                dir_path, _ = QFileDialog.getOpenFileName(
                    None, prompt, "", "Images (*.png *.jpg *.jpeg *.bmp *.gif)"
                )

            if dir_path:
                directories.append(Path(dir_path))

        # run a.quit after 100ms
        QTimer.singleShot(100, app.quit)
        # run mainloop
        app.exec()

    return directories


# bazel run //siamese_second  -- --action=train --working-directory="Y:\machine-learning-data\"
def main(args: list[str]) -> None:
    # assicurati che Abseil default logger stia usando stdout
    logging.get_absl_handler().stream = sys.stdout

    matplotlib.use("QtAgg")
    logging.info("Hello World! %s", args)
    logging.info(
        "CUDA Capable devices detected by tensorflow: %s",
        tf.config.list_physical_devices("GPU"),
    )
    random.seed()
    conf = AppConfig(
        data_path=Path(_FLAGS[EArgsName.working_directory.value].value),
        seed=_FLAGS[EArgsName.working_directory.value]
        if _FLAGS[EArgsName.working_directory.value] is not None
        else random.randint(),
    )
    if _FLAGS.action is None or _FLAGS.action == EActions.help.value:
        print_filtered_help()
        return
    if _FLAGS.action == EActions.train.value:
        api = authenticate_kaggle()
        data_path = conf.data_path / "animals"
        should_download_kaggle_dataset = True
        if data_path.exists():
            logging.info("Path %s Exists", data_path)
            if binary_prompt():
                should_download_kaggle_dataset = False

        if should_download_kaggle_dataset:
            api.dataset_download_files(
                "alessiocorrado99/animals10", path=data_path, quiet=False, unzip=True
            )
        # hardcoded finche usi un solo dataset
        classes_path = data_path / "raw-img"
        if not classes_path.exists():
            raise ValueError(f"Classes Path {classes_path} doesn't exist")

        target_size = (224, 224)

        # load or create datasets
        dataset_paths = [
            conf.data_path / "siamese_second_train_data",
            conf.data_path / "siamese_second_val_data",
            conf.data_path / "siamese_second_test_data",
            conf.data_path / "siamese_second_cardinalities",
        ]
        process_datasets = True
        if all([p.exists() and p.is_dir() for p in dataset_paths]):
            logging.info("Found datasets in paths %s", dataset_paths)
            if binary_prompt():
                process_datasets = False

        if process_datasets:
            image_bag = ImageBag(classes_path, target_size)
            (
                (train_dataset, val_dataset, test_dataset),
                (train_numbatches, val_numbatches, test_numbatches),
            ) = image_bag.get_datasets()
            if binary_prompt("Save Datasets to disk? (Y/N) "):
                logging.info("Saving Train dataset to %s", dataset_paths[0])
                dataset_paths[0].mkdir(parents=True, exist_ok=True)
                tf.data.Dataset.save(train_dataset, str(dataset_paths[0]))

                logging.info("Saving Train dataset to %s", dataset_paths[1])
                dataset_paths[1].mkdir(parents=True, exist_ok=True)
                tf.data.Dataset.save(val_dataset, str(dataset_paths[1]))

                logging.info("Saving Train dataset to %s", str(dataset_paths[2]))
                dataset_paths[2].mkdir(parents=True, exist_ok=True)
                tf.data.Dataset.save(test_dataset, str(dataset_paths[2]))

                logging.info("Saving dataset cardinalities to %s", dataset_paths[3])
                dataset_paths[3].mkdir(parents=True, exist_ok=True)
                save_dict_to_json(
                    {
                        "train": train_numbatches,
                        "val": val_numbatches,
                        "test": test_numbatches,
                    },
                    dataset_paths[3] / "cardinalities.json",
                )
        else:
            train_dataset, val_dataset, test_dataset = [
                tf.data.Dataset.load(str(p)) for p in dataset_paths[:-1]
            ]
            train_numbatches, val_numbatches, test_numbatches = load_dict_from_json(
                dataset_paths[3] / "cardinalities.json"
            ).values()

        if binary_prompt("Want to display Datasets number of batches? (Y/N): "):
            print(f"Train: {train_numbatches}")
            print(f"Val:   {val_numbatches}")
            print(f"Test:  {test_numbatches}")

        learning_rate = 1e-4
        logging.info("Building siamese model for target size %s", target_size)

        model_path = conf.data_path / "siamese_second_model_saves"
        siamese_network = None
        trained = False
        if binary_prompt("Load Model?(Y/N): "):
            trained = True
            siamese_network_loaded = tf.keras.models.load_model(
                model_path,
                custom_objects={
                    "DistanceLayer": DistanceLayer,
                    "SiameseModel": SiameseModel,
                    "Mean": keras.metrics.Mean,  # Explicitly add Mean metric
                    "maximum": tf.maximum,  # If TensorFlow operations need registering
                },
            )
            siamese_network = SiameseModel(target_size)
            siamese_network.set_weights(siamese_network_loaded.get_weights())
        else:
            siamese_network = SiameseModel(target_size)
            # siamese_network.build(target_size + (3,))

        dummy_input = [tf.zeros((1, *target_size, 3))] * 3  # Batch size = 1
        siamese_network(dummy_input)  # Calls forward pass to define input shape
        logging.info(
            "Compiling siamese model With Adam Optimizer, learning rate = %.4g",
            learning_rate,
        )
        siamese_network.compile(optimizer=tf.keras.optimizers.Adam(learning_rate))

        def on_train_invalid() -> None:
            logging.debug("on_train_invalid")
            print("Syntax: train <epochs: int>")

        def on_train(epochs: int) -> None:
            # TODO add save callback
            logging.debug("on_train %d", epochs)
            nonlocal siamese_network, trained, conf, model_path
            nonlocal train_dataset, val_dataset, test_dataset

            if trained and not binary_prompt("Already Fit. Train Again? (Y/N): "):
                return

            callbacks = []

            if not model_path.exists():
                model_path.mkdir(parents=True)
            elif not model_path.is_dir():
                raise ValueError(f'Model Path "{str(model_path)}" is not a directory')

            model_checkpoint_path = (
                str(model_path)
                if platform.system() == "Windows"
                else str(model_path / "model.keras")
            )

            callbacks.extend(
                [
                    tf.keras.callbacks.ModelCheckpoint(
                        filepath=model_checkpoint_path,
                        save_weights_only=False,
                        verbose=1,
                    )
                    # SaveModelCallback(siamese_network, model_path, "model_{epoch}.tf"),
                    # tf.keras.callbacks.ProgbarLogger(),
                    # tf.keras.callbacks.History(),
                ]
            )
            siamese_network.fit(
                train_dataset,
                epochs=epochs,
                validation_data=val_dataset,
                callbacks=callbacks,
            )
            trained = True

        def on_visualize_invalid() -> None:
            logging.debug("on_visualize_invalid")
            print(
                'Syntax: visualize <dataset_name: "train"|"val"|"test"> <batch_index: int>'
            )

        def on_visualize(dataset_name: str, batch_index: int) -> None:
            logging.debug('on_visualize "%s" %d', dataset_name, batch_index)
            nonlocal train_dataset, val_dataset, test_dataset, conf
            nonlocal train_numbatches, val_numbatches, test_numbatches

            def visualize(
                dataset: tf.data.Dataset, batch: int, dataset_numbatches: np.int64
            ) -> None:
                if batch >= dataset_numbatches:
                    logging.info(
                        "dataset only has %d batches, requested %d",
                        dataset_numbatches,
                        batch,
                    )
                else:
                    logging.info("Close the upcoming window to go on")
                    siamese_visualize(
                        *next(dataset.skip(batch).take(1).as_numpy_iterator())
                    )

            match dataset_name:
                case "train":
                    visualize(train_dataset, batch_index, train_numbatches)
                case "val":
                    visualize(val_dataset, batch_index, val_numbatches)
                case "test":
                    visualize(test_dataset, batch_index, test_numbatches)
                case _:
                    logging.warn('Unrecognized dataset %s, using "train"', dataset_name)
                    visualize(train_dataset, batch_index)

        should_quit = False

        def on_quit() -> None:
            logging.debug("on_quit")
            nonlocal should_quit
            should_quit = True

        def on_inference() -> None:
            logging.debug("on_inference")
            nonlocal siamese_network, target_size
            image_paths = ask_filenames(["anchor", "positive", "negative"])
            if logging.get_verbosity() >= logging.INFO:
                for num, path in enumerate(image_paths):
                    logging.info("image path %d: %s", num, str(path))

            anchor = tf.expand_dims(
                preprocess_image(image_paths[0], target_size), axis=0
            )
            positive = tf.expand_dims(
                preprocess_image(image_paths[1], target_size), axis=0
            )
            negative = tf.expand_dims(
                preprocess_image(image_paths[2], target_size), axis=0
            )

            ap_distance, an_distance = siamese_network([anchor, positive, negative])
            anchor_embedding, positive_embedding, negative_embedding = (
                siamese_network.embedding(anchor),
                siamese_network.embedding(positive),
                siamese_network.embedding(negative),
            )
            logging.info(
                "AP Distance: %f, AN Distance: %f",
                ap_distance.numpy(),
                an_distance.numpy(),
            )

            cosine_similarity = tf.keras.metrics.CosineSimilarity()
            positive_similarity = cosine_similarity(
                anchor_embedding, positive_embedding
            )
            negative_similarity = cosine_similarity(
                anchor_embedding, negative_embedding
            )
            logging.info(
                "Positive Similarity: %f, Negative Similarity: %f",
                positive_similarity.numpy(),
                negative_similarity.numpy(),
            )

        def on_test_invalid() -> None:
            logging.debug("on_test_invalid")
            print("Syntax: test")

        def on_test() -> None:
            logging.debug("on_test")
            nonlocal siamese_network, target_size
            nonlocal test_dataset
            # Evaluate the model on the test dataset
            results = siamese_network.evaluate(test_dataset)

            # Log and print results
            logging.info("Test Loss: %f", results)
            print(f"Test Results: Loss = {results}")

        command_processor = CommandProcessor(
            lambda valid_commands: print(f"Invalid command. Commands: {valid_commands}")
        )
        command_processor.register_command(
            ("train", "t"), on_train, (int,), on_train_invalid
        )
        command_processor.register_command(
            ("visualize", "v"), on_visualize, (str, int), on_visualize_invalid
        )
        command_processor.register_command(("quit", "exit", "q"), on_quit)
        command_processor.register_command(("inference", "i"), on_inference)
        command_processor.register_command(("test", "te"), on_test, (), on_test_invalid)

        print(f"Available Commands: {command_processor.available_commands}")
        while not should_quit:
            # input non supporta history di comandi. serve `readline` (linux) o `pyreadline` (windows)
            command_processor(input("insert a command: "))


if __name__ == "__main__":
    app.run(main)
