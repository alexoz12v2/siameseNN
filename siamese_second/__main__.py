from dataclasses import dataclass
import utils
import tensorflow as tf
import keras
from pathlib import Path
from typing import Callable, NamedTuple, Tuple
from enum import Enum
from absl import logging, app, flags
import textwrap
from kaggle.api.kaggle_api_extended import KaggleApi
from siamese_first.siamese_lib.layers import SiameseModel
import os
import json
import getpass
import random
from PIL import Image


@dataclass
class AppConfig:
    data_path: Path
    seed: int


class FlagHolderHelpPair(NamedTuple):
    holder: flags.FlagHolder[any]
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
    try:
        with Image.open(file_path) as img:
            img.verify()  # Verify if the image is not corrupted
        return True
    except Exception:
        return False


class ImageBag:
    def __init__(self, base_path: Path, target_size: tuple[int, int] | list[int]):
        if not base_path.exists():
            raise ValueError(f"Path ${base_path} doesn't exist")

        self._target_size = target_size
        self._all_images: list[Path] = []
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
            import pprint

            logging.debug("Path Dict: %s", pprint.pprint(self._class_to_image_dict))

    def _split_data(self):
        """Split data into 80% train, 10% validation, 10% test."""
        images = self._all_images.copy()
        random.shuffle(images)
        total = len(images)
        train_split = int(0.8 * total)
        val_split = int(0.9 * total)

        return (
            images[:train_split],
            images[train_split:val_split],
            images[val_split:],
        )

    def _generate_triplets(self, images):
        """Generate triplets (anchor, positive, negative)."""
        triplets = []

        for anchor in images:
            anchor_class = self._image_to_class_dict[anchor]
            positives = [
                img for img in self._class_to_image_dict[anchor_class] if img != anchor
            ]
            negatives = [
                img for img in images if self._image_to_class_dict[img] != anchor_class
            ]

            if positives and negatives:
                triplets.append(
                    (anchor, random.choice(positives), random.choice(negatives))
                )

        return triplets

    def _load_image(self, path):
        """Load and preprocess an image."""
        img = tf.io.read_file(str(path))
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, self._target_size)  # Adjust as needed
        img = img / 255.0  # Normalize
        return img

    def _triplet_generator(self, triplets):
        for anchor, positive, negative in triplets:
            yield (
                self._load_image(anchor),
                self._load_image(positive),
                self._load_image(negative),
            )

    def get_datasets(self) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Return TensorFlow datasets for training, validation, and testing."""
        train_images, val_images, test_images = self._split_data()

        train_triplets = self._generate_triplets(train_images)
        val_triplets = self._generate_triplets(val_images)
        test_triplets = self._generate_triplets(test_images)

        train_ds = (
            tf.data.Dataset.from_generator(
                lambda: self._triplet_generator(train_triplets),
                output_signature=(
                    tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
                    tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
                    tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
                ),
            )
            .batch(32, drop_remainder=True)
            .shuffle(1000)
            .prefetch(tf.data.AUTOTUNE)
        )

        val_ds = (
            tf.data.Dataset.from_generator(
                lambda: self._triplet_generator(val_triplets),
                output_signature=(
                    tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
                    tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
                    tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
                ),
            )
            .batch(32, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )

        test_ds = (
            tf.data.Dataset.from_generator(
                lambda: self._triplet_generator(test_triplets),
                output_signature=(
                    tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
                    tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
                    tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
                ),
            )
            .batch(32, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )

        return train_ds, val_ds, test_ds

    @property
    def image_count(self):
        return len(getattr(self, "_all_images", []))


class CommandProcessor:
    def __init__(self, command_not_found_callback: Callable[[list[str]], None]) -> None:
        self._commands: dict[Tuple, Callable[[], None]] = {}
        self._command_not_found_callback = command_not_found_callback

    def register_command(self, command: Tuple, func: Callable[[], None]) -> None:
        self._commands[command] = func

    @property
    def available_commands(self) -> list[Tuple]:
        return getattr(self, "_commands", {}).keys()

    def __call__(self, command: str) -> None:
        funcList = [v for k, v in self._commands.items() if command in k]
        if len(funcList) == 0:
            self._command_not_found_callback(self._commands.keys())
        else:
            funcList[0]()


def binary_prompt(prompt: str = "Use existing? (Y/N): ") -> bool:
    use = ""
    while use not in ["y", "n", "yes", "no"]:
        use = input(prompt).lower()
    if use == "y":
        return True
    else:
        return False


# bazel run //siamese_second  -- --action=train --working-directory="Y:\machine-learning-data\"
def main(args: list[str]) -> None:
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
        ]
        process_datasets = True
        if all([p.exists() and p.is_dir() for p in dataset_paths]):
            logging.info("Found datasets in paths %s", dataset_paths)
            if binary_prompt():
                process_datasets = False

        if process_datasets:
            image_bag = ImageBag(classes_path, target_size)
            train_dataset, val_dataset, test_dataset = image_bag.get_datasets()
            if binary_prompt("Save Datasets to disk? (Y/N) "):
                logging.info("Saving Train dataset to %s", dataset_paths[0])
                tf.data.Dataset.save(train_dataset, dataset_paths[0])
                logging.info("Saving Train dataset to %s", dataset_paths[1])
                tf.data.Dataset.save(val_dataset, dataset_paths[1])
                logging.info("Saving Train dataset to %s", dataset_paths[2])
                tf.data.Dataset.save(test_dataset, dataset_paths[2])
        else:
            train_dataset, val_dataset, test_dataset = [
                tf.data.Dataset.load(p) for p in dataset_paths
            ]

        logging.info("Building siamese model for target size %s", target_size)
        siamese_network = SiameseModel(target_size)
        siamese_network.build(target_size + (3,))

        def on_train() -> None:
            logging.debug("on_train")
            pass

        def on_visualize() -> None:
            logging.debug("on_visualize")
            pass

        should_quit = False

        def on_quit() -> None:
            logging.debug("on_quit")
            nonlocal should_quit
            should_quit = True

        command_processor = CommandProcessor(
            lambda valid_commands: print(f"Invalid command. Commands: {valid_commands}")
        )
        command_processor.register_command(("train", "t"), on_train)
        command_processor.register_command(("visualize", "v"), on_visualize)
        command_processor.register_command(("quit", "exit", "q"), on_quit)

        print(f"{command_processor.available_commands}")
        while not should_quit:
            command_processor(input("insert a command: "))


if __name__ == "__main__":
    app.run(main)
