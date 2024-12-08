from absl import logging
from absl import app
from pathlib import Path
import keras
import tensorflow as tf
import classification_from_scratch.class_utils as utils


def compressed_tree(dir_path: Path, prefix: str = ""):
    # prefix components:
    space = "    "
    branch = "│   "
    # pointers:
    tee = "├── "
    last = "└── "

    contents = list(dir_path.iterdir())
    directories = [path for path in contents if path.is_dir()]
    files = [path for path in contents if path.is_file()]

    # List directories as usual
    for idx, path in enumerate(directories):
        pointer = tee if idx < len(directories) - 1 else last
        yield prefix + pointer + path.name
        yield from compressed_tree(
            path, prefix=prefix + (branch if pointer == tee else space)
        )

    # Compress files into a file count
    if files:
        # File count entry
        file_count = len(files)
        yield prefix + (tee if directories else "") + f"[{file_count} files]"


def tree(dir_path: Path, prefix: str = ""):
    # prefix components:
    space = "    "
    branch = "│   "
    # pointers:
    tee = "├── "
    last = "└── "
    contents = list(dir_path.iterdir())
    # contents each get pointers that are ├── with a final └── :
    pointers = [tee] * (len(contents) - 1) + [last]
    for pointer, path in zip(pointers, contents):
        yield prefix + pointer + path.name
        if path.is_dir():  # extend the prefix and recurse:
            extension = branch if pointer == tee else space
            # i.e. space because last, └── , above so no more |
            yield from tree(path, prefix=prefix + extension)


def main(argv: list[str]) -> None:
    del argv
    logging.info("Hello World")
    logging.info(f"cwd: {Path.cwd()}")
    data_path = Path.cwd() / "classification_from_scratch" / "extracted_files"
    for line in compressed_tree(data_path):
        logging.info(line)

    # hello
    utils.set_keras_backend("tensorflow")
    keras.utils.set_random_seed(812)
    tf.config.experimental.enable_op_determinism()
    logging.info(f"Active Keras Backend: {keras.backend.backend()}")
    logging.info(f"CUDA Devices: {tf.config.list_physical_devices('GPU')}")

    # prova a caricare il dataset
    dataset, dataset_len = utils.custom_image_dataset_from_directory(
        data_path / "PetImages",
        use_seed=True,
    )
    dataset_numbatches = dataset_len // 32
    logging.info(dataset)
    logging.info(f"images count: {dataset_len}")

    # splittalo in training e validation
    validation_size = int(0.2 * dataset_numbatches)

    # splitta e fai shuffling del solo training set
    dataset_val = dataset.take(validation_size)
    dataset_train = dataset.skip(validation_size)
    dataset_train = dataset_train.shuffle(
        buffer_size=dataset_len, reshuffle_each_iteration=True
    )

    # visualizza 9 immagini
    utils.visualize_first_9_images(dataset_train)


if __name__ == "__main__":
    app.run(main)
