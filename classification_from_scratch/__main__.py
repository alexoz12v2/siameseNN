from absl import logging
from absl import app
from pathlib import Path
from PyQt6.QtWidgets import QApplication, QWidget
import keras
import tensorflow as tf
import classification_from_scratch.class_utils as utils
import matplotlib.pyplot as plt
import matplotlib


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

            
@tf.function
def calculate_validation_size(dataset_numbatches: tf.Tensor) -> tf.Tensor:
    validation_size = tf.cast(0.2 * tf.cast(dataset_numbatches, tf.float32), tf.int64)
    return validation_size


def main(argv: list[str]) -> None:
    del argv
    logging.info("Hello World")
    logging.info(f"cwd: {Path.cwd()}")
    data_path = Path.cwd() / "classification_from_scratch" / "extracted_files"
    for line in compressed_tree(data_path):
        logging.info(line)

    # hello
    matplotlib.use('QtAgg')
    utils.set_keras_backend("tensorflow")
    keras.utils.set_random_seed(812)
    tf.config.experimental.enable_op_determinism()
    logging.info(f"Active Keras Backend: {keras.backend.backend()}")
    logging.info(f"CUDA Devices: {tf.config.list_physical_devices('GPU')}")

    # prova a caricare il dataset
    batch_size = 32
    #dataset, dataset_len = utils.custom_image_dataset_from_directory(
    #    data_path / "PetImages",
    #    use_seed=True,
    #)
    #dataset_numbatches = dataset_len // batch_size
    #logging.info(dataset)
    #logging.info(f"images count: {dataset_len}")

    ## splittalo in training e validation
    ## se aggiungi tf.function in custom_image_dataset_from_directory, il che non riesce
    ## a velocizzare molto perche usa generators, allora devi racchiudere anche il calcolo
    ## del validation size perche nel grafo di computazione il cast da int a float deve
    ## essere esplicitato
    ## validation_size = int(0.2 * dataset_numbatches)
    #validation_size = calculate_validation_size(dataset_numbatches)

    ## splitta e fai shuffling del solo training set
    #dataset_val: tf.data.Dataset = dataset.take(validation_size)
    #dataset_train: tf.data.Dataset = dataset.skip(validation_size)
    input_path = data_path / "PetImages"
    output_path = data_path.parent / "cats_and_dogs"
    utils.select_valid_images(input_path, output_path, True)
    dataset_train, dataset_val = keras.utils.image_dataset_from_directory(
        str(data_path / "PetImages"),
        validation_split=0.2,
        subset="both",
        image_size=(256, 256),
        seed=32,
        batch_size=batch_size,
    )

    logging.info("Dataset for training and validation created.")
    logging.info(f"\tTrain:      {dataset_train.cardinality()}")
    logging.info(f"\tValidation: {dataset_val.cardinality()}")

    # logging.info("Enabling per training step shuffling of the training set..")
    # https://stackoverflow.com/questions/46444018/meaning-of-buffer-size-in-dataset-map-dataset-prefetch-and-dataset-shuffle
    # dataset_train = dataset_train.shuffle(
    #     buffer_size=batch_size, reshuffle_each_iteration=True
    # )

    # visualizza 9 immagini
    utils.visualize_first_9_images(dataset_train, transpose=False)
    plt.show(block=True)


if __name__ == "__main__":
    app.run(main)
