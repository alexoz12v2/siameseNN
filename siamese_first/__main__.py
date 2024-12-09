from absl import logging
from absl import app
from pathlib import Path
from PyQt6.QtWidgets import QApplication, QWidget
import keras
import tensorflow as tf
import siamese_first.siamese_lib as utils
import matplotlib.pyplot as plt
import matplotlib
import pydot
import numpy as np
from matplotlib.axes import Axes
from tensorflow.keras.preprocessing.image import ImageDataGenerator


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


def main(argv: list[str]) -> None:
    del argv
    logging.info("Hello World")
    logging.info(f"cwd: {Path.cwd()}")
    # hello keras
    utils.set_keras_backend("tensorflow")
    keras.utils.set_random_seed(812)
    tf.config.experimental.enable_op_determinism()
    logging.info(f"Active Keras Backend: {keras.backend.backend()}")
    logging.info(f"CUDA Devices: {tf.config.list_physical_devices('GPU')}")

    batch_size = 8
    target_shape = (200, 200)

    # paths: I dataset left e right hanno immagini numerate jpg. le coppie aventi lo
    # stesso numero corrispondono alla stessa persona
    base_path = Path.cwd() / 'siamese_first'
    siamese_left_path = base_path / 'siamese_left' / 'left'
    siamese_right_path = base_path / 'siamese_right' / 'right'

    dataset_train, dataset_val = utils.triplet_datasets(siamese_left_path, siamese_right_path, batch_size=batch_size, seed=812, target_shape=target_shape)
    logging.info("showing 9 images from the dataset (Close the window to continue)...")

    # dataset_train.take(1) -> primo batch
    # it =dataset_train.take(1).as_numpy_iterator() -> NumpyIterator di una terna di buffer 3D
    # list(it) -> accumula tutti gli elementi dell'iteratore in una lista
    # (*list(it)[0], ) -> prendi la prima terna del batch ed espandila in una tupla
    utils.visualize(*(next(dataset_train.take(1).as_numpy_iterator())))

    logging.info("creating siamese model and outputting its png structure to working dir as \"siamese.png\"...")
    siamese_model = utils.SiameseModel(target_shape)
    siamese_model.build(target_shape + (3,))
    keras.utils.plot_model(siamese_model, show_shapes=True, to_file="siamese.png")
    siamese_model.compile(optimizer=keras.optimizers.Adam(1e-4))
    siamese_model.fit(dataset_train.take(20), epochs=1, validation_data=dataset_val)

    logging.info("Training complete! Picking up a sample from the val dataset (Close the window to comtinue)")
    sample = next(iter(dataset_val))
    utils.visualize(*sample, go_on=True)
    anchor_emb, positive_emb, negative_emb = (
        siamese_model.embedding(keras.applications.resnet.preprocess_input(sample[0])),
        siamese_model.embedding(keras.applications.resnet.preprocess_input(sample[1])),
        siamese_model.embedding(keras.applications.resnet.preprocess_input(sample[2])),
    )

    logging.info("measuring cosine similarity anchor-positive and anchor-negative")
    cosine_similarity = keras.metrics.CosineSimilarity()
    positive_similarity = cosine_similarity(anchor_emb, positive_emb)
    negative_similarity = cosine_similarity(anchor_emb, negative_emb)
    logging.info(f"Positive Similarity: {positive_similarity.numpy()}")
    logging.info(f"Negative Similarity: {negative_similarity.numpy()}")
    input("press any key to exit...")


if __name__ == "__main__":
    app.run(main)
