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


if __name__ == "__main__":
    app.run(main)
