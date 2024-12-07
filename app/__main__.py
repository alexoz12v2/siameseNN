import os
import absl
from absl import logging, app, flags
import tensorflow as tf
import keras
import utils


def main(argv: list[str]) -> None:
    del argv
    logging.info("Hello World!")
    utils.set_keras_backend('tensorflow')
    logging.info(f"Active Keras Backend: {keras.backend.backend()}")
    logging.info(f"CUDA Devices: {tf.config.list_physical_devices('GPU')}")

    my_linear = utils.MLPBlock()
    num_batches = 10
    input = keras.random.normal(shape=(num_batches, 32))
    out = my_linear(input)
    logging.info(f"output from MLP: {tf.squeeze(out)}")
    logging.info(f"layer.losses:    {[tfloss.numpy().item() for tfloss in my_linear.losses]}")


if __name__ == "__main__":
    absl.app.run(main)
