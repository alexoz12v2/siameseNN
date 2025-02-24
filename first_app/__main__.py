import absl
from absl import logging
import utils
import tensorflow as tf
import keras
import first_app.apputils as apputils  # tutte le path sono relative a WORKSPACE


def synthetic_dataset() -> dict[str, tf.Tensor]:
    # setta seed per reproducib ility https://keras.io/examples/keras_recipes/reproducibility_recipes/
    keras.utils.set_random_seed(812)

    # riduci performance GPU per rendere i calcoli floating points piu precisi, quindi piu riproducibili
    # su Tensorflow 2.10 GPU crasha
    # tf.config.experimental.enable_op_determinism()

    # Parameters
    num_samples_train: int = 5000
    num_samples_val: int = 1000
    num_features: int = 32
    num_classes: int = 10

    # Random training data
    x_train: tf.Tensor = tf.random.normal(shape=[num_samples_train, num_features])
    y_train: tf.Tensor = tf.random.uniform(
        shape=[num_samples_train], minval=0, maxval=num_classes, dtype=tf.int32
    )

    # Random validation data
    x_val: tf.Tensor = tf.random.normal(shape=[num_samples_val, num_features])
    y_val: tf.Tensor = tf.random.uniform(
        shape=[num_samples_val], minval=0, maxval=num_classes, dtype=tf.int32
    )

    # one-hot encoding -> NO. le loss functions di keras vogliono l'argmax
    # y_train_one_hot: tf.Tensor = tf.keras.utils.to_categorical(y_train, num_classes)
    # y_val_one_hot: tf.Tensor = tf.keras.utils.to_categorical(y_val, num_classes)

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
    }


def main(argv: list[str]) -> None:
    del argv
    logging.info("Hello World!")
    apputils.keras_utils.set_keras_backend("tensorflow")
    logging.info(f"Active Keras Backend: {tf.keras.backend.backend()}")
    logging.info(f"CUDA Devices: {tf.config.list_physical_devices('GPU')}")

    my_linear = apputils.layers.MLPBlock()
    num_batches = 10
    input = tf.random.normal(shape=(num_batches, 32))
    out = my_linear(input)
    logging.info(f"output from MLP: {tf.squeeze(out)}")
    logging.info(
        f"layer.losses:    {[tfloss.numpy().item() for tfloss in my_linear.losses]}"
    )

    dataset = synthetic_dataset()
    logging.info(f"x_train: {dataset['x_train'].shape}")
    logging.info(f"y_train: {dataset['y_train'].shape}")
    logging.info(f"x_val:   {dataset['x_val'].shape}")
    logging.info(f"y_val:   {dataset['y_val'].shape}")

    trainer = apputils.layers.MLPTrainer()
    trainer.epochs = 100
    trainer.batch_size = 50
    trainer.train(
        my_linear,
        dataset["x_train"],
        dataset["y_train"],
        dataset["x_val"],
        dataset["y_val"],
    )


if __name__ == "__main__":
    absl.app.run(main)
