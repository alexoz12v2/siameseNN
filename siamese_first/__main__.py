from pathlib import Path

import keras
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pydot
import tensorflow as tf
from absl import app, flags, logging
from matplotlib.axes import Axes
from PyQt6.QtWidgets import QApplication, QWidget
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import siamese_first.siamese_lib as utils

# from argparse import ArgumentParser


FLAGS = flags.FLAGS
flags.DEFINE_boolean(
    "contrastive-loss", False, "Choose contrastive loss instead of triplet loss"
)
flags.DEFINE_boolean(
    "fast-train", False, "Train the network on a tiny portion of the dataset"
)


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
    logging.info("Hello World")
    logging.info(f"cwd: {Path.cwd()}")
    # hello keras
    utils.set_keras_backend("tensorflow")
    keras.utils.set_random_seed(812)
    tf.config.experimental.enable_op_determinism()
    logging.info(f"Active Keras Backend: {keras.backend.backend()}")
    logging.info(f"CUDA Devices: {tf.config.list_physical_devices('GPU')}")

    # usa absl, non argparse
    # parser = ArgumentParser()
    # parser.add_argument("-c", "--contrastive-loss", action="store_true")
    # parser.add_argument("-f", "--fast-train", action="store_true")
    # args = parser.parse_args(argv[1:]) # e' il default
    # logging.info(f"you passed --contrastive-loss={args.contrastive_loss}, --fast-train={args.fast_train}")
    logging.info(
        f"you passed flags: {'--contrastive-loss' if FLAGS['contrastive-loss'].value else ''} {'--fast-train' if FLAGS['fast-train'].value else ''}"
    )

    batch_size = 8
    epochs = 3
    margin = 1

    # if args.contrastive_loss:
    if FLAGS["contrastive-loss"].value:
        clamp_num = 30 if FLAGS["fast-train"].value else None
        dataset = utils.download_mnist()
        logging.info(
            "Visualising 4 image pairs from MNIST (Close the window to continue)..."
        )
        utils.mnist_visualize(dataset.train, go_on=False, to_show=4, num_col=2)
        siamese_model = utils.contrastive_siamese_model()
        siamese_model.compile(
            loss=utils.contrastive_loss(margin=margin),
            optimizer="RMSprop",
            metrics=["accuracy"],
        )
        siamese_model.summary()
        # esempio di history data
        # history = {
        #    "loss": [1.0, 0.8, 0.6, 0.4],
        #    "val_loss": [1.2, 1.0, 0.8, 0.5],
        #    "accuracy": [0.5, 0.6, 0.7, 0.8],
        #    "val_accuracy": [0.4, 0.5, 0.6, 0.7],
        # }
        history = siamese_model.fit(
            [dataset.train.pairs[:clamp_num, 0], dataset.train.pairs[:clamp_num, 1]],
            dataset.train.labels[:clamp_num],
            validation_data=(
                [dataset.val.pairs[:clamp_num, 0], dataset.val.pairs[:clamp_num, 1]],
                dataset.val.labels[:clamp_num],
            ),
            batch_size=batch_size,
            epochs=epochs,
        )
        # utils.plt_metric(history=history.history, metric="accuracy", title="Model Accuracy")
        # utils.plt_metric(history=history.history, metric="loss", title="Contrastive Loss")
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        utils.add_metric_plot(
            history.history,
            metric="loss",
            title="Model Loss",
            ax=axs[0],
            has_valid=True,
        )
        utils.add_metric_plot(
            history.history,
            metric="accuracy",
            title="Model Accuracy",
            ax=axs[1],
            has_valid=True,
        )
        plt.show(block=False)
    else:
        target_shape = (200, 200)

        # paths: I dataset left e right hanno immagini numerate jpg. le coppie aventi lo
        # stesso numero corrispondono alla stessa persona
        base_path = Path.cwd() / "siamese_first"
        siamese_left_path = base_path / "siamese_left" / "left"
        siamese_right_path = base_path / "siamese_right" / "right"

        dataset_train, dataset_val = utils.triplet_datasets(
            siamese_left_path,
            siamese_right_path,
            batch_size=batch_size,
            seed=812,
            target_shape=target_shape,
        )
        logging.info(
            "showing 9 images from the dataset (Close the window to continue)..."
        )

        # dataset_train.take(1) -> primo batch
        # it =dataset_train.take(1).as_numpy_iterator() -> NumpyIterator di una terna di buffer 3D
        # list(it) -> accumula tutti gli elementi dell'iteratore in una lista
        # (*list(it)[0], ) -> prendi la prima terna del batch ed espandila in una tupla
        utils.visualize(*(next(dataset_train.take(1).as_numpy_iterator())))

        logging.info(
            'creating siamese model and outputting its png structure to working dir as "siamese.png"...'
        )
        siamese_model = utils.SiameseModel(target_shape)
        siamese_model.build(target_shape + (3,))
        keras.utils.plot_model(siamese_model, show_shapes=True, to_file="siamese.png")
        siamese_model.compile(optimizer=keras.optimizers.Adam(1e-4))
        siamese_model.fit(
            dataset_train.take(20) if FLAGS["fast-train"].value else dataset_train,
            epochs=epochs,
            validation_data=dataset_val,
        )

        logging.info(
            "Training complete! Picking up a sample from the val dataset (Close the window to comtinue)"
        )
        sample = next(iter(dataset_val))
        utils.visualize(*sample, go_on=True)
        anchor_emb, positive_emb, negative_emb = (
            siamese_model.embedding(
                keras.applications.resnet.preprocess_input(sample[0])
            ),
            siamese_model.embedding(
                keras.applications.resnet.preprocess_input(sample[1])
            ),
            siamese_model.embedding(
                keras.applications.resnet.preprocess_input(sample[2])
            ),
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
