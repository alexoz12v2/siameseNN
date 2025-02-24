import os
import re
import shutil
import traceback
from importlib import reload
from pathlib import Path
from typing import (
    BinaryIO,
    Callable,
    Generator,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import keras
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pydot
import tensorflow as tf
from absl import logging
from matplotlib.axes import Axes
from matplotlib.axis import Axis
from matplotlib.figure import Figure
from PIL import Image


def set_keras_backend(backend):
    if keras.backend.backend() != backend:
        os.environ["KERAS_BACKEND"] = backend
        reload(keras.backend)
        assert keras.backend.backend() == backend


# ------------------------------ TRIPLET LOSS ----------------------------------
# coppia di funzioni responsabili per preprocessare una tripletta di immagini
# https://keras.io/examples/vision/siamese_network/
def preprocess_image(
    filename: Union[str, tf.Tensor], target_shape: Union[list[int], Tuple[int, int]]
) -> tf.Tensor:
    image_bytes = tf.io.read_file(filename)
    # a quanto pare esplode in graph mode
    # image = tf.io.decode_image(image_bytes, channels=3) The tf.io.decode_image function may produce a tensor without a fully defined shape if the input is ambiguous
    image = tf.io.decode_jpeg(image_bytes, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, target_shape)
    return image


def preprocess_triplet(
    target_shape: Union[list[int], Tuple[int, int]],
) -> Callable[[str, str, str], Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
    def f(
        anchor: str, positive: str, negative: str
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        return (
            preprocess_image(anchor, target_shape),
            preprocess_image(positive, target_shape),
            preprocess_image(negative, target_shape),
        )

    return f


def triplet_datasets(
    anchor_images_path: Path,
    positive_images_path: Path,
    *,
    batch_size: int,
    target_shape: Tuple[int, int],
    seed: Optional[int],
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    # essendo numerate per corrispondenza numerica, le voglio ordinate
    # https://www.tensorflow.org/guide/data#consuming_sets_of_files
    # anchor_dataset = tf.data.Dataset.list_files(str(anchor_images_path / '*.jpg'))
    # positive_dataset = tf.data.Dataset.list_files(str(positive_images_path / '*.jpg'))

    # mescolo le immagini left e right, cosi non corrispondono piu, e le tratto come negative
    anchor_images = sorted(
        [str(f) for f in anchor_images_path.iterdir() if f.is_file()]
    )
    positive_images = sorted(
        [str(f) for f in positive_images_path.iterdir() if f.is_file()]
    )
    image_count = len(anchor_images)

    # creo un dataset di paths
    anchor_dataset = tf.data.Dataset.from_tensor_slices(anchor_images)
    positive_dataset = tf.data.Dataset.from_tensor_slices(positive_images)

    if seed is not None:
        rng = np.random.RandomState(seed=seed)
        rng.shuffle(anchor_images)
        rng.shuffle(positive_images)

    negative_images = anchor_images + positive_images
    np.random.RandomState(seed=32).shuffle(negative_images)

    negative_dataset = tf.data.Dataset.from_tensor_slices(negative_images)

    dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))
    dataset = dataset.shuffle(buffer_size=1024)

    # testing
    # for anchor, positive, negative in dataset.take(1):
    #     preprocess_image(tf.constant(anchor, dtype=tf.string), target_shape=target_shape)

    # adesso apriamo tutte le immagini
    dataset = dataset.map(preprocess_triplet(target_shape=target_shape))

    # splitto il dataset in training e val 80% 20% e faccio batching e prefetching
    train_dataset = dataset.take(round(image_count * 0.8))
    val_dataset = dataset.skip(round(image_count * 0.8))

    # drop remainder a True permette di silenziare il warning END_OF_SEQUENCE
    return (
        train_dataset.batch(batch_size=batch_size, drop_remainder=True).prefetch(
            tf.data.AUTOTUNE
        ),
        val_dataset.batch(batch_size=batch_size, drop_remainder=True).prefetch(
            tf.data.AUTOTUNE
        ),
    )


# list dims: batch width height channels
def visualize(
    anchor: list[list[list[list[float]]]],
    positive: list[list[list[list[float]]]],
    negative: list[list[list[list[float]]]],
    *,
    go_on=False,
) -> None:
    Axes3x3 = Tuple[
        Tuple[Axes, Axes, Axes], Tuple[Axes, Axes, Axes], Tuple[Axes, Axes, Axes]
    ]

    def show(ax: Axes, image: list[list[list[list[float]]]]) -> None:
        ax.imshow(image)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    fig = plt.figure(figsize=(9, 9))
    axs: Axes3x3 = fig.subplots(3, 3)
    for i in range(3):
        show(axs[i, 0], anchor[i])
        show(axs[i, 1], positive[i])
        show(axs[i, 2], negative[i])
    plt.show(block=not go_on)


# definizione di una rete gemella: Prendi il modello resnet 50, preallenato su
# resnet, togliendogli la testa fully connected, e gli appiccico dei
# layers fully connected per dare in output un embedding dell'immagine
# inoltre, rendi allenabili soltanto dal layer convolutivo 5 in poi
# (transfer learning).
# il python package keras.application contiene i modelli prefatti
def triplet_embedding_model(target_shape: Tuple[int, int]) -> keras.Model:
    # nota che nella target shape includiamo anche le triplette
    # non ho bisogno di definire un keras.Input, perche a seconda della
    # input shape ResNet50 mi da a disposizione il tensore simbolico in output
    envvar = os.getenv("KERAS_HOME")
    logstrs = {
        "where": "KERAS_HOME" if envvar is not None else "${Home}/.keras",
        "value": f" = {envvar}" if envvar is not None else "",
    }
    logging.info(
        f"about to download ResNet50 imagenet's weights to {logstrs['where'] + logstrs['value']}"
    )
    base_cnn = keras.applications.resnet.ResNet50(
        weights="imagenet", input_shape=target_shape + (3,), include_top=False
    )

    flatten = keras.layers.Flatten()(base_cnn.output)
    dense1 = keras.layers.Dense(units=512, activation="relu")(flatten)
    dense1 = keras.layers.BatchNormalization()(dense1)
    dense2 = keras.layers.Dense(units=256, activation="relu")(dense1)
    dense2 = keras.layers.BatchNormalization()(dense2)
    output = keras.layers.Dense(units=256)(dense2)  # logits

    embedding = keras.Model(base_cnn.input, output, name="Embedding")

    # setting up transfer learning: freezing
    trainable = False
    for layer in base_cnn.layers:
        if layer.name == "conv5_block1_out":
            trainable = True
        layer.trainable = trainable

    return embedding


# definizione della rete di distanza: A partire da una tripletta di embeddings
# prodotti dal triplet_embedding_model, calcola distanza euclidea tra
# anchor-positive, anchor-negative
class DistanceLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(
        self, anchor: tf.Tensor, positive: tf.Tensor, negative: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        ap_distance = tf.math.reduce_sum(tf.square(anchor - positive), axis=None)
        an_distance = tf.math.reduce_sum(tf.square(anchor - negative), axis=None)
        return (ap_distance, an_distance)


def triplet_siamese_model(target_shape: Tuple[int, int]) -> keras.Model:
    anchor_input = keras.layers.Input(name="anchor", shape=target_shape + (3,))
    positive_input = keras.layers.Input(name="positive", shape=target_shape + (3,))
    negative_input = keras.layers.Input(name="negative", shape=target_shape + (3,))

    embedding = triplet_embedding_model(target_shape)

    # preprocess, che si aspetta o tensore o numpy array con canale o alla fine o all inizio (dopo asse batch)
    # performa la standardizzazione (media nulla, std unitaria) del dataset (graph mode)
    distances = DistanceLayer()(
        embedding(keras.applications.resnet.preprocess_input(anchor_input)),
        embedding(keras.applications.resnet.preprocess_input(positive_input)),
        embedding(keras.applications.resnet.preprocess_input(negative_input)),
    )
    # nota come metto in lista i 3 input, ecco perche nella target shape aggiungo 3 alla fine
    siamese_network = keras.Model(
        inputs=[anchor_input, positive_input, negative_input],
        outputs=distances,
    )
    return siamese_network, embedding


# modello che prende la rete costruita con la functional API e ci aggiunge loss
# e quindi training step con loss, metric e optimizer
# applicazione del Trainer Pattern
class SiameseModel(keras.Model):
    _TupleType = Union[Tuple[tf.Tensor, tf.Tensor, tf.Tensor], list[tf.Tensor]]

    # L(A, P, N) = max(|f(A) - f(P)|^2 - |f(A)-f(N)|^2 + margin, 0)
    # ne calcolo la media di questo valore con keras.metrics.Mean
    def __init__(self, target_shape, margin=0.5):
        super().__init__()
        self.siamese_network, self.embedding = triplet_siamese_model(target_shape)
        self.margin = margin
        self.loss_tracker = keras.metrics.Mean(name="loss")

    def call(self, inputs: _TupleType) -> Tuple[tf.Tensor, tf.Tensor]:
        # input: tripletta di batch di immagini
        return self.siamese_network(inputs)

    def train_step(self, data: _TupleType) -> dict[str, float]:
        # train_step e' una funzione chiamata durante il model.fit(), nel quale
        # io ho disponibili nel self tutti i key params passati al model.compile()
        # come l'optimizer. Qui posso usare un gradient tape per memorizzare
        # tutti i calcoli del gradiente
        with tf.GradientTape() as tape:
            loss = self._compute_triplet_loss(data)  # esiste ma faccio override

        # recupero dalla loss tutti i gradienti associati ai trainable weights
        # della rete
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # applica la correzione ai parametri secondo l'optimizer specificato
        # a partire dai gradienti
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # fai update della running loss
        self.loss_tracker.update_state(loss)

        # il return deve essere un dict come specificato dalle metriche
        # https://keras.io/examples/keras_recipes/trainer_pattern/
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data: _TupleType) -> dict[str, float]:
        loss = self._compute_triplet_loss(data)

        # metrica, in model.predict() usata come metrica di performance
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    # mi aspetto un tensore con shape (batch, w, h, c)
    def _compute_triplet_loss(self, data: _TupleType) -> Union[float, tf.Tensor]:
        ap_distance, an_distance = self.siamese_network(data)

        # applico formula
        loss = tf.maximum(ap_distance - an_distance + self.margin, 0.0)
        return loss

    @property
    def metrics(self) -> list[keras.Metric]:
        # listo le metriche che ritorno nel training step affinche possano
        # essere resettate con `reset_states()`
        return [self.loss_tracker]


# --------------------------- CONTRASTIVE LOSS ---------------------------------


class MNISTDatasetPair(NamedTuple):
    pairs: npt.NDArray[np.float32]
    labels: npt.NDArray[np.float32]


class MNISTDatasetOutput(NamedTuple):
    train: MNISTDatasetPair
    val: MNISTDatasetPair
    test: MNISTDatasetPair


def download_mnist() -> MNISTDatasetOutput:
    # type annotations per IDE
    x_train_val: npt.NDArray[np.uint8]
    y_train_val: npt.NDArray[np.uint8]
    x_test: npt.NDArray[np.uint8]
    y_test: npt.NDArray[np.uint8]

    envvar = os.getenv("KERAS_HOME")
    logstrs = {
        "where": "KERAS_HOME" if envvar is not None else "${Home}/.keras",
        "value": f" = {envvar}" if envvar is not None else "",
    }
    logging.info(f"Downloading MNIST datset in {logstrs['where']} {logstrs['value']}")
    (x_train_val, y_train_val), (x_test, y_test) = keras.datasets.mnist.load_data()
    logging.info(
        f"downloaded MNIST. Are they NpzFiles? {isinstance(x_train_val, np.lib.npyio.NpzFile)}"
    )
    logging.info(
        f"downloaded MNIST. Are they array[dtype=uint8]? {isinstance(x_train_val, np.ndarray) and x_train_val.dtype == np.uint8}"
    )
    # to floating point
    x_train_val: npt.NDArray[np.float32] = x_train_val.astype(np.float32)
    x_test: npt.NDArray[np.float32] = x_test.astype(np.float32)

    # 50% dati a validation set
    x_train, x_val = x_train_val[:30000], x_train_val[30000:]
    y_train, y_val = y_train_val[:30000], y_train_val[30000:]
    del x_train_val, y_train_val

    # https://stackoverflow.com/questions/71109838/numpy-typing-with-specific-shape-and-datatype
    def make_pairs(
        x: npt.NDArray[np.float32], y: npt.NDArray[np.uint8]
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.uint8]]:
        num_classes = max(y) + 1
        # passa da un array di digits ad un array di array. Un array interno per ogni cifra
        # ciascuno degli array interni e' associato ad una cifra mediante il suo indice
        # ciascun array interno contiene gli indici riferiti all'array original in cui si trova la
        # cifra considerata. Stiamo essenzialmente facendo un istogramma sulle cifre (bins)
        # lo [0] serve perche np.where ritorna una tupla (caso multidim, ma l'array qui e' 1D)
        # >>> arr = np.array([2,3,4,2,1,2,3,3,3], dtype=np.uint8)
        # array([2, 3, 4, 2, 1, 2, 3, 3, 3], dtype=uint8)
        # >>> [np.where(arr == i) for i in range(max(arr))]
        # [(array([], dtype=int64),), (array([4]),), (array([0, 3, 5]),), (array([1, 6, 7, 8]),)]
        # >>> [np.where(arr == i)[0] for i in range(max(arr))]
        # [array([], dtype=int64), array([4]), array([0, 3, 5]), array([1, 6, 7, 8])]
        digit_indices = [np.where(y == i)[0] for i in range(num_classes)]

        pairs = []
        labels = []

        # per ogni immagine
        for idx1 in range(len(x)):
            # in ogni iterazione, aggiungiamo nella lista 2 coppie di immagini
            # la prima contiene   (immagine corrente, immagine a caso con stesso label)
            # la seconda contiene (immagine corrente, immagine a caso con label diverso)
            x1 = x[idx1]
            label1 = y[idx1]

            # prima coppia
            idx2 = np.random.choice(digit_indices[label1])
            x2 = x[idx2]

            pairs += [[x1, x2]]
            labels += [0]  # il label per due immagini uguali deve essere differenza = 0

            # seconda coppia
            label2 = np.random.randint(0, num_classes)
            while label2 == label1:
                label2 = np.random.randint(0, num_classes)

            idx2 = np.random.choice(digit_indices[label2])
            x2 = x[idx2]

            pairs += [[x1, x2]]
            labels += [
                1
            ]  # il label per due immagini diverse deve essere differenza = 1

        # metto i labels a float per permettere valori intermedi in training
        return np.array(pairs), np.array(labels).astype(np.float32)

    pairs_train, labels_train = make_pairs(x_train, y_train)
    pairs_val, labels_val = make_pairs(x_val, y_val)
    pairs_test, labels_test = make_pairs(x_test, y_test)

    return MNISTDatasetOutput(
        train=MNISTDatasetPair(
            pairs=pairs_train,
            labels=labels_train,
        ),
        val=MNISTDatasetPair(
            pairs=pairs_val,
            labels=labels_val,
        ),
        test=MNISTDatasetPair(
            pairs=pairs_test,
            labels=labels_test,
        ),
    )


def mnist_visualize(
    dataset_pairs: MNISTDatasetPair,
    *,
    go_on: bool,
    to_show=6,
    num_col=3,
    predictions=None,
    test=False,
) -> None:
    Axes5x5 = Tuple[
        Tuple[Axes, Axes, Axes, Axes, Axes],
        Tuple[Axes, Axes, Axes, Axes, Axes],
        Tuple[Axes, Axes, Axes, Axes, Axes],
        Tuple[Axes, Axes, Axes, Axes, Axes],
        Tuple[Axes, Axes, Axes, Axes, Axes],
    ]
    num_row = to_show // num_col if to_show // num_col != 0 else 1
    to_show = num_row * num_col
    axs: Axes5x5

    fig, axs = plt.subplots(num_row, num_col, figsize=(5, 5))
    for i in range(to_show):
        if num_row == 1:
            ax: Axes = axs[i % num_col]
        else:
            ax: Axes = axs[i // num_col, i % num_col]

        # mostra iesima coppia, prima e seconda immagine
        ax.imshow(
            tf.concat(
                [dataset_pairs.pairs[i][0], dataset_pairs.pairs[i][1]], axis=1
            ),
            cmap="gray",
        )
        ax.set_axis_off()
        if test:
            ax.set_title(
                "True: {} | Pred: {:.5f}".format(
                    dataset_pairs.labels[i], predictions[i][0]
                )
            )
        else:
            ax.set_title("Label: {}".format(dataset_pairs.labels[i]))
    # if test:
    #    plt.tight_layout(rect=(0, 0, 1.9, 1.9), w_pad=0.0)
    # else:
    #    plt.tight_layout(rect=(0, 0, 1.5, 1.5))
    plt.show(block=not go_on)


def euclidean_distance(vects):
    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=1)
    return tf.math.sqrt(tf.math.maximum(sum_square, keras.backend.epsilon()))


def contrastive_loss(margin=1):
    def _contrastive_loss(
        y_true: npt.NDArray[np.float32], y_pred: npt.NDArray[np.float32]
    ):
        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
        # calcola la contrastive loss e fai la media per ogni coppia di labels passati
        return tf.math.reduce_mean((1 - y_true) * square_pred + (y_true) * margin_square)

    return _contrastive_loss


def contrastive_siamese_model():
    # nota come sta usando tanh. Proprieta':
    # - bounded output = *distanze* relative assumono importanza, ed essendo output tra -1 e 1
    #   puoi usare metriche come cosine similarity
    # - simmetria e smooth gradient
    # specifico la rete che mi fa passare ad un embedding/rappresentazione in feature space
    x_input = keras.layers.Input(shape=(28, 28, 1))
    x = keras.layers.BatchNormalization()(x_input)
    x = keras.layers.Conv2D(filters=4, kernel_size=(5, 5), activation="tanh")(x)
    x = keras.layers.AveragePooling2D(pool_size=(2, 2))(
        x
    )  # stride defaulted a pool_size
    x = keras.layers.Conv2D(filters=16, kernel_size=(5, 5), activation="tanh")(x)
    x = keras.layers.Flatten()(x)

    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(units=10, activation="tanh")(x)

    # crea rete gemella a parametri condivisi tra le due immagini della coppia
    embedding_network = keras.Model(inputs=x_input, outputs=x)

    x_input_1 = keras.layers.Input(shape=(28, 28, 1))
    x_input_2 = keras.layers.Input(shape=(28, 28, 1))

    tower_1 = embedding_network(x_input_1)
    tower_2 = embedding_network(x_input_2)

    # definizione nodo con fuzione custom com keras.layres.Lambda
    merge_layer = keras.layers.Lambda(function=euclidean_distance, output_shape=(1,))(
        [tower_1, tower_2]
    )

    # passo dal vettore di distanza ad una scelta binaria "stessa cifra"(0) o "cifre diverse"(1)
    normal_layer = keras.layers.BatchNormalization()(merge_layer)
    output_layer = keras.layers.Dense(units=1, activation="sigmoid")(normal_layer)
    siamese = keras.Model(inputs=[x_input_1, x_input_2], outputs=output_layer)
    return siamese


# metric: "loss" o "accuracy"
def plt_metric(history, metric, title: str, has_valid=True) -> None:
    plt.plot(history[metric])
    if has_valid:
        plt.plot(history["val_" + metric])
        plt.legend(["train", "validation"], loc="upper left")
    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel("epoch")
    plt.show(block=False)


def add_metric_plot(
    history, metric: str, title: str, ax: Optional[Axes], has_valid=True
) -> None:
    if ax is None:
        raise ValueError("An axis (ax) must be provided to add the plot.")

    ax.plot(history[metric], label="train")
    if has_valid:
        ax.plot(history["val_" + metric], label="validation")
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.set_xlabel("epoch")
    ax.legend(loc="upper left")
