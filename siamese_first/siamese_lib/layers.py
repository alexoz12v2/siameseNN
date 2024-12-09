from typing import Callable, Generator, Tuple, BinaryIO, Optional, Union
from absl import logging
import traceback
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.axis import Axis
from matplotlib.figure import Figure
import pydot
import os
from importlib import reload
from pathlib import Path
import re
import numpy as np
from PIL import Image
import shutil


def set_keras_backend(backend):
    if keras.backend.backend() != backend:
        os.environ["KERAS_BACKEND"] = backend
        reload(keras.backend)
        assert keras.backend.backend() == backend


# ------------------------------ TRIPLET LOSS ----------------------------------
# coppia di funzioni responsabili per preprocessare una tripletta di immagini
# https://keras.io/examples/vision/siamese_network/
def preprocess_image(filename: Union[str, tf.Tensor], target_shape: Union[list[int], Tuple[int, int]]) -> tf.Tensor:
    image_bytes = tf.io.read_file(filename)
    # a quanto pare esplode in graph mode
    # image = tf.io.decode_image(image_bytes, channels=3) The tf.io.decode_image function may produce a tensor without a fully defined shape if the input is ambiguous
    image = tf.io.decode_jpeg(image_bytes, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, target_shape)
    return image

    
def preprocess_triplet(target_shape: Union[list[int], Tuple[int, int]]) -> Callable[[str, str, str], Tuple[tf.Tensor, tf.Tensor, tf.Tensor]]:
    def f(anchor: str, positive: str, negative: str) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        return (
            preprocess_image(anchor, target_shape),
            preprocess_image(positive, target_shape),
            preprocess_image(negative, target_shape),
        )
    return f


def triplet_datasets(anchor_images_path: Path, positive_images_path: Path, *, batch_size: int, target_shape: Tuple[int, int], seed: Optional[int]) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    # essendo numerate per corrispondenza numerica, le voglio ordinate
    # https://www.tensorflow.org/guide/data#consuming_sets_of_files
    # anchor_dataset = tf.data.Dataset.list_files(str(anchor_images_path / '*.jpg'))
    # positive_dataset = tf.data.Dataset.list_files(str(positive_images_path / '*.jpg'))

    # mescolo le immagini left e right, cosi non corrispondono piu, e le tratto come negative
    anchor_images = sorted([str(f) for f in anchor_images_path.iterdir() if f.is_file()])
    positive_images = sorted([str(f) for f in positive_images_path.iterdir() if f.is_file()])
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
        train_dataset.batch(batch_size=batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE),
        val_dataset.batch(batch_size=batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE),
    )


# list dims: batch width height channels
def visualize(anchor: list[list[list[list[float]]]], positive: list[list[list[list[float]]]], negative: list[list[list[list[float]]]], *, go_on=False) -> None:
    Axes3x3 = Tuple[Tuple[Axes, Axes, Axes], Tuple[Axes, Axes, Axes], Tuple[Axes, Axes, Axes]]
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
    envvar = os.getenv('KERAS_HOME')
    logstrs = {
        'where': 'KERAS_HOME' if envvar is not None else '${Home}/.keras',
        'value': f" = {envvar}" if envvar is not None else '',
    }
    logging.info(f"about to download ResNet50 imagenet's weights to {logstrs['where'] + logstrs['value']}")
    base_cnn = keras.applications.resnet.ResNet50(weights="imagenet", input_shape=target_shape + (3,), include_top=False)

    flatten = keras.layers.Flatten()(base_cnn.output)
    dense1 = keras.layers.Dense(units=512, activation="relu")(flatten)
    dense1 = keras.layers.BatchNormalization()(dense1)
    dense2 = keras.layers.Dense(units=256, activation="relu")(dense1)
    dense2 = keras.layers.BatchNormalization()(dense2)
    output = keras.layers.Dense(units=256)(dense2) # logits

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

    def call(self, anchor: tf.Tensor, positive: tf.Tensor, negative: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        ap_distance = keras.ops.sum(tf.square(anchor - positive), axis=None)
        an_distance = keras.ops.sum(tf.square(anchor - negative), axis=None)
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
            loss = self._compute_triplet_loss(data) # esiste ma faccio override

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
        return { "loss": self.loss_tracker.result() }

    
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