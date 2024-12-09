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


# seguo https://keras.io/examples/vision/image_classification_from_scratch/
# https://stackoverflow.com/questions/24501462/what-type-are-file-objects-in-python
# https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_generator
def preprocess_image(
    img: tf.Tensor, label: int, img_size: Tuple[int, int]
) -> Tuple[tf.Tensor, int]:
    """Load and preprocess the image."""
    img = tf.image.resize(img, img_size)  # Resize the image
    img = img / 255.0  # Normalize the image
    img = tf.transpose(
        img, perm=[2, 1, 0]
    )  # da (width, height, num_channels) a (num_channels, height, width) (abitudine pytorch)
    return img, label


def filter_images(
    directory_path: Path,
    img_size: Tuple[int, int],
    filename_predicate: Callable[[str], bool] = lambda x: True,
    predicate: Callable[[Path, tf.Tensor], bool] = lambda x: True,
) -> Callable[[], Generator[Tuple[tf.Tensor, int], tf.Tensor, None]]:
    def gen():
        current_label = 0
        for subdir in directory_path.iterdir():
            if subdir.is_dir():
                for file_path in subdir.iterdir():
                    if file_path.is_file() and filename_predicate(file_path.name):
                        try:  # Apply the predicate to decide if the file should be included
                            with open(file_path, "rb") as fobj:
                                cond, img = predicate(file_path, fobj)
                                if cond:
                                    yield preprocess_image(
                                        img,
                                        current_label,
                                        img_size,
                                    )
                        except Exception:
                            logging.error(
                                f"Error reading {file_path}: {traceback.print_exc()}"
                            )
                current_label += 1

    return gen


# allora provo a fare una decodifica due volte, al fine di poter filtrare le immagini meglio
def jfif_filter_2(
    file_path: Path, file_content: BinaryIO
) -> Tuple[bool, Optional[tf.Tensor]]:
    """Predicate function that checks if the file is a valid JFIF image."""
    try:
        logging.debug(f"-- Checking Image {file_path.name} ...")

        # Check if the header contains 'JFIF'
        contains_jfif = tf.compat.as_bytes("JFIF") in file_content.peek(10)
        if contains_jfif:  # check if the image is RGB and jpeg
            logging.debug(f"\t image {file_path.name} contains the JFIF marker")
            img = tf.io.read_file(str(file_path))
            is_jpg = tf.io.is_jpeg(img)
            if not is_jpg:
                return False, None

            image_validity_filter(file_path)

            img = tf.io.decode_image(img)  # Decode the image (tensore dtype uint8)
            logging.debug(f"\t image has is jpg? {is_jpg}")
            logging.debug(f"\t image has {img.ndim} dimensions and {img.shape} shape")
            logging.debug(f"-------------------------------------------")
            result = img.ndim == 3 and img.shape[2] == 3
            return result, img if result else None
        else:
            logging.debug(f"\t image {file_path.name} does NOT the JFIF marker")
            logging.debug(f"-------------------------------------------")
            return False, None
    except tf.errors.InvalidArgumentError:
        # Log the corruption and filter out the file
        logging.debug(f"Invalid JPEG data:\n\t{traceback.print_exc()}")
        return False, None


def image_validity_filter(file_path: Path) -> None:
    with Image.open(file_path) as img:
        img.verify()


# keras fornisce la `keras.utils.image_dataset_from_directory`, pero non permette di filtrare i
# files in base a un predicato arbitrario
@tf.function
def custom_image_dataset_from_directory(
    directory_path: Path,
    *,
    batch_size: int = 32,
    img_size=(256, 128),
    use_seed: bool = False,
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_generator(
        filter_images(
            directory_path,
            img_size,
            filename_predicate=lambda x: re.search("\.jpe?g$", x),
            predicate=jfif_filter_2,
        ),
        output_signature=(
            tf.TensorSpec(shape=(3, *reversed(img_size)), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )
    logging.info(f"dataset before batch+prefetch = {dataset}")
    # se non va, usa dataset.reduce(0, lambda x,_: x+1).numpy()
    # dataset_length = tf.data.experimental.cardinality(dataset).numpy()
    dataset_length = dataset.reduce(0, lambda x, _: x + 1)

    # piuttosto che dare un generatore di 1 immagine alla volta, fai lo stack di `batch_size``
    # tensori la botta. Inoltre, fai si che tensorflow prepari piu di un batch in anticipo
    # con una buffersize dinamica
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    # seed fisso per mischiare le immagini con i due labels, in modo uniforme tra esecuzioni diverse
    return dataset.shuffle(
        buffer_size=batch_size, seed=42 if use_seed else None
    ), dataset_length


def select_valid_images(
    input_path: Path, output_path: Path, *, return_if_exists: bool, max_images: int
) -> None:
    if not output_path.exists():
        output_path.mkdir()
    elif return_if_exists:
        return

    # i primi 3 bytes di un file jpeg
    jpeg_magic_bytes = b"\xff\xd8\xff" 
    for subdir in input_path.iterdir():
        counter = 0
        if subdir.is_dir():
            path = output_path / subdir.name
            path.mkdir()
            for file_path in subdir.iterdir():
                if file_path.is_file():
                    try:
                        fobj = open(file_path, "rb")
                        is_jfif = b"JFIF" in fobj.peek(10)
                        is_jpg = fobj.peek(10).startswith(jpeg_magic_bytes)
                    finally:
                        fobj.close()

                    if is_jfif and is_jpg and counter < max_images:
                        shutil.copy(file_path, path / file_path.name)
                        counter += 1


def visualize_first_9_images(dataset: tf.data.Dataset, *, transpose, batch_size: int) -> Figure:
    fig = plt.figure(figsize=(10, 10))
    for images, labels in dataset.take(1):  # il dataset e' batched
        for i in range(min(9, batch_size)):
            ax = plt.subplot(3, 3, i + 1)
            if transpose:
                plt.imshow(
                    np.array(tf.transpose(images[i], perm=[2, 1, 0])).astype("uint8")
                )
            else:
                plt.imshow(np.array(images[i]).astype("uint8"))

            plt.title(int(labels[i]))
            plt.axis("off")

    return fig


def visualize_images(images: tf.Tensor, labels: tf.Tensor, *, transpose: bool, batch_size: int) -> None:
    for i in range(min(9, batch_size)):
        ax = plt.subplot(3, 3, i + 1)
        if transpose:
            plt.imshow(
                np.array(tf.transpose(images[i], perm=[2, 1, 0])).astype("uint8")
            )
        else:
            plt.imshow(np.array(images[i]).astype("uint8"))

        plt.title(int(labels[i]))
        plt.axis("off")


# la data augmentation e' la modifica con un fattore di randomicita del dataset di training
# essa puo essere introdotta nel dataset, oppure integrata all'interno del modello
# la seguente funzione prende un batch di immagini del tipo (num_batches, width, height, num_channels)
# e applica alle sue prime num_images la data augmentation e ritorna le immagini trasformate
# (l'input viene da un take da un batched dataset)
# tutti i layers "Random" sono inattivi a test time, quindi quando chiami evaluate()
def augment_images_from_batch(images: tf.Tensor, num_images: int = 9) -> tf.Tensor:
    data_augmentation_layers = [
        keras.layers.RandomFlip(mode="horizontal"),
        keras.layers.RandomRotation(factor=0.1),  # radianti, antiorario
    ]
    for layer in data_augmentation_layers:
        images = layer(images)

    return images


# keras ha diverse api per costruire un modello. il target //app:keras_test usa l'api
# con le sottoclassi. Qui invece creiamo un modello con una funzione che accorpa
# piu layers
# @tf.function -> serve a poco perche la costruzione del modello va 1 sola volta
def make_model(
    input_shape: Union[Tuple[int, int, int, int], list[int]], num_classes: int
) -> keras.Model:
    # tutti i calcoli fatti qui usano computazione simbolica, senza actual numbers
    # quindi usando la tensorflow "graph mode", al fine di costruire il grafo di
    # computazione
    the_inputs = keras.Input(shape=input_shape)  # un batch

    # entry block (normalizzazione dei dati)
    # dopo ogni convoluzione e' norma mettere un blocco di batch normalization e activation
    # padding same aggiunge la quantita di padding necessaria a preservare la size
    # assumendo stride a 1. Dato che gli ho passato stride a 2, sto dimezzando
    x = keras.layers.Rescaling(1.0 / 255)(the_inputs)
    x = keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    # salva il residuo
    previous_block_activation = x

    for size in [256, 512, 728]:
        # depthwise separable convolutions o grouped convolutions affinche ciascun kernel
        # del layer convolutivo veda soltanto una parte delle features, al fine di diminuire
        # il numero di parametri, che esplode al crescere della depth (numero features)
        # https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.SeparableConv2D(filters=size, kernel_size=3, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.Activation("relu")(x)
        x = keras.layers.SeparableConv2D(filters=size, kernel_size=3, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)

        # downsampling, o strided convolutions o max pooling
        #* finestra a 3, stride 2, padding same -> Dimezza
        x = keras.layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(x)

        # convoluzione 1x1 per cambiare il numero di canali del blocco residuale
        residual = keras.layers.Conv2D(
            filters=size, kernel_size=1, strides=2, padding="same"
        )(previous_block_activation)

        # aggiungi blocco residuale
        # - keras.ops.add aggiunge due tensori numerici
        # - keras.layers.add crea un nodo di somma nel grafo di computazione
        x = keras.layers.add([x, residual])

        # aggiorna blocco residuale corrente
        previous_block_activation = x

    x = keras.layers.SeparableConv2D(filters=1024, kernel_size=3, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)

    # mi serve un class label, quindi collasso tutte le posizioni spaziali per ottenere
    # un vettore di dimansione 1024
    x = keras.layers.GlobalAveragePooling2D()(x)

    # dai 1024 numeri dobbiamo passare a `num_classes` scores. (caso particolare, se le
    # classi sono 2, passa a un solo score, perche se (tradotto in probabilita) e'
    # minore di 0.5, scegli la prima classe, la seconda altrimenti)
    output_units = 1 if num_classes == 2 else num_classes

    # layer attivo solo in `fit()` mode: Dropout, che mette a caso a 0 degli scores
    # al fine di evitare overfitting
    x = keras.layers.Dropout(0.25)(x)

    # ultimo layer fully connected per ridurre da 1024 a `output_units`, senza attivazione
    # per ritornare "logits" o "scores", piuttosto che probabilita, ottenibili con la softmax
    outputs = keras.layers.Dense(units=output_units, activation=None)(x)
    return keras.Model(inputs=the_inputs, outputs=outputs)


def dataset_with_pillow(data_dir: Path, image_size: Tuple[int, int], batch_size: int, validation_split: float, seed: int) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    # Create DirectoryIterators
    train_iterator = train_datagen.flow_from_directory(
        data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        seed=seed,
    )

    val_iterator = val_datagen.flow_from_directory(
        data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        seed=seed,
    )

    # Obtain the class names from the train_iterator (they are sorted lexicographically)
    class_names = sorted(train_iterator.class_indices.keys())  # Sorted class names

    # Create a mapping of class names to integer labels (index)
    class_to_index = {class_name: idx for idx, class_name in enumerate(class_names)}

    # Convert one-hot labels to class indices
    def convert_to_index(image, label):
        # Convert one-hot encoded label to index
        label_index = tf.argmax(label, axis=-1, output_type=tf.int32)  # Get index of the 1 in the one-hot vector
        return image, label_index

    # Convert DirectoryIterator to tf.data.Dataset
    def iterator_to_dataset(iterator):
        # Convert each batch of data
        dataset = tf.data.Dataset.from_generator(
            lambda: (next(iterator) for _ in iter(int, 1)),  # Infinite generator from iterator
            output_signature=(
                tf.TensorSpec(shape=(None, *image_size, 3), dtype=tf.float32),  # Batch of images
                tf.TensorSpec(shape=(None,), dtype=tf.int32),  # Batch of class indices
            )
        )

        # Apply conversion function to each batch to convert one-hot labels to indices
        return dataset.map(convert_to_index)

    # Convert train and validation iterators to tf.data.Dataset with class indices
    dataset_train = iterator_to_dataset(train_iterator)
    dataset_val = iterator_to_dataset(val_iterator)
    return dataset_train, dataset_val