from typing import Callable, Generator, Tuple, BinaryIO, Optional
from absl import logging
import traceback
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.axis import Axis 
from matplotlib.figure import Figure
import os
from importlib import reload
from pathlib import Path
import re
import numpy as np


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
def jfif_filter_2(file_path: Path, file_content: BinaryIO) -> Tuple[bool, Optional[tf.Tensor]]:
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

            img = tf.io.decode_image(img) # Decode the image (tensore dtype uint8)
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


# keras fornisce la `keras.utils.image_dataset_from_directory`, pero non permette di filtrare i
# files in base a un predicato arbitrario
# @tf.function
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
    dataset_length = dataset.reduce(0, lambda x, _: x + 1).numpy()

    # piuttosto che dare un generatore di 1 immagine alla volta, fai lo stack di `batch_size``
    # tensori la botta. Inoltre, fai si che tensorflow prepari piu di un batch in anticipo
    # con una buffersize dinamica
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    # seed fisso per mischiare le immagini con i due labels, in modo uniforme tra esecuzioni diverse
    return dataset.shuffle(
        buffer_size=dataset.cardinality(), seed=42 if use_seed else None
    ), dataset_length


def visualize_first_9_images(dataset: tf.data.Dataset) -> Figure:
    fig = plt.figure(figsize=(10,10))
    for images, labels in dataset.take(1): # il dataset e' batched
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(np.array(tf.transpose(images[i], perm=[2, 1, 0])).astype("uint8"))
            plt.title(int(labels[i]))
            plt.axis("off")

    return fig