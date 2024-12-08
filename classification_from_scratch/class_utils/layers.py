from typing import Callable, Generator, Tuple
from absl import logging
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from importlib import reload
from pathlib import Path
import re


def set_keras_backend(backend):
    if keras.backend.backend() != backend:
        os.environ["KERAS_BACKEND"] = backend
        reload(keras.backend)
        assert keras.backend.backend() == backend


# seguo https://keras.io/examples/vision/image_classification_from_scratch/
# https://stackoverflow.com/questions/24501462/what-type-are-file-objects-in-python
# https://www.tensorflow.org/api_docs/python/tf/data/Dataset#from_generator
def load_image(
    img: tf.Tensor, label: int, img_size: Tuple[int, int]
) -> Tuple[tf.Tensor, int]:
    """Load and preprocess the image."""
    img = tf.image.decode_jpeg(img, channels=3)  # Decode the image
    img = tf.image.resize(img, img_size)  # Resize the image
    img = img / 255.0  # Normalize the image
    return img, label


def filter_images(
    directory_path: Path,
    img_size: Tuple[int, int],
    filename_predicate: Callable[[str], bool] = lambda x: True,
    predicate: Callable[[tf.Tensor], bool] = lambda x: True,
) -> Callable[[], Generator[Tuple[tf.Tensor, int], tf.Tensor, None]]:
    def gen():
        current_label = 0
        for subdir in directory_path.iterdir():
            if subdir.is_dir():
                for file_path in subdir.iterdir():
                    if file_path.is_file() and filename_predicate(file_path.name):
                        try:  # Apply the predicate to decide if the file should be included
                            with tf.io.read_file(str(file_path)) as fobj:
                                if predicate(fobj):
                                    yield load_image(fobj, current_label, img_size)
                        except Exception as e:
                            # In case there are any issues with reading the file, skip it
                            print(f"Error reading {file_path}: {e}")
                current_label += 1

    return gen


def jfif_filter(file_content: tf.Tensor) -> bool:
    """Predicate function that checks if the file is a JFIF image."""
    # Check if the first 4 bytes contain the 'JFIF' header
    # Convert the first 4 bytes of the file content to a string to compare with 'JFIF'
    return tf.strings.regex_full_match(file_content[:4], b"JFIF")


# keras fornisce la `keras.utils.image_dataset_from_directory`, pero non permette di filtrare i
# files in base a un predicato arbitrario
# @tf.function
def custom_image_dataset_from_directory(
    directory_path: Path, batch_size: int = 32, img_size=(256, 256)
) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_generator(
        filter_images(
            directory_path,
            img_size,
            filename_predicate=lambda x: bool(re.search("\.jpe?g$", x)),
            predicate=jfif_filter,
        ),
        output_signature=(
            tf.TensorSpec(shape=(3, *img_size), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )
    logging.info(f"dataset before batch+prefetch = {dataset}")

    # piuttosto che dare un generatore di 1 immagine alla volta, fai lo stack di `batch_size``
    # tensori la botta. Inoltre, fai si che tensorflow prepari piu di un batch in anticipo
    # con una buffersize dinamica
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
