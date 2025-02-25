from absl import logging
from absl import app
from pathlib import Path
import numpy as np
import utils
from PyQt6.QtWidgets import QApplication, QWidget
import tensorflow as tf
import keras
import classification_from_scratch.class_utils as cutils
import matplotlib.pyplot as plt
import matplotlib
import pydot
from matplotlib.axes import Axes


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


def tree(dir_path: Path, prefix: str = ""):
    # prefix components:
    space = "    "
    branch = "│   "
    # pointers:
    tee = "├── "
    last = "└── "
    contents = list(dir_path.iterdir())
    # contents each get pointers that are ├── with a final └── :
    pointers = [tee] * (len(contents) - 1) + [last]
    for pointer, path in zip(pointers, contents):
        yield prefix + pointer + path.name
        if path.is_dir():  # extend the prefix and recurse:
            extension = branch if pointer == tee else space
            # i.e. space because last, └── , above so no more |
            yield from tree(path, prefix=prefix + extension)


@tf.function
def calculate_validation_size(dataset_numbatches: tf.Tensor) -> tf.Tensor:
    validation_size = tf.cast(0.2 * tf.cast(dataset_numbatches, tf.float32), tf.int64)
    return validation_size


def main(argv: list[str]) -> None:
    del argv
    logging.info("Hello World")
    logging.info("cwd: %s", str(Path.cwd()))
    data_path = utils.base_file_path() / "classification_from_scratch" / "extracted_files"
    for line in compressed_tree(data_path):
        logging.info(line)

    # hello
    matplotlib.use("QtAgg")
    cutils.set_keras_backend("tensorflow")
    keras.utils.set_random_seed(812)

    # su Tensorflow 2.10 GPU crasha
    # tf.config.experimental.enable_op_determinism()

    logging.info("Active Keras Backend: %s", keras.backend.backend())
    logging.info("CUDA Devices: %s", tf.config.list_physical_devices('GPU'))

    # prova a caricare il dataset
    batch_size = 8
    # validation_size = calculate_validation_size(dataset_numbatches)

    ## splitta e fai shuffling del solo training set
    input_path = data_path / "PetImages"
    output_path = data_path.parent / "cats_and_dogs"
    image_size = (256, 256)
    # c'e una immagine corrotta nel dataset che non riesco a filtrare, allora ne prendo 50
    # e prego che sia corretta
    cutils.select_valid_images(
        input_path, output_path, return_if_exists=True, max_images=50
    )
    dataset_train, dataset_val = keras.utils.image_dataset_from_directory(
        str(output_path),
        validation_split=0.2,
        subset="both",
        image_size=image_size,
        seed=32,
        batch_size=batch_size,
    )

    logging.info("Dataset for training and validation created.")
    logging.info("\tTrain:      %d batches of %d", dataset_train.cardinality(), batch_size)
    logging.info("\tValidation: %d batches of %d", dataset_val.cardinality(), batch_size)

    # visualizza 9 immagini
    logging.info(
        "Visualising the first 9 images of the dataset, close the window to proceed...\n"
    )
    cutils.visualize_first_9_images(
        dataset_train, transpose=False, batch_size=batch_size
    )
    plt.show(block=True)

    for images, labels in dataset_train.take(1):
        logging.info("Augmenting first image 9 times, and showing the results")
        logging.info("Close the Window to continue...\n")
        for i in range(9):
            ax: Axes = plt.subplot(3, 3, i + 1)
            augmented_images = cutils.augment_images_from_batch(images)
            plt.imshow(np.array(augmented_images[0]).astype("uint8"))
            plt.axis("off")
        plt.show(block=True)

    logging.info(
        "Creating a model for cats_and_dogs dataset, with %d input size and 2 classes", image_size
    )
    model = cutils.make_model(input_shape=image_size + (3,), num_classes=2)

    # questa funzione salva il modello in {cwd}/model.png, che quindi puoi o aprire da qua o
    # aprire manualmente, vedi nel bazel sandbox attraverso il convenience symlink `bazel-bin`
    keras.utils.plot_model(model, show_shapes=True)
    logging.info("you can open the `model.png` on directory %s", str(Path.cwd()))

    # training del modello
    epochs = 1  # e' lento
    # funzioni che keras (tensorflow) richiama a fine epoca
    callbacks = [keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras")]
    # prepara il modello per essere allenato, devinendogli ottimizzatore, loss, metric
    # cose che con la api con sottoclassi di keras.layers.Layers devi fare manuali
    model.compile(
        optimizer=keras.optimizers.Adam(3e-4),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=[keras.metrics.BinaryAccuracy(name="acc")],
    )
    logging.info(
        "Started traning with Adam Optimizer, CrossEntropy loss, Binary Accuracy for %d epochs", epochs
    )
    # allena!
    model.fit(
        dataset_train,
        epochs=epochs,
        callbacks=callbacks,
        validation_data=dataset_val,
    )

    # scegli una immagine a caso, mostrala e fai inferenza
    # keras.utils.load_img = tf.io.read_file + tf.io.decode_image
    img_path = next((output_path / "Cat").iterdir())
    img = keras.utils.load_img(str(img_path), target_size=image_size)
    logging.info(
        "Trying inference with image Cat/%s, showing it. Close window to continue...", img_path.name
    )
    plt.imshow(img)
    plt.show(block=True)

    # convertiamo in tensore float valori normalizzati la immagine da PIL, crea asse del batch (a 1 perche hai 1 sola img)
    img_array = keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = float(
        tf.math.sigmoid(predictions[0][0])
    )  # tira fuori il numero da asse dei batch(None) e num_classes(1)
    logging.info(
        "This image is %.2f%% cat and %.2f%% dog.\n",
        100 * (1 - score),
        100 * score,
    )


if __name__ == "__main__":
    app.run(main)
