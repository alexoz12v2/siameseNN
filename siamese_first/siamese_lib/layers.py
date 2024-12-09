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
