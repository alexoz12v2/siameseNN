from . import pyqt_import_hook
import platform

if platform.system() == "Windows":
    __all__ = ["pyqt_import_hook"]