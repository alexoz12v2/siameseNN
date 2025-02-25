from . import pyqt_import_hook
import platform

if platform.system() == "Windows":
    __all__ = [pyqt_import_hook]

base_file_path = pyqt_import_hook.base_file_path

__all__.append(base_file_path)
