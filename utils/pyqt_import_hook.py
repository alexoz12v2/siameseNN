import sys
import os
import platform
from pathlib import Path
from importlib.abc import MetaPathFinder, Loader
from . import win_dll_import

class PyQtImportHook(MetaPathFinder, Loader):
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.dlls_added = False  # Avoid re-adding DLL paths multiple times

    def find_spec(self, fullname, path, target=None):
        if fullname.startswith("PyQt6") and not self.dlls_added:
            self.add_dll_directories()  # Hook triggers here
            self.dlls_added = True

        return None  # Allow normal import flow

    def add_dll_directories(self):
        print("[PyQtImportHook] Adding DLL directories...")
        win_dll_import.add_dynamic_library_directories(
            self.base_path,
            [".*pyqt6_qt6.*"],
            lambda p: p / "site-packages" / "PyQt6"
        )

class CUDAImportHook(MetaPathFinder, Loader):
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.dlls_added = False  # Avoid re-adding DLL paths multiple times

    def find_spec(self, fullname, path, target=None):
        if (fullname.startswith("nvidia") or fullname.startswith("cu") or fullname.startswith("tensorflow")) and not self.dlls_added:
            self.add_dll_directories()  # Hook triggers here
            self.dlls_added = True

        return None  # Allow normal import flow

    def add_dll_directories(self):
        print(f"[CUDA] Adding DLL directories... from base path {self.base_path}")
        win_dll_import.add_dynamic_library_directories(
            self.base_path,
            [".*nvidia.*"],
            lambda p: p
        )

# Register the import hook
if platform.system() == "Windows" and os.environ.get("BAZEL_FIX_DLL") is not None:
    sys.meta_path.insert(0, PyQtImportHook(Path(sys.argv[0]).parent.parent.parent.resolve()))
    sys.meta_path.insert(0, CUDAImportHook(Path(sys.argv[0]).parent.parent.parent.resolve()))