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
            self.base_path, [".*pyqt6_qt6.*"], lambda p: p / "site-packages" / "PyQt6"
        )


class CUDAImportHook(MetaPathFinder, Loader):
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.dlls_added = False  # Avoid re-adding DLL paths multiple times

    def find_spec(self, fullname, path, target=None):
        if (
            fullname.startswith("nvidia")
            or fullname.startswith("cu")
            or fullname.startswith("tensorflow")
        ) and not self.dlls_added:
            self.add_dll_directories()  # Hook triggers here
            self.dlls_added = True

        return None  # Allow normal import flow

    def add_dll_directories(self):
        print(f"[CUDA] Adding DLL directories... from base path {self.base_path}")
        if os.environ.get("DEVENV_NVIDIA_PATH") is not None:
            win_dll_import.add_dynamic_library_directories(
                self.base_path, [".*"], lambda p: p
            )
        else:
            win_dll_import.add_dynamic_library_directories(
                self.base_path, [".*nvidia.*"], lambda p: p
            )


def is_running_from_bazel() -> bool:
    return "runfiles" in sys.executable or "bazel-out" in sys.executable


def base_file_path() -> Path:
    if is_running_from_bazel():
        return Path(sys.argv[0]).parent.parent.parent.resolve() / "_main"
    elif "runfiles" in str(Path.cwd()):
        return Path.cwd()
    else:  # assuming we are on the workspace with convenience symlinks, and that you already did bazel build
        path = str(Path(sys.argv[0]).resolve())
        print("asdfjskaldfjdkfjla s ", path)
        base = "siameseNN"

        # Get the substring after "siameseNN"
        _, after_base = path.split(base + os.path.sep, 1)

        # Remove "__main__.py" and trailing separators
        result = after_base.rsplit(os.path.sep + "__main__.py", 1)[0]
        bpath = sys.argv[0].split(os.path.sep)[-2]
        if platform.system() == "Windows":
            runfiles = bpath + ".exe.runfiles"
        else:
            runfiles = bpath + ".runfiles"
        return (Path.cwd() / "bazel-bin" / result / runfiles / "_main").resolve()


# Register the import hook
if platform.system() == "Windows" and is_running_from_bazel():
    sys.meta_path.insert(
        0, PyQtImportHook(Path(sys.argv[0]).parent.parent.parent.resolve())
    )
    sys.meta_path.insert(
        0, CUDAImportHook(Path(sys.argv[0]).parent.parent.parent.resolve())
    )
elif platform.system() == "Windows" and os.environ.get("DEVENV_NVIDIA_PATH") is not None:
    path = Path(os.environ.get("DEVENV_NVIDIA_PATH"))
    if path.exists() and path.is_dir():
        path = path.resolve()
        sys.meta_path.insert(0, CUDAImportHook(path))
# else:
#    path = Path(sys.argv[0]).parent.parent.parent.resolve() # /home/alessio/.cache/bazel/_bazel_alessio/ec037b2a92d4035b87428840523bc6cc/execroot/
#    sys.meta_path.insert(
#        0, PyQtImportHook(path)
#    )
