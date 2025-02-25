import ctypes
import re
import os
from pathlib import Path
import platform
from typing import Callable


def _win32_short_path_from(path: Path) -> Path:
    long_path = str(path.resolve())
    num_chars = ctypes.windll.kernel32.GetShortPathNameW(long_path, None, 0)
    buffer = ctypes.create_unicode_buffer(num_chars)
    ctypes.windll.kernel32.GetShortPathNameW(long_path, buffer, num_chars)
    return Path(buffer.value)


def _find_first_matching_path(parent_path: Path, name_pattern: str) -> Path:
    regex = re.compile(name_pattern)
    for p in parent_path.iterdir():
        print("- " + p.name)
    matching_paths = [p for p in parent_path.iterdir() if regex.match(p.name)]
    return matching_paths[0]


def _find_all_matching_path(parent_path: Path, name_pattern: str) -> list[Path]:
    regex = re.compile(name_pattern)
    for p in parent_path.iterdir():
        print("- " + p.name)
    matching_paths = [p for p in parent_path.iterdir() if regex.match(p.name)]
    return matching_paths


def _contains_dll_files(path: Path) -> bool:
    return any(file.suffix == ".dll" for file in path.iterdir() if file.is_file())


def _add_dll_path(path: Path) -> None:
    if not path.is_dir():
        raise ValueError(f"{str(path)} doesn't exist")
    os.add_dll_directory(path)
    os.environ["Path"] = str(path) + ";" + os.environ.get("Path")
    print(f"Added {path} to User DLL path")


def _process_directories(
    base_path: Path, predicate: Callable[[Path], bool], func: Callable[[Path], None]
):
    try:
        if base_path.is_dir():
            if predicate(base_path):
                func(base_path)

            for subpath in base_path.iterdir():
                if subpath.is_dir():
                    _process_directories(subpath, predicate, func)
    except PermissionError as e:
        print(f"Permission denied: {base_path}", e)
    except Exception as e:
        print(f"Error: {base_path}", e)


def add_dynamic_library_directories(
    base_path: Path, patterns: list[str], path_processor: Callable[[Path], Path]
) -> None:
    """To be run in bazel run before importing any pypi package which is composed by multiple modules having dynamic libraries"""
    if platform.system() == "Windows":
        base_path = base_path.resolve()
        if not base_path.is_dir():
            raise ValueError(f"Path {str(base_path)} is not a directory")
        for pattern in patterns:
            for p in _find_all_matching_path(base_path, pattern):
                base = path_processor(p)
                _process_directories(
                    base,
                    _contains_dll_files,
                    lambda p: _add_dll_path(_win32_short_path_from(p)),
                )
