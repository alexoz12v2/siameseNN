# supposed to be run by venv

import os
from pathlib import Path

base_path = Path.cwd() / ".venv" / "Lib" / "site-packages" / "nvidia"
if not base_path.is_dir():
    raise ValueError("Check venv installed with dependencies and that you are on the project root")

# Iterate through all directories and subdirectories
for p in base_path.rglob("*"):  # Use rglob for recursive globbing
    if p.is_dir():
        # Add to DLL search path
        os.add_dll_directory(p)
        print(f"Added directory {p} to DLL search path")
        
        # Add to PATH environment variable
        os.environ["PATH"] += os.pathsep + str(p)
        print(f"Added directory {p} to PATH")

import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))