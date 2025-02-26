import utils
import tensorflow as tf
import keras
from pathlib import Path
from typing import NamedTuple
from enum import Enum
from absl import logging, app, flags
import textwrap


class FlagHolderHelpPair(NamedTuple):
    holder: flags.FlagHolder[any]
    help: str


class EArgsName(str, Enum):
    working_directory = "working-directory"
    action = "action"


class EActions(str, Enum):
    help = "help"


_FLAGS = flags.FLAGS
_flag_helps: dict[EArgsName, str] = {
    EArgsName.working_directory: "Directory Used for RW operations",
    EArgsName.action: "Action to be performed by the application"
}
_flag_holders: list[FlagHolderHelpPair] = [
    FlagHolderHelpPair(flags.DEFINE_string(EArgsName.working_directory, str(Path.home() / ".siamese_data"), _flag_helps[EArgsName.working_directory]), _flag_helps[EArgsName.working_directory]),
    FlagHolderHelpPair(flags.DEFINE_enum(EArgsName.action, None, [EActions.help], _flag_helps[EArgsName.action]), _flag_helps[EArgsName.action]),
]


def print_filtered_help():
    print("Usage:")
    max_name_length = max(len(holder.name) for holder, _ in _flag_holders)
    indent = max_name_length + 8
    help_wrap_width = 120
    
    for holder, help_text in _flag_holders:
        first_line = f"    --{holder.name.ljust(max_name_length)}  "
        wrapped_help = textwrap.wrap(help_text, width=help_wrap_width - indent)
        
        print(first_line + wrapped_help[0])
        for line in wrapped_help[1:]:
            print(" " * indent + line)


def main(args: list[str]) -> None:
    logging.info("Hello World! %s", args)
    logging.info("CUDA Capable devices detected by tensorflow: %s", tf.config.list_physical_devices('GPU'))
    if _FLAGS.action is None or _FLAGS.action == EActions.help:
        print_filtered_help()
        return


if __name__ == "__main__":
    app.run(main)