from absl import logging
from absl import app
from pathlib import Path


def compressed_tree(dir_path: Path, prefix: str=''):
    # prefix components:
    space =  '    '
    branch = '│   '
    # pointers:
    tee =    '├── '
    last =   '└── '
    
    contents = list(dir_path.iterdir())
    directories = [path for path in contents if path.is_dir()]
    files = [path for path in contents if path.is_file()]

    # List directories as usual
    for idx, path in enumerate(directories):
        pointer = tee if idx < len(directories) - 1 else last
        yield prefix + pointer + path.name
        yield from compressed_tree(path, prefix=prefix + (branch if pointer == tee else space))

    # Compress files into a file count
    if files:
        # File count entry
        file_count = len(files)
        yield prefix + (tee if directories else '') + f'[{file_count} files]'


def tree(dir_path: Path, prefix: str=''):
    # prefix components:
    space =  '    '
    branch = '│   '
    # pointers:
    tee =    '├── '
    last =   '└── '
    contents = list(dir_path.iterdir())
    # contents each get pointers that are ├── with a final └── :
    pointers = [tee] * (len(contents) - 1) + [last]
    for pointer, path in zip(pointers, contents):
        yield prefix + pointer + path.name
        if path.is_dir(): # extend the prefix and recurse:
            extension = branch if pointer == tee else space 
            # i.e. space because last, └── , above so no more |
            yield from tree(path, prefix=prefix+extension) 


def main(argv: list[str]) -> None:
    del argv
    logging.info('Hello World')
    logging.info(f'cwd: {Path.cwd()}')
    for line in compressed_tree(Path.cwd() / 'classification_from_scratch' / 'extracted_files'):
        logging.info(line)


if __name__ == "__main__":
    app.run(main)
