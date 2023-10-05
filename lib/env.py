import os
from pathlib import Path

# PROJECT_DIR = Path(os.environ['PROJECT_DIR']).absolute().resolve()
# need the full path or set env variable 'PROJECT_DIR' like the above line
procject_root = '/home/vincentqin/数据/ICRL24_CODE' # need the full path

PROJECT_DIR = Path(procject_root).absolute().resolve()
DATA_DIR = PROJECT_DIR / 'data'
OUTPUT_DIR = PROJECT_DIR / 'output'


def get_path(relative_path: str) -> Path:
    return (
        Path(relative_path)
        if relative_path.startswith('/')
        else PROJECT_DIR / relative_path
    )
