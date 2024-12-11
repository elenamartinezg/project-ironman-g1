from pathlib import Path
import pickle

def create_dir(dir):
    """Create directory if it does not exists

    Args:
        dir: Directory or folder path
    """
    if not dir.exists():
        dir.mkdir(parents=True, exist_ok=True)

def load_pickle(pickle_fpath):
    """Load Pickle File

    Args:
        pickle_fpath (Path): Input Pickle Filepath

    Returns:
        data: Loaded file
    """
    with open(pickle_fpath, "rb") as file:
        data = pickle.load(file)
    return data


def dump_pickle(data, pickle_fpath):
    """Dump Pickle File

    Args:
        data: Loaded file
        pickle_fpath (Path): Output Pickle Filepath

    """
    create_dir(pickle_fpath.parent)
    with open(pickle_fpath, "wb") as f:
        pickle.dump(data, f)
        