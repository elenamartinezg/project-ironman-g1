from pathlib import Path

def create_dir(dir):
    """Create directory if it does not exists

    Args:
        dir: Directory or folder path
    """
    if not dir.exists():
        dir.mkdir(parents=True, exist_ok=True)
