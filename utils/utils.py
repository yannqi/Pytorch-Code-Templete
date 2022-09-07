
import os.path as osp
import sys


def add_sys_path(path):
    """Add the path to import package.
    Args:
        path : The package path.
    """
    if path not in sys.path:
        sys.path.insert(0, path)



