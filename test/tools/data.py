import scprep
import os
import numpy as np


def _os_agnostic_fullpath_join(path):
    """Join a list of directory names that start from the root

    Handles both Windows and Unix
    >>> os.path.join(os.sep, 'C:' + os.sep, ...)
    'C:\\...'
    >>> os.path.join(os.sep, 'usr' + os.sep, ...)
    '/usr/...'

    Parameters
    ----------
    path : list of directory names, starting from the root

    Returns
    -------
    path : absolute path string
    """
    end = path[1:]
    if not isinstance(end, list):
        end = list(end)
    path = os.path.join(os.sep, path[0] + os.sep, *end)
    return path


def _get_root_dir():
    """Get path to scprep root
    """
    cwd = os.getcwd().split(os.path.sep)
    while cwd[-1] in ["test_utils", "test"]:
        cwd = cwd[:-1]
    return _os_agnostic_fullpath_join(cwd)


def _get_data_dir():
    """Get path to scprep data directory
    """
    return os.path.join(_get_root_dir(), "data", "test_data")


data_dir = _get_data_dir()


def load_10X(**kwargs):
    """Load sample 10X data

    Parameters
    ----------
    **kwargs : keyword arguments for scprep.io.load_10X

    Returns
    -------
    data : array-like 10X data
    """
    return scprep.io.load_10X(os.path.join(data_dir, "test_10X"), **kwargs)


def generate_positive_sparse_matrix(shape=[200, 500], seed=42, poisson_mean=0.1):
    """Returns an ndarray of shape=shape filled mostly with zeros

    Creates a matrix with np.random.normal and multiplies the result
    with np.random.poisson

    Parameters
    ----------
    shape: list or tuple, matrix shape
    seed: random seed
    poisson_mean: controls sparsity. Values close to zero give a sparse matrix.

    Returns
    -------
    np.ndarray
    """
    np.random.seed(seed)
    X = np.random.normal(0, 1, shape) * np.random.poisson(poisson_mean, shape)
    X = np.abs(X)
    return X
