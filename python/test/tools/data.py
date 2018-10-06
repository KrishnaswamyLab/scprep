import scprep
import os
import numpy as np

data_dir = os.getcwd().split(os.path.sep)
while data_dir[-1] in ["test_utils", "test", "python"]:
    data_dir = data_dir[:-1]

data_dir = data_dir + ["data", "test_data"]
end = data_dir[1:] if len(data_dir) > 2 else [data_dir[1]]
data_dir = [data_dir[0]] + [os.path.sep] + end
data_dir = os.path.join(*data_dir)


def load_10X(**kwargs):
    return scprep.io.load_10X(
        os.path.join(data_dir, "test_10X"), **kwargs)


def generate_positive_sparse_matrix(shape=[500, 500], seed=42,
                                    poisson_mean=0.1):
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
    X = np.random.normal(0, 1, shape) * \
        np.random.poisson(poisson_mean, shape)
    X = np.abs(X)
    return X
