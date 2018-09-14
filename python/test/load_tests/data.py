import scprep
import os
import numpy as np

data_dir = os.getcwd().split(os.path.sep)
while data_dir[-1] in ["load_tests", "test", "python"]:
    data_dir = data_dir[:-1]

data_dir = data_dir + ["data", "test_data"]
end = data_dir[1:] if len(data_dir) > 2 else [data_dir[1]]
data_dir = [data_dir[0]] + [os.path.sep] + end
data_dir = os.path.join(*data_dir)


def load_10X(**kwargs):
    return scprep.io.load_10X(
        os.path.join(data_dir, "test_10X"), **kwargs)


def generate_positive_sparse_matrix(shape=[500, 500], seed=42, l=0.1):
    """ Returns an ndarray of shape=shape filled mostly with zeros """
    np.random.seed(seed)
    X = np.random.normal(0, 1, shape) * \
        np.random.poisson(l, shape)
    X = np.abs(X)
    return X
