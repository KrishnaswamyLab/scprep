import preprocessing
import pandas as pd
import numpy as np
from scipy import sparse
import os

if os.getcwd().strip("/").endswith("test"):
    data_dir = os.path.join("..", "..", "data", "test_data")
else:
    data_dir = os.path.join("..", "data", "test_data")


def load_10X(**kwargs):
    return preprocessing.io.load_10X(
        os.path.join(data_dir, "test_10X"), **kwargs)

# TODO: write tests


def test_remove_empty_cells():
    data = load_10X(sparse=False)
    sanitized_data = preprocessing.filter.remove_empty_cells(data)
    assert sanitized_data.shape[1] == data.shape[1]
    assert not np.any(sanitized_data.sum(1) == 0)


def test_remove_empty_cells_sparse():
    data = load_10X(sparse=True)
    sanitized_data = preprocessing.filter.remove_empty_cells(data)
    assert sanitized_data.shape[1] == data.shape[1]
    assert not np.any(sanitized_data.sum(1) == 0)


def test_remove_empty_genes():
    data = load_10X(sparse=False)
    sanitized_data = preprocessing.filter.remove_empty_genes(data)
    assert sanitized_data.shape[0] == data.shape[0]
    assert not np.any(sanitized_data.sum(0) == 0)


def test_remove_empty_genes_sparse():
    data = load_10X(sparse=True)
    sanitized_data = preprocessing.filter.remove_empty_genes(data)
    assert sanitized_data.shape[0] == data.shape[0]
    assert not np.any(sanitized_data.sum(0) == 0)


def test_library_size_filter():
    data = load_10X(sparse=True)
    sanitized_data = preprocessing.filter.filter_library_size(data, 100)
    assert sanitized_data.shape[1] == data.shape[1]
    assert not np.any(sanitized_data.sum(1) < 100)


def test_sparse_dataframe_library_size():
    data = pd.SparseDataFrame(sparse.coo_matrix((10**7, 2 * 10**4)),
                              default_fill_value=0.0)
    cell_sums = preprocessing.filter.library_size(data)
    assert cell_sums.shape[0] == data.shape[0]
