import preprocessing
import pandas as pd
import numpy as np
from sklearn.utils.testing import assert_warns_message
from scipy import sparse
import os
from load_tests.utils import (
    check_all_matrix_types,
    check_transform_equivalent,
)
from functools import partial

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
    check_all_matrix_types(
        data, check_transform_equivalent,
        Y=sanitized_data, transform=preprocessing.filter.remove_empty_cells)


def test_remove_empty_cells_sparse():
    data = load_10X(sparse=True)
    sanitized_data = preprocessing.filter.remove_empty_cells(data)
    assert sanitized_data.shape[1] == data.shape[1]
    assert not np.any(sanitized_data.sum(1) == 0)
    check_all_matrix_types(
        data, check_transform_equivalent,
        Y=sanitized_data, transform=preprocessing.filter.remove_empty_cells)


def test_remove_empty_genes():
    data = load_10X(sparse=False)
    sanitized_data = preprocessing.filter.remove_empty_genes(data)
    assert sanitized_data.shape[0] == data.shape[0]
    assert not np.any(sanitized_data.sum(0) == 0)
    check_all_matrix_types(
        data, check_transform_equivalent,
        Y=sanitized_data, transform=preprocessing.filter.remove_empty_genes)


def test_remove_empty_genes_sparse():
    data = load_10X(sparse=True)
    sanitized_data = preprocessing.filter.remove_empty_genes(data)
    assert sanitized_data.shape[0] == data.shape[0]
    assert not np.any(sanitized_data.sum(0) == 0)
    check_all_matrix_types(
        data, check_transform_equivalent,
        Y=sanitized_data, transform=preprocessing.filter.remove_empty_genes)


def test_library_size_filter():
    data = load_10X(sparse=True)
    sanitized_data = preprocessing.filter.filter_library_size(data, 100)
    assert sanitized_data.shape[1] == data.shape[1]
    assert not np.any(sanitized_data.sum(1) < 100)
    check_all_matrix_types(
        data, check_transform_equivalent,
        Y=sanitized_data, transform=partial(
            preprocessing.filter.filter_library_size, 100))


def test_gene_expression_filter():
    data = load_10X(sparse=True)
    genes = np.arange(10)
    sanitized_data = preprocessing.filter.filter_gene_expression(
        data, genes, percentile=90, keep_cells='below')
    assert sanitized_data.shape[1] == data.shape[1]
    assert np.max(np.sum(data[genes], axis=1)) > np.max(
        np.sum(sanitized_data[genes], axis=1))
    check_all_matrix_types(
        data, check_transform_equivalent,
        Y=sanitized_data, transform=partial(
            preprocessing.filter.filter_gene_expression, genes,
            percentile=90, keep_cells='below'))
    sanitized_data = preprocessing.filter.filter_gene_expression(
        data, genes, percentile=10, keep_cells='above')
    assert sanitized_data.shape[1] == data.shape[1]
    assert np.min(np.sum(data[genes], axis=1)) < np.min(
        np.sum(sanitized_data[genes], axis=1))
    check_all_matrix_types(
        data, check_transform_equivalent,
        Y=sanitized_data, transform=partial(
            preprocessing.filter.filter_gene_expression, genes,
            percentile=10, keep_cells='above'))


def test_gene_expression_filter_warning():
    data = load_10X(sparse=True)
    genes = np.arange(10)
    assert_warns_message(
        UserWarning,
        "`percentile` expects values between 0 and 100. "
        "Got 0.90. Did you mean 90?",
        preprocessing.filter.filter_gene_expression,
        data, genes, percentile=0.90, keep_cells='below')
    assert_warns_message(
        UserWarning,
        "Only one of `cutoff` and `percentile` should be given.",
        preprocessing.filter.filter_gene_expression,
        data, genes, percentile=0.90, cutoff=50)


def test_large_sparse_dataframe_library_size():
    data = pd.SparseDataFrame(sparse.coo_matrix((10**7, 2 * 10**4)),
                              default_fill_value=0.0)
    cell_sums = preprocessing.filter.library_size(data)
    assert cell_sums.shape[0] == data.shape[0]
