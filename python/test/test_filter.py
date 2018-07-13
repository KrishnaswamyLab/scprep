import scprep
import pandas as pd
import numpy as np
from sklearn.utils.testing import assert_warns_message, assert_raise_message
from scipy import sparse
from load_tests import utils, matrix, data
from functools import partial


def test_remove_empty_cells():
    X = data.load_10X(sparse=False)
    X_filtered = scprep.filter.remove_empty_cells(X)
    assert X_filtered.shape[1] == X.shape[1]
    assert not np.any(X_filtered.sum(1) == 0)
    matrix.check_all_matrix_types(
        X, utils.check_output_equivalent,
        Y=X_filtered, transform=scprep.filter.remove_empty_cells)


def test_remove_empty_cells_sparse():
    X = data.load_10X(sparse=True)
    X_filtered = scprep.filter.remove_empty_cells(X)
    assert X_filtered.shape[1] == X.shape[1]
    assert not np.any(X_filtered.sum(1) == 0)
    matrix.check_all_matrix_types(
        X, utils.check_output_equivalent,
        Y=X_filtered, transform=scprep.filter.remove_empty_cells)


def test_remove_empty_genes():
    X = data.load_10X(sparse=False)
    X_filtered = scprep.filter.remove_empty_genes(X)
    assert X_filtered.shape[0] == X.shape[0]
    assert not np.any(X_filtered.sum(0) == 0)
    matrix.check_all_matrix_types(
        X, utils.check_output_equivalent,
        Y=X_filtered, transform=scprep.filter.remove_empty_genes)


def test_remove_empty_genes_sparse():
    X = data.load_10X(sparse=True)
    X_filtered = scprep.filter.remove_empty_genes(X)
    assert X_filtered.shape[0] == X.shape[0]
    assert not np.any(X_filtered.sum(0) == 0)
    matrix.check_all_matrix_types(
        X, utils.check_output_equivalent,
        Y=X_filtered, transform=scprep.filter.remove_empty_genes)


def test_remove_rare_genes():
    X = data.load_10X(sparse=False)
    X_filtered = scprep.filter.remove_rare_genes(X)
    assert X_filtered.shape[0] == X.shape[0]
    assert not np.any(X_filtered.sum(0) < 5)
    matrix.check_all_matrix_types(
        X, utils.check_output_equivalent,
        Y=X_filtered, transform=scprep.filter.remove_rare_genes)


def test_library_size_filter():
    X = data.load_10X(sparse=True)
    X_filtered = scprep.filter.filter_library_size(X, 100)
    assert X_filtered.shape[1] == X.shape[1]
    assert not np.any(X_filtered.sum(1) < 100)
    matrix.check_all_matrix_types(
        X, utils.check_output_equivalent,
        Y=X_filtered, transform=partial(
            scprep.filter.filter_library_size, cutoff=100))


def test_gene_expression_filter():
    X = data.load_10X(sparse=True)
    genes = np.arange(10)
    X_filtered = scprep.filter.filter_gene_set_expression(
        X, genes, percentile=90, keep_cells='below')
    gene_cols = np.array(X.columns)[genes]
    assert X_filtered.shape[1] == X.shape[1]
    assert np.max(np.sum(X[gene_cols], axis=1)) > np.max(
        np.sum(X_filtered[gene_cols], axis=1))
    matrix.check_all_matrix_types(
        X, utils.check_output_equivalent,
        Y=X_filtered, transform=partial(
            scprep.filter.filter_gene_set_expression, genes=genes,
            percentile=90, keep_cells='below'))
    X_filtered = scprep.filter.filter_gene_set_expression(
        X, genes, percentile=10, keep_cells='above')
    assert X_filtered.shape[1] == X.shape[1]
    assert np.min(np.sum(X[gene_cols], axis=1)) < np.min(
        np.sum(X_filtered[gene_cols], axis=1))
    matrix.check_all_matrix_types(
        X, utils.check_output_equivalent,
        Y=X_filtered, transform=partial(
            scprep.filter.filter_gene_set_expression, genes=genes,
            percentile=10, keep_cells='above'))


def test_gene_expression_filter_warning():
    X = data.load_10X(sparse=True)
    genes = np.arange(10)
    gene_outside_range = 100
    no_genes = 'not_a_gene'
    assert_warns_message(
        UserWarning,
        "`percentile` expects values between 0 and 100."
        "Got 0.9. Did you mean 90.0?",
        scprep.filter.filter_gene_set_expression,
        X, genes, percentile=0.90, keep_cells='below')
    assert_raise_message(
        ValueError,
        "Only one of `cutoff` and `percentile` should be given.",
        scprep.filter.filter_gene_set_expression,
        X, genes, percentile=0.90, cutoff=50)
    assert_raise_message(
        ValueError,
        "Expected `keep_cells` in ['above', 'below']."
        "Got neither",
        scprep.filter.filter_gene_set_expression,
        X, genes, percentile=90.0, keep_cells='neither')
    assert_warns_message(
        UserWarning,
        "`percentile` expects values between 0 and 100."
        "Got 0.9. Did you mean 90.0?",
        scprep.filter.filter_gene_set_expression,
        X, genes, percentile=0.90, keep_cells='below')
    assert_raise_message(
        ValueError,
        "One of either `cutoff` or `percentile` must be given.",
        scprep.filter.filter_gene_set_expression,
        X, genes, cutoff=None, percentile=None)
    assert_raise_message(
        KeyError,
        "the label [not_a_gene] is not in the [columns]",
        scprep.filter.filter_gene_set_expression,
        X, no_genes, percentile=90.0, keep_cells='below')
    assert_warns_message(
        UserWarning,
        "Selecting 0 columns",
        scprep.utils.select_cols, X, (X.sum(axis=0) < 0))


def test_large_sparse_Xframe_library_size():
    X = pd.SparseXFrame(sparse.coo_matrix((10**7, 2 * 10**4)),
                        default_fill_value=0.0)
    cell_sums = scprep.filter.library_size(X)
    assert cell_sums.shape[0] == X.shape[0]
