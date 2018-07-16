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


def test_remove_empty_cells_sample_label():
    X = data.load_10X(sparse=False)
    sample_labels = np.arange(X.shape[0])
    sample_labels_filt = sample_labels[X.sum(1) > 0]
    X_filtered, sample_labels = scprep.filter.remove_empty_cells(
        X, sample_labels=sample_labels)
    assert X_filtered.shape[0] == len(sample_labels)
    assert np.all(sample_labels == sample_labels_filt)


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
    assert not np.any(X_filtered.sum(1) <= 100)
    matrix.check_all_matrix_types(
        X, utils.check_output_equivalent,
        Y=X_filtered, transform=partial(
            scprep.filter.filter_library_size, cutoff=100))


def test_library_size_filter_sample_label():
    X = data.load_10X(sparse=False)
    sample_labels = pd.DataFrame(np.arange(X.shape[0]), index=X.index)
    sample_labels_filt = sample_labels.loc[X.sum(1) > 100]
    X_filtered, sample_labels = scprep.filter.filter_library_size(
        X, cutoff=100, sample_labels=sample_labels)
    assert X_filtered.shape[0] == len(sample_labels)
    assert np.all(sample_labels == sample_labels_filt)


def test_gene_expression_filter_below():
    X = data.load_10X(sparse=True)
    genes = np.arange(10)
    X_filtered = scprep.filter.filter_gene_set_expression(
        X, genes, percentile=90, keep_cells='below',
        library_size_normalize=False)
    gene_cols = np.array(X.columns)[genes]
    assert X_filtered.shape[1] == X.shape[1]
    assert np.max(np.sum(X[gene_cols], axis=1)) > np.max(
        np.sum(X_filtered[gene_cols], axis=1))
    matrix.check_all_matrix_types(
        X, utils.check_output_equivalent,
        Y=X_filtered, transform=partial(
            scprep.filter.filter_gene_set_expression, genes=genes,
            percentile=90, keep_cells='below',
            library_size_normalize=False))


def test_gene_expression_filter_above():
    X = data.load_10X(sparse=True)
    genes = np.arange(10)
    gene_cols = np.array(X.columns)[genes]
    X_filtered = scprep.filter.filter_gene_set_expression(
        X, genes, percentile=10, keep_cells='above',
        library_size_normalize=False)
    assert X_filtered.shape[1] == X.shape[1]
    assert np.min(np.sum(X[gene_cols], axis=1)) < np.min(
        np.sum(X_filtered[gene_cols], axis=1))
    matrix.check_all_matrix_types(
        X, utils.check_output_equivalent,
        Y=X_filtered, transform=partial(
            scprep.filter.filter_gene_set_expression, genes=genes,
            percentile=10, keep_cells='above',
            library_size_normalize=False))


def test_gene_expression_libsize():
    X = data.load_10X(sparse=True)
    genes = np.arange(10)
    X_filtered = scprep.filter.filter_gene_set_expression(
        X, genes, percentile=10, keep_cells='above',
        library_size_normalize=True)
    X_libsize = scprep.normalize.library_size_normalize(X)
    Y = scprep.filter.filter_gene_set_expression(
        X_libsize, genes, percentile=10, keep_cells='above',
        library_size_normalize=False)
    assert X_filtered.shape == Y.shape
    assert np.all(X_filtered.index == Y.index)


def test_gene_expression_filter_sample_label():
    X = data.load_10X(sparse=False)
    genes = np.arange(10)
    sample_labels = pd.DataFrame(np.arange(X.shape[0]), index=X.index)
    X_filtered, sample_labels = scprep.filter.filter_gene_set_expression(
        X, genes, percentile=90, sample_labels=sample_labels)
    assert X_filtered.shape[0] == len(sample_labels)


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


def test_large_sparse_dataframe_library_size():
    X = pd.SparseDataFrame(sparse.coo_matrix((10**7, 2 * 10**4)),
                           default_fill_value=0.0)
    cell_sums = scprep.measure.library_size(X)
    assert cell_sums.shape[0] == X.shape[0]
