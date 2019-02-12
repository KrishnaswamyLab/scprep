from tools import data, matrix, utils
import scprep
from sklearn.utils.testing import assert_raise_message, assert_warns_message
import numpy as np
import pandas as pd


def test_combine_batches():
    X = data.load_10X()
    Y = pd.concat([X, scprep.select.select_rows(
        X, idx=np.arange(X.shape[0] // 2))])
    Y2, sample_labels = scprep.utils.combine_batches(
        [X, scprep.select.select_rows(
            X, idx=np.arange(X.shape[0] // 2))],
        batch_labels=[0, 1])
    assert utils.assert_matrix_class_equivalent(Y, Y2)
    utils.assert_all_equal(Y, Y2)
    assert np.all(Y.index == Y2.index)
    assert np.all(sample_labels == np.concatenate(
        [np.repeat(0, X.shape[0]), np.repeat(1, X.shape[0] // 2)]))
    Y2, sample_labels = scprep.utils.combine_batches(
        [X, scprep.select.select_rows(
            X, idx=np.arange(X.shape[0] // 2))],
        batch_labels=[0, 1],
        append_to_cell_names=True)
    assert np.all(Y.index == np.array([i[:-2] for i in Y2.index]))
    assert np.all(np.core.defchararray.add(
        "_", sample_labels.astype(str)) == np.array(
        [i[-2:] for i in Y2.index], dtype=str))
    transform = lambda X: scprep.utils.combine_batches(
        [X, scprep.select.select_rows(X, idx=np.arange(X.shape[0] // 2))],
        batch_labels=[0, 1])[0]
    matrix.test_matrix_types(
        X,
        utils.assert_transform_equals,
        matrix._indexable_matrix_types,
        Y=Y,
        transform=transform,
        check=utils.assert_all_equal)


def test_combine_batches_errors():
    X = data.load_10X()
    assert_warns_message(
        UserWarning,
        "append_to_cell_names only valid for pd.DataFrame input. "
        "Got coo_matrix",
        scprep.utils.combine_batches,
        [X.to_coo(), X.iloc[:X.shape[0] // 2].to_coo()],
        batch_labels=[0, 1],
        append_to_cell_names=True)
    assert_raise_message(
        TypeError,
        "Expected data all of the same class. Got SparseDataFrame, coo_matrix",
        scprep.utils.combine_batches,
        [X, X.iloc[:X.shape[0] // 2].to_coo()],
        batch_labels=[0, 1])
    assert_raise_message(
        ValueError,
        "Expected data all with the same number of columns. "
        "Got {}, {}".format(X.shape[1], X.shape[1] // 2),
        scprep.utils.combine_batches,
        [X, scprep.select.select_cols(X, idx=np.arange(X.shape[1] // 2))],
        batch_labels=[0, 1])
    assert_raise_message(
        ValueError,
        "Expected data (2) and batch_labels (1) to be the same length.",
        scprep.utils.combine_batches,
        [X, scprep.select.select_rows(X, idx=np.arange(X.shape[0] // 2))],
        batch_labels=[0])
    assert_raise_message(
        ValueError,
        "Expected data to contain pandas DataFrames, "
        "scipy sparse matrices or numpy arrays. Got str",
        scprep.utils.combine_batches,
        ["hello", "world"],
        batch_labels=[0, 1])


def test_matrix_any():
    X = data.generate_positive_sparse_matrix(shape=(50, 50))
    assert not np.any(X == 500000)

    def test_fun(X):
        assert not scprep.utils.matrix_any(X == 500000)
    matrix.test_all_matrix_types(X,
                                 test_fun)

    def test_fun(X):
        assert scprep.utils.matrix_any(X == 500000)
    X[0, 0] = 500000
    matrix.test_all_matrix_types(X,
                                 test_fun)


def test_toarray():
    X = data.generate_positive_sparse_matrix(shape=(50, 50))

    def test_fun(X):
        assert isinstance(scprep.utils.toarray(X), np.ndarray)
    matrix.test_all_matrix_types(X,
                                 test_fun)
    test_fun([X, np.matrix(X)])
    assert_raise_message(TypeError,
                         "Expected pandas DataFrame, scipy sparse matrix or "
                         "numpy matrix. Got ",
                         scprep.utils.toarray,
                         "hello")


def test_matrix_sum():
    X = data.generate_positive_sparse_matrix(shape=(50, 100))
    sums = np.array(X.sum(0)).flatten()

    def test_fun(X):
        assert np.allclose(np.array(scprep.utils.matrix_sum(X, axis=0)), sums)
    matrix.test_all_matrix_types(X,
                                 test_fun)
    test_fun(np.matrix(X))
    sums = np.array(X.sum(1)).flatten()

    def test_fun(X):
        assert np.allclose(
            np.array(scprep.utils.matrix_sum(X, axis=1)), sums)
    matrix.test_all_matrix_types(X,
                                 test_fun)
    test_fun(np.matrix(X))
    sums = np.array(X.sum(None)).flatten()

    def test_fun(X):
        assert np.allclose(scprep.utils.matrix_sum(X, axis=None), sums)
    matrix.test_all_matrix_types(X,
                                 test_fun)
    test_fun(np.matrix(X))
    assert_raise_message(ValueError,
                         "Expected axis in [0, 1, None]. Got 5",
                         scprep.utils.matrix_sum,
                         data,
                         5)


def test_deprecated():
    X = data.load_10X()
    assert_warns_message(DeprecationWarning,
                         "`scprep.utils.select_cols` is deprecated. Use "
                         "`scprep.select.select_cols` instead.",
                         scprep.utils.select_cols,
                         X,
                         [1, 2, 3])
    assert_warns_message(DeprecationWarning,
                         "`scprep.utils.select_rows` is deprecated. Use "
                         "`scprep.select.select_rows` instead.",
                         scprep.utils.select_rows,
                         X,
                         [1, 2, 3])
    assert_warns_message(DeprecationWarning,
                         "`scprep.utils.get_gene_set` is deprecated. Use "
                         "`scprep.select.get_gene_set` instead.",
                         scprep.utils.get_gene_set,
                         X,
                         starts_with="D")
    assert_warns_message(DeprecationWarning,
                         "`scprep.utils.get_cell_set` is deprecated. Use "
                         "`scprep.select.get_cell_set` instead.",
                         scprep.utils.get_cell_set,
                         X,
                         starts_with="A")
    assert_warns_message(DeprecationWarning,
                         "`scprep.utils.subsample` is deprecated. Use "
                         "`scprep.select.subsample` instead.",
                         scprep.utils.subsample,
                         X,
                         n=10)
