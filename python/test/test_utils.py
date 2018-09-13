from load_tests import data, matrix, utils
import scprep
from sklearn.utils.testing import assert_raise_message, assert_warns_message
import numpy as np
import pandas as pd


def test_get_gene_set():
    X = data.load_10X()
    gene_idx = np.argwhere([g.startswith("D") for g in X.columns]).flatten()
    gene_names = X.columns[gene_idx]
    assert np.all(scprep.utils.get_gene_set(X, starts_with="D") == gene_names)
    assert np.all(scprep.utils.get_gene_set(X, regex="^D") == gene_names)
    assert np.all(scprep.utils.get_gene_set(
        X.columns, regex="^D") == gene_names)
    gene_idx = np.argwhere([g.endswith("8") for g in X.columns]).flatten()
    gene_names = X.columns[gene_idx]
    assert np.all(scprep.utils.get_gene_set(X, ends_with="8") == gene_names)
    assert np.all(scprep.utils.get_gene_set(X, regex="8$") == gene_names)
    assert_raise_message(
        TypeError,
        "data must be a list of gene names or a pandas "
        "DataFrame. Got ndarray",
        scprep.utils.get_gene_set,
        data=X.values, regex="8$")


def test_get_cell_set():
    X = data.load_10X()
    cell_idx = np.argwhere([g.startswith("A") for g in X.index]).flatten()
    cell_names = X.index[cell_idx]
    assert np.all(scprep.utils.get_cell_set(X, starts_with="A") == cell_names)
    assert np.all(scprep.utils.get_cell_set(X, regex="^A") == cell_names)
    assert np.all(scprep.utils.get_cell_set(
        X.index, regex="^A") == cell_names)
    cell_idx = np.argwhere([g.endswith("G-1") for g in X.index]).flatten()
    cell_names = X.index[cell_idx]
    assert np.all(scprep.utils.get_cell_set(X, ends_with="G-1") == cell_names)
    assert np.all(scprep.utils.get_cell_set(X, regex="G\\-1$") == cell_names)
    assert_raise_message(
        TypeError,
        "data must be a list of cell names or a pandas "
        "DataFrame. Got ndarray",
        scprep.utils.get_cell_set,
        data=X.values, regex="G\\-1$")


def test_combine_batches():
    X = data.load_10X()
    Y = pd.concat([X, scprep.utils.select_rows(
        X, np.arange(X.shape[0] // 2))])
    Y2, sample_labels = scprep.utils.combine_batches(
        [X, scprep.utils.select_rows(
            X, np.arange(X.shape[0] // 2))],
        batch_labels=[0, 1])
    assert utils.matrix_class_equivalent(Y, Y2)
    utils.assert_all_equal(Y, Y2)
    assert np.all(Y.index == Y2.index)
    assert np.all(sample_labels == np.concatenate(
        [np.repeat(0, X.shape[0]), np.repeat(1, X.shape[0] // 2)]))
    Y2, sample_labels = scprep.utils.combine_batches(
        [X, scprep.utils.select_rows(
            X, np.arange(X.shape[0] // 2))],
        batch_labels=[0, 1],
        append_to_cell_names=True)
    assert np.all(Y.index == np.array([i[:-2] for i in Y2.index]))
    assert np.all(np.core.defchararray.add(
        "_", sample_labels.astype(str)) == np.array(
        [i[-2:] for i in Y2.index], dtype=str))
    transform = lambda X: scprep.utils.combine_batches(
        [X, scprep.utils.select_rows(X, np.arange(X.shape[0] // 2))],
        batch_labels=[0, 1])[0]
    matrix.check_matrix_types(
        X,
        utils.check_transform_equivalent,
        matrix._indexable_matrix_types,
        Y=Y,
        transform=transform,
        check=utils.assert_all_equal,
        matrix_form_unchanged=False)


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
        [X, scprep.utils.select_cols(X, np.arange(X.shape[1] // 2))],
        batch_labels=[0, 1])
    assert_raise_message(
        ValueError,
        "Expected data (2) and batch_labels (1) to be the same length.",
        scprep.utils.combine_batches,
        [X, scprep.utils.select_rows(X, np.arange(X.shape[0] // 2))],
        batch_labels=[0])
    assert_raise_message(
        ValueError,
        "Expected data to contain pandas DataFrames, "
        "scipy sparse matrices or numpy arrays. Got str",
        scprep.utils.combine_batches,
        ["hello", "world"],
        batch_labels=[0, 1])


def test_select_error():
    X = data.load_10X()
    assert_raise_message(KeyError,
                         "the label [not_a_cell] is not in the [index]",
                         scprep.utils.select_rows,
                         X,
                         'not_a_cell')
    assert_raise_message(KeyError,
                         "the label [not_a_gene] is not in the [columns]",
                         scprep.utils.select_cols,
                         X,
                         'not_a_gene')
    scprep.utils.select_rows(
        X, pd.DataFrame(np.random.choice([True, False], [X.shape[0], 1]),
                        index=X.index))
    assert_raise_message(ValueError,
                         "Expected idx to be 1D. Got shape ",
                         scprep.utils.select_rows,
                         X,
                         pd.DataFrame([X.index, X.index]))


def test_toarray():
    X = data.generate_positive_sparse_matrix(shape=(50, 50))

    def test_fun(X):
        assert isinstance(scprep.utils.toarray(X), np.ndarray)
    matrix.check_all_matrix_types(X,
                                  test_fun)
    test_fun(np.matrix(X))
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
    matrix.check_all_matrix_types(X,
                                  test_fun)
    test_fun(np.matrix(X))
    sums = np.array(X.sum(1)).flatten()

    def test_fun(X):
        assert np.allclose(
            np.array(scprep.utils.matrix_sum(X, axis=1)), sums)
    matrix.check_all_matrix_types(X,
                                  test_fun)
    test_fun(np.matrix(X))
    sums = np.array(X.sum(None)).flatten()

    def test_fun(X):
        assert np.allclose(scprep.utils.matrix_sum(X, axis=None), sums)
    matrix.check_all_matrix_types(X,
                                  test_fun)
    test_fun(np.matrix(X))
    assert_raise_message(ValueError,
                         "Expected axis in [0, 1, None]. Got 5",
                         scprep.utils.matrix_sum,
                         data,
                         5)
