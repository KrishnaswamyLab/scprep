from tools import data, matrix, utils
import scprep
from sklearn.utils.testing import assert_raise_message, assert_warns_message
import numpy as np
import pandas as pd


def test_get_gene_set():
    X = data.load_10X()
    gene_idx = np.argwhere([g.startswith("D") for g in X.columns]).flatten()
    gene_names = X.columns[gene_idx]
    assert np.all(scprep.select.get_gene_set(X, starts_with="D") == gene_names)
    assert np.all(scprep.select.get_gene_set(X, regex="^D") == gene_names)
    assert np.all(scprep.select.get_gene_set(
        X.columns, regex="^D") == gene_names)
    gene_idx = np.argwhere([g.endswith("8") for g in X.columns]).flatten()
    gene_names = X.columns[gene_idx]
    assert np.all(scprep.select.get_gene_set(X, ends_with="8") == gene_names)
    assert np.all(scprep.select.get_gene_set(X, regex="8$") == gene_names)
    assert_raise_message(
        TypeError,
        "data must be a list of gene names or a pandas "
        "DataFrame. Got ndarray",
        scprep.select.get_gene_set,
        data=X.values, regex="8$")


def test_get_cell_set():
    X = data.load_10X()
    cell_idx = np.argwhere([g.startswith("A") for g in X.index]).flatten()
    cell_names = X.index[cell_idx]
    assert np.all(scprep.select.get_cell_set(X, starts_with="A") == cell_names)
    assert np.all(scprep.select.get_cell_set(X, regex="^A") == cell_names)
    assert np.all(scprep.select.get_cell_set(
        X.index, regex="^A") == cell_names)
    cell_idx = np.argwhere([g.endswith("G-1") for g in X.index]).flatten()
    cell_names = X.index[cell_idx]
    assert np.all(scprep.select.get_cell_set(X, ends_with="G-1") == cell_names)
    assert np.all(scprep.select.get_cell_set(X, regex="G\\-1$") == cell_names)
    assert_raise_message(
        TypeError,
        "data must be a list of cell names or a pandas "
        "DataFrame. Got ndarray",
        scprep.select.get_cell_set,
        data=X.values, regex="G\\-1$")


def test_select_rows():
    X = data.load_10X()
    # boolean array index
    matrix.test_all_matrix_types(
        X, scprep.select.select_rows,
        idx=np.random.choice([True, False], [X.shape[0]]))
    # integer array index
    matrix.test_all_matrix_types(
        X, scprep.select.select_rows,
        idx=np.random.choice(X.shape[0], X.shape[0] // 2))
    # integer list index
    matrix.test_all_matrix_types(
        X, scprep.select.select_rows,
        idx=np.random.choice(X.shape[0], X.shape[0] // 2).tolist())
    # integer index
    matrix.test_all_matrix_types(
        X, scprep.select.select_rows,
        idx=np.random.choice(X.shape[0]))
    # string array index
    matrix.test_pandas_matrix_types(
        X, scprep.select.select_rows,
        idx=np.random.choice(X.index.values, X.shape[0] // 2))
    # index index
    matrix.test_pandas_matrix_types(
        X, scprep.select.select_rows,
        idx=X.index[np.random.choice([True, False], [X.shape[0]])])
    # series index
    matrix.test_pandas_matrix_types(
        X, scprep.select.select_rows,
        idx=pd.Series(X.index[np.random.choice([True, False], [X.shape[0]])]))
    # dataframe index
    matrix.test_all_matrix_types(
        X, scprep.select.select_rows,
        idx=pd.DataFrame(np.random.choice([True, False], [X.shape[0], 1]),
                         index=X.index))
    # series data
    scprep.select.select_rows(
        X.iloc[:, 0], idx=np.random.choice([True, False], [X.shape[0]]))
    # 1D array data
    scprep.select.select_rows(
        X.to_coo().toarray()[:, 0], idx=np.random.choice([True, False], [X.shape[0]]))
    # get_cell_set
    matrix.test_pandas_matrix_types(
        X, scprep.select.select_rows, X.iloc[:, 0],
        starts_with="A")


def test_select_cols():
    X = data.load_10X()
    # boolean array index
    matrix.test_all_matrix_types(
        X, scprep.select.select_cols,
        idx=np.random.choice([True, False], [X.shape[1]]))
    # integer array index
    matrix.test_all_matrix_types(
        X, scprep.select.select_cols,
        idx=np.random.choice(X.shape[1], X.shape[1] // 2))
    # integer list index
    matrix.test_all_matrix_types(
        X, scprep.select.select_cols,
        idx=np.random.choice(X.shape[1], X.shape[1] // 2).tolist())
    # integer index
    matrix.test_all_matrix_types(
        X, scprep.select.select_cols,
        idx=np.random.choice(X.shape[1]))
    # string array index
    matrix.test_pandas_matrix_types(
        X, scprep.select.select_cols,
        idx=np.random.choice(X.columns.values, X.shape[1] // 2))
    # index index
    matrix.test_pandas_matrix_types(
        X, scprep.select.select_cols,
        idx=X.columns[np.random.choice([True, False], [X.shape[1]])])
    # series index
    matrix.test_pandas_matrix_types(
        X, scprep.select.select_cols,
        idx=pd.Series(X.columns[np.random.choice([True, False], [X.shape[1]])]))
    # dataframe index
    matrix.test_all_matrix_types(
        X, scprep.select.select_cols,
        idx=pd.DataFrame(np.random.choice([True, False], [1, X.shape[1]]),
                         index=[1], columns=X.columns))
    # series data
    scprep.select.select_cols(
        X.iloc[0, :], idx=np.random.choice([True, False], [X.shape[1]]))
    # 1D array data
    scprep.select.select_cols(
        X.to_coo().toarray()[0, :], idx=np.random.choice([True, False], [X.shape[1]]))
    # get_gene_set
    matrix.test_pandas_matrix_types(
        X, scprep.select.select_cols, X.iloc[0, :],
        starts_with="D")
    assert_warns_message(
        UserWarning,
        "Selecting 0 columns",
        scprep.select.select_cols, X,
        idx=(X.sum(axis=0) < 0))
    assert_warns_message(
        UserWarning,
        "No selection conditions provided. Returning all columns.",
        scprep.select.select_cols, X)


def test_select_error():
    X = data.load_10X()
    assert_raise_message(KeyError,
                         "the label [not_a_cell] is not in the [index]",
                         scprep.select.select_rows,
                         X,
                         idx='not_a_cell')
    assert_raise_message(KeyError,
                         "the label [not_a_gene] is not in the [columns]",
                         scprep.select.select_cols,
                         X,
                         idx='not_a_gene')
    assert_raise_message(ValueError,
                         "Expected idx to be 1D. Got shape ",
                         scprep.select.select_rows,
                         X,
                         idx=pd.DataFrame([X.index, X.index]))
    assert_raise_message(ValueError,
                         "Expected idx to be 1D. Got shape ",
                         scprep.select.select_cols,
                         X,
                         idx=pd.DataFrame([X.index, X.index]))
    assert_raise_message(
        ValueError,
        "Expected all pandas inputs to have the same columns. "
        "Fix with "
        "`scprep.select.select_cols(extra_data, data.columns)`",
        scprep.select.select_cols,
        X,
        scprep.select.subsample(X.T, n=X.shape[0]).T)
    assert_raise_message(
        ValueError,
        "Expected all pandas inputs to have the same index. "
        "Fix with "
        "`scprep.select.select_rows(extra_data, data.index)`",
        scprep.select.select_rows,
        X,
        scprep.select.subsample(X, n=X.shape[0]))


def test_subsample():
    X = data.generate_positive_sparse_matrix(shape=(50, 100))
    Y = scprep.select.subsample(X, n=20, seed=42)
    matrix.test_all_matrix_types(
        X, utils.assert_transform_equals, Y=Y,
        transform=scprep.select.subsample,
        check=utils.assert_all_equal, n=20, seed=42)
    libsize = scprep.measure.library_size(X)
    Y, libsize_sub = scprep.select.subsample(X, libsize, n=20, seed=42)

    def test_fun(X, **kwargs):
        libsize = scprep.measure.library_size(X)
        return scprep.select.subsample(X, libsize, **kwargs)[0]
    matrix.test_all_matrix_types(
        X, utils.assert_transform_equals, Y=Y,
        transform=test_fun,
        check=utils.assert_all_equal, n=20, seed=42)

    def test_fun(X, **kwargs):
        libsize = scprep.measure.library_size(X)
        return scprep.select.subsample(X, libsize, **kwargs)[1]
    matrix.test_all_matrix_types(
        X, utils.assert_transform_equals, Y=libsize_sub,
        transform=test_fun,
        check=utils.assert_all_close, n=20, seed=42)


def test_subsample_mismatch_size():
    X = data.generate_positive_sparse_matrix(shape=(50, 100))
    libsize = scprep.measure.library_size(X)[:20]
    assert_raise_message(
        ValueError,
        "Expected all data to have the same number of rows. "
        "Got [50, 20]",
        scprep.select.subsample, X, libsize, n=20)


def test_subsample_n_too_large():
    X = data.generate_positive_sparse_matrix(shape=(50, 100))
    assert_raise_message(
        ValueError,
        "Expected n (60) <= n_samples (50)",
        scprep.select.subsample, X, n=60)
