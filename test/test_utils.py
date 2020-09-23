from tools import data, matrix, utils
import scprep
from scipy import sparse
import numpy as np
import pandas as pd
from parameterized import parameterized


def test_with_pkg():
    @scprep.utils._with_pkg(pkg="invalid")
    def invalid():
        pass

    utils.assert_raises_message(
        ImportError,
        "invalid not found. Please install it with e.g. "
        "`pip install --user invalid`",
        invalid,
    )


def test_with_pkg_version_none():
    @scprep.utils._with_pkg(pkg="numpy")
    def test():
        return True

    assert test()


def test_with_pkg_version_exact():
    major, minor = [int(v) for v in np.__version__.split(".")[:2]]

    @scprep.utils._with_pkg(pkg="numpy", min_version="{}.{}".format(major, minor))
    def test():
        return True

    assert test()


def test_with_pkg_version_exact_no_minor():
    major, minor = [int(v) for v in np.__version__.split(".")[:2]]

    @scprep.utils._with_pkg(pkg="numpy", min_version=major)
    def test():
        return True

    assert test()


def test_with_pkg_version_pass_major():
    major, minor = [int(v) for v in np.__version__.split(".")[:2]]

    @scprep.utils._with_pkg(pkg="numpy", min_version=major - 1)
    def test():
        return True

    assert test()


def test_with_pkg_version_pass_minor():
    major, minor = [int(v) for v in np.__version__.split(".")[:2]]

    @scprep.utils._with_pkg(pkg="numpy", min_version="{}.{}".format(major, minor - 1))
    def test():
        return True

    assert test()


def test_with_pkg_version_fail_major():
    major, minor = [int(v) for v in np.__version__.split(".")[:2]]

    @scprep.utils._with_pkg(pkg="numpy", min_version=major + 1)
    def test():
        return True

    utils.assert_raises_message(
        ImportError,
        "numpy>={0} is required (installed: {1}). "
        "Please upgrade it with e.g."
        " `pip install --user --upgrade numpy".format(major + 1, np.__version__),
        test,
    )


def test_with_pkg_version_fail_minor():
    major, minor = [int(v) for v in np.__version__.split(".")[:2]]

    @scprep.utils._with_pkg(pkg="numpy", min_version="{}.{}".format(major, minor + 1))
    def test():
        return True

    utils.assert_raises_message(
        ImportError,
        "numpy>={0}.{1} is required (installed: {2}). "
        "Please upgrade it with e.g."
        " `pip install --user --upgrade numpy".format(major, minor + 1, np.__version__),
        test,
    )


def test_with_pkg_version_memoize():
    major, minor = [int(v) for v in np.__version__.split(".")[:2]]
    min_version = "{}.{}".format(major, minor + 1)

    @scprep.utils._with_pkg(pkg="numpy", min_version=min_version)
    def test():
        return True

    true_version = np.__version__
    np.__version__ = min_version
    # should pass
    assert test()
    np.__version__ = true_version
    # should fail, but already memoized
    assert test()


def test_try_import():
    assert scprep.utils._try_import("invalid") is None
    assert scprep.utils._try_import("numpy") is np


def test_combine_batches():
    X = data.load_10X()
    Y = pd.concat(
        [X, scprep.select.select_rows(X, idx=np.arange(X.shape[0] // 2))],
        axis=0,
        sort=True,
    )
    Y2, sample_labels = scprep.utils.combine_batches(
        [X, scprep.select.select_rows(X, idx=np.arange(X.shape[0] // 2))],
        batch_labels=[0, 1],
        append_to_cell_names=False,
    )
    assert utils.assert_matrix_class_equivalent(Y, Y2)
    utils.assert_all_equal(Y, Y2)
    assert np.all(Y.index == Y2.index)
    assert np.all(
        sample_labels
        == np.concatenate([np.repeat(0, X.shape[0]), np.repeat(1, X.shape[0] // 2)])
    )
    assert np.all(sample_labels.index == Y2.index)
    assert sample_labels.name == "sample_labels"
    Y2, sample_labels = scprep.utils.combine_batches(
        [X, scprep.select.select_rows(X, idx=np.arange(X.shape[0] // 2))],
        batch_labels=[0, 1],
        append_to_cell_names=True,
    )
    assert np.all(Y.index == np.array([i[:-2] for i in Y2.index]))
    assert np.all(
        np.core.defchararray.add("_", sample_labels.astype(str))
        == np.array([i[-2:] for i in Y2.index], dtype=str)
    )
    assert np.all(sample_labels.index == Y2.index)
    assert sample_labels.name == "sample_labels"
    transform = lambda X: scprep.utils.combine_batches(
        [X, scprep.select.select_rows(X, idx=np.arange(X.shape[0] // 2))],
        batch_labels=[0, 1],
    )[0]
    matrix.test_matrix_types(
        X,
        utils.assert_transform_equals,
        matrix._pandas_matrix_types,
        Y=Y,
        transform=transform,
        check=utils.assert_all_equal,
    )
    # don't sort for non pandas
    Y = pd.concat(
        [X, scprep.select.select_rows(X, idx=np.arange(X.shape[0] // 2))],
        axis=0,
        sort=False,
    )
    matrix.test_matrix_types(
        X,
        utils.assert_transform_equals,
        matrix._scipy_indexable_matrix_types + matrix._numpy_matrix_types,
        Y=Y,
        transform=transform,
        check=utils.assert_all_equal,
    )

    def test_fun(X):
        Y, sample_labels = scprep.utils.combine_batches(
            [X, scprep.select.select_rows(X, idx=np.arange(X.shape[0] // 2))],
            batch_labels=[0, 1],
        )
        assert np.all(sample_labels.index == Y.index)
        assert sample_labels.name == "sample_labels"

    matrix.test_pandas_matrix_types(X, test_fun)


def test_combine_batches_rangeindex():
    X = data.load_10X()
    X = X.reset_index(drop=True)
    Y = X.iloc[: X.shape[0] // 2]
    data_combined, labels = scprep.utils.combine_batches([X, Y], ["x", "y"])
    assert isinstance(data_combined.index, pd.RangeIndex)
    assert np.all(np.sort(data_combined.columns) == np.sort(X.columns))
    assert np.all(
        data_combined.iloc[:100][np.sort(X.columns)].to_numpy()
        == X[np.sort(X.columns)].to_numpy()
    )
    assert np.all(
        data_combined.iloc[100:][np.sort(X.columns)].to_numpy()
        == Y[np.sort(X.columns)].to_numpy()
    )


@parameterized([(True,), (False,)])
def test_combine_batches_uncommon_genes(sparse):
    X = data.load_10X(sparse=sparse)
    Y = X.iloc[:, : X.shape[1] // 2]
    utils.assert_warns_message(
        UserWarning,
        "Input data has inconsistent column names. "
        "Subsetting to {} common columns.".format(Y.shape[1]),
        scprep.utils.combine_batches,
        [X, Y],
        ["x", "y"],
    )
    utils.assert_warns_message(
        UserWarning,
        "Input data has inconsistent column names. "
        "Padding with zeros to {} total columns.".format(X.shape[1]),
        scprep.utils.combine_batches,
        [X, Y],
        ["x", "y"],
        common_columns_only=False,
    )


def test_combine_batches_errors():
    X = data.load_10X()
    utils.assert_warns_message(
        UserWarning,
        "append_to_cell_names only valid for pd.DataFrame input. " "Got coo_matrix",
        scprep.utils.combine_batches,
        [X.sparse.to_coo(), X.iloc[: X.shape[0] // 2].sparse.to_coo()],
        batch_labels=[0, 1],
        append_to_cell_names=True,
    )
    utils.assert_raises_message(
        TypeError,
        "Expected data all of the same class. Got DataFrame, coo_matrix",
        scprep.utils.combine_batches,
        [X, X.iloc[: X.shape[0] // 2].sparse.to_coo()],
        batch_labels=[0, 1],
    )
    utils.assert_raises_message(
        ValueError,
        "Expected data all with the same number of columns. "
        "Got {}, {}".format(X.shape[1], X.shape[1] // 2),
        scprep.utils.combine_batches,
        [
            scprep.utils.toarray(X),
            scprep.select.select_cols(
                scprep.utils.toarray(X), idx=np.arange(X.shape[1] // 2)
            ),
        ],
        batch_labels=[0, 1],
    )
    utils.assert_raises_message(
        ValueError,
        "Expected data (2) and batch_labels (1) to be the same length.",
        scprep.utils.combine_batches,
        [X, scprep.select.select_rows(X, idx=np.arange(X.shape[0] // 2))],
        batch_labels=[0],
    )
    utils.assert_raises_message(
        ValueError,
        "Expected data to contain pandas DataFrames, "
        "scipy sparse matrices or numpy arrays. Got str",
        scprep.utils.combine_batches,
        ["hello", "world"],
        batch_labels=[0, 1],
    )


def test_matrix_any():
    X = data.generate_positive_sparse_matrix(shape=(50, 50))
    assert not np.any(X == 500000)

    def test_fun(X):
        assert not scprep.utils.matrix_any(X == 500000)

    matrix.test_all_matrix_types(X, test_fun)

    def test_fun(X):
        assert scprep.utils.matrix_any(X == 500000)

    X[0, 0] = 500000
    matrix.test_all_matrix_types(X, test_fun)


def test_toarray():
    X = data.generate_positive_sparse_matrix(shape=(50, 50))

    def test_fun(X):
        assert isinstance(scprep.utils.toarray(X), np.ndarray)

    matrix.test_all_matrix_types(X, test_fun)
    test_fun([X, np.matrix(X)])


def test_toarray_string_error():
    utils.assert_raises_message(
        TypeError, "Expected array-like. Got ", scprep.utils.toarray, "hello"
    )


def test_toarray_vector():
    X = data.generate_positive_sparse_matrix(shape=(50,))

    def test_fun(X):
        assert isinstance(scprep.utils.toarray(X), np.ndarray)

    matrix.test_matrix_types(X, test_fun, matrix._pandas_vector_types)


def test_toarray_list_of_strings():
    X = ["hello", "world", [1, 2, 3]]
    X = scprep.utils.toarray(X)
    assert isinstance(X[2], np.ndarray)


def test_to_array_or_spmatrix_list_of_strings():
    X = data.generate_positive_sparse_matrix(shape=(50, 50))
    X = scprep.utils.to_array_or_spmatrix(
        [X, sparse.csr_matrix(X), "hello", "world", [1, 2, 3]]
    )
    assert isinstance(X[0], np.ndarray)
    assert isinstance(X[1], sparse.csr_matrix)
    assert isinstance(X[4], np.ndarray)


def test_matrix_sum():
    X = data.generate_positive_sparse_matrix(shape=(50, 100))
    sums = np.array(X.sum(0)).flatten()
    matrix.test_all_matrix_types(
        X,
        utils.assert_transform_equals,
        Y=sums,
        transform=scprep.utils.matrix_sum,
        axis=0,
        check=utils.assert_all_close,
    )
    matrix.test_numpy_matrix(
        X,
        utils.assert_transform_equals,
        Y=sums,
        transform=scprep.utils.matrix_sum,
        axis=0,
        check=utils.assert_all_close,
    )

    sums = np.array(X.sum(1)).flatten()
    matrix.test_all_matrix_types(
        X,
        utils.assert_transform_equals,
        Y=sums,
        transform=scprep.utils.matrix_sum,
        axis=1,
        check=utils.assert_all_close,
    )
    matrix.test_numpy_matrix(
        X,
        utils.assert_transform_equals,
        Y=sums,
        transform=scprep.utils.matrix_sum,
        axis=1,
        check=utils.assert_all_close,
    )

    sums = np.array(X.sum(None)).flatten()
    matrix.test_all_matrix_types(
        X,
        utils.assert_transform_equals,
        Y=sums,
        transform=scprep.utils.matrix_sum,
        axis=None,
        check=utils.assert_all_close,
    )
    matrix.test_numpy_matrix(
        X,
        utils.assert_transform_equals,
        Y=sums,
        transform=scprep.utils.matrix_sum,
        axis=None,
        check=utils.assert_all_close,
    )

    utils.assert_raises_message(
        ValueError,
        "Expected axis in [0, 1, None]. Got 5",
        scprep.utils.matrix_sum,
        data,
        5,
    )


def test_matrix_std():
    X = data.generate_positive_sparse_matrix(shape=(50, 100))
    stds = np.array(X.std(0)).flatten()
    matrix.test_all_matrix_types(
        X,
        utils.assert_transform_equals,
        Y=stds,
        transform=scprep.utils.matrix_std,
        axis=0,
        check=utils.assert_all_close,
    )
    matrix.test_numpy_matrix(
        X,
        utils.assert_transform_equals,
        Y=stds,
        transform=scprep.utils.matrix_std,
        axis=0,
        check=utils.assert_all_close,
    )

    stds = np.array(X.std(1)).flatten()
    matrix.test_all_matrix_types(
        X,
        utils.assert_transform_equals,
        Y=stds,
        transform=scprep.utils.matrix_std,
        axis=1,
        check=utils.assert_all_close,
    )
    matrix.test_numpy_matrix(
        X,
        utils.assert_transform_equals,
        Y=stds,
        transform=scprep.utils.matrix_std,
        axis=1,
        check=utils.assert_all_close,
    )

    stds = np.array(X.std(None)).flatten()
    matrix.test_all_matrix_types(
        X,
        utils.assert_transform_equals,
        Y=stds,
        transform=scprep.utils.matrix_std,
        axis=None,
        check=utils.assert_all_close,
    )
    matrix.test_numpy_matrix(
        X,
        utils.assert_transform_equals,
        Y=stds,
        transform=scprep.utils.matrix_std,
        axis=None,
        check=utils.assert_all_close,
    )

    X_df = pd.DataFrame(
        X,
        index=np.arange(X.shape[0]).astype(str),
        columns=np.arange(X.shape[1]).astype(str),
    )

    def test_fun(X):
        x = scprep.utils.matrix_std(X, axis=0)
        assert x.name == "std"
        assert np.all(x.index == X_df.columns)
        x = scprep.utils.matrix_std(X, axis=1)
        assert x.name == "std"
        assert np.all(x.index == X_df.index)

    matrix.test_pandas_matrix_types(X_df, test_fun)
    utils.assert_raises_message(
        ValueError,
        "Expected axis in [0, 1, None]. Got 5",
        scprep.utils.matrix_std,
        data,
        5,
    )


def test_matrix_elementwise_multiply_row():
    X = data.generate_positive_sparse_matrix(shape=(50, 100))
    x = X[:, 0] + 1
    Y = pd.DataFrame(X).mul(x, axis=0)
    matrix.test_all_matrix_types(
        X,
        utils.assert_transform_equivalent,
        Y=Y,
        transform=scprep.utils.matrix_vector_elementwise_multiply,
        check=utils.assert_all_close,
        axis=0,
        multiplier=x,
    )


def test_matrix_elementwise_multiply_col():
    X = data.generate_positive_sparse_matrix(shape=(50, 100))
    x = X[0] + 1
    Y = pd.DataFrame(X).mul(x, axis=1)
    matrix.test_all_matrix_types(
        X,
        utils.assert_transform_equivalent,
        Y=Y,
        transform=scprep.utils.matrix_vector_elementwise_multiply,
        check=utils.assert_all_close,
        axis=1,
        multiplier=x,
    )


def test_matrix_elementwise_multiply_guess_row():
    X = data.generate_positive_sparse_matrix(shape=(50, 100))
    x = X[:, 0] + 1
    Y = pd.DataFrame(X).mul(x, axis=0)
    matrix.test_all_matrix_types(
        X,
        utils.assert_transform_equivalent,
        Y=Y,
        transform=scprep.utils.matrix_vector_elementwise_multiply,
        check=utils.assert_all_close,
        axis=None,
        multiplier=x,
    )


def test_matrix_elementwise_multiply_guess_col():
    X = data.generate_positive_sparse_matrix(shape=(50, 100))
    x = X[0] + 1
    Y = pd.DataFrame(X).mul(x, axis=1)
    matrix.test_all_matrix_types(
        X,
        utils.assert_transform_equivalent,
        Y=Y,
        transform=scprep.utils.matrix_vector_elementwise_multiply,
        check=utils.assert_all_close,
        axis=None,
        multiplier=x,
    )


def test_matrix_elementwise_multiply_square_guess():
    X = data.generate_positive_sparse_matrix(shape=(50, 50))
    utils.assert_raises_message(
        RuntimeError,
        "`data` is square, cannot guess axis from input. Please provide "
        "`axis=0` to multiply along rows or "
        "`axis=1` to multiply along columns.",
        scprep.utils.matrix_vector_elementwise_multiply,
        X,
        X[0],
    )


def test_matrix_elementwise_multiply_row_wrong_size():
    X = data.generate_positive_sparse_matrix(shape=(50, 100))
    utils.assert_raises_message(
        ValueError,
        "Expected `multiplier` to be a vector of length `data.shape[0]` (50)."
        " Got (100,)",
        scprep.utils.matrix_vector_elementwise_multiply,
        X,
        X[0],
        axis=0,
    )


def test_matrix_elementwise_multiply_col_wrong_size():
    X = data.generate_positive_sparse_matrix(shape=(50, 100))
    utils.assert_raises_message(
        ValueError,
        "Expected `multiplier` to be a vector of length `data.shape[1]` (100)."
        " Got (50,)",
        scprep.utils.matrix_vector_elementwise_multiply,
        X,
        X[:, 0],
        axis=1,
    )


def test_matrix_elementwise_multiply_guess_wrong_size():
    X = data.generate_positive_sparse_matrix(shape=(50, 100))
    utils.assert_raises_message(
        ValueError,
        "Expected `multiplier` to be a vector of length `data.shape[0]` (50) "
        "or `data.shape[1]` (100). Got (10,)",
        scprep.utils.matrix_vector_elementwise_multiply,
        X,
        X[0, :10],
    )


def test_matrix_elementwise_multiply_invalid_axis():
    X = data.generate_positive_sparse_matrix(shape=(50, 100))
    utils.assert_raises_message(
        ValueError,
        "Expected axis in [0, 1, None]. Got 5",
        scprep.utils.matrix_vector_elementwise_multiply,
        X,
        X[0],
        axis=5,
    )


def test_deprecated():
    X = data.load_10X()
    utils.assert_raises_message(
        RuntimeError,
        "`scprep.utils.select_cols` is deprecated. Use "
        "`scprep.select.select_cols` instead.",
        scprep.utils.select_cols,
        X,
        [1, 2, 3],
    )
    utils.assert_raises_message(
        RuntimeError,
        "`scprep.utils.select_rows` is deprecated. Use "
        "`scprep.select.select_rows` instead.",
        scprep.utils.select_rows,
        X,
        [1, 2, 3],
    )
    utils.assert_raises_message(
        RuntimeError,
        "`scprep.utils.get_gene_set` is deprecated. Use "
        "`scprep.select.get_gene_set` instead.",
        scprep.utils.get_gene_set,
        X,
        starts_with="D",
    )
    utils.assert_raises_message(
        RuntimeError,
        "`scprep.utils.get_cell_set` is deprecated. Use "
        "`scprep.select.get_cell_set` instead.",
        scprep.utils.get_cell_set,
        X,
        starts_with="A",
    )
    utils.assert_raises_message(
        RuntimeError,
        "`scprep.utils.subsample` is deprecated. Use "
        "`scprep.select.subsample` instead.",
        scprep.utils.subsample,
        X,
    )


def test_is_sparse_dataframe():
    X = data.load_10X(sparse=False)
    Y = X.astype(pd.SparseDtype(float, fill_value=0.0))
    assert scprep.utils.is_sparse_dataframe(Y)

    def test_fun(X):
        assert not scprep.utils.is_sparse_dataframe(X)

    types = (
        matrix._scipy_matrix_types
        + matrix._numpy_matrix_types
        + matrix._pandas_dense_matrix_types
    )
    if matrix._pandas_0:
        types.append(matrix.SparseDataFrame_deprecated)
    matrix.test_matrix_types(
        X,
        test_fun,
        types,
    )


def test_SparseDataFrame():
    X = data.load_10X(sparse=False)
    Y = X.astype(pd.SparseDtype(float, fill_value=0.0))
    index = X.index
    columns = X.columns

    def test_fun(X):
        X = scprep.utils.SparseDataFrame(X, index=index, columns=columns)
        utils.assert_matrix_class_equivalent(X, Y)

    matrix.test_all_matrix_types(X, test_fun)
    matrix.test_pandas_matrix_types(
        X,
        utils.assert_transform_equivalent,
        Y=Y,
        transform=scprep.utils.SparseDataFrame,
    )


def test_is_sparse_series():
    X = data.load_10X(sparse=True)
    assert scprep.utils.is_sparse_series(X[X.columns[0]])

    def test_fun(X):
        if scprep.utils.is_SparseDataFrame(X):
            x = X[X.columns[0]]
        else:
            x = scprep.select.select_cols(X, idx=0)
        assert not scprep.utils.is_sparse_series(x)

    types = (
        matrix._scipy_matrix_types
        + matrix._numpy_matrix_types
        + matrix._pandas_dense_matrix_types
    )
    if matrix._pandas_0:
        types.append(matrix.SparseDataFrame_deprecated)
    matrix.test_matrix_types(X.to_numpy(), test_fun, types)


def test_sort_clusters_by_values_accurate():
    clusters = [0, 0, 1, 1, 2, 2]
    values = [5, 5, 1, 1, 2, 2]
    new_clusters = scprep.utils.sort_clusters_by_values(clusters, values)
    test_array = scprep.utils.toarray([2, 2, 0, 0, 1, 1])
    np.testing.assert_array_equal(new_clusters, test_array)


def test_sort_clusters_by_values_wrong_len():
    clusters = [0, 0, 1, 1, 2, 2]
    values = [5, 5, 1, 1, 2]
    utils.assert_raises_message(
        ValueError,
        "Expected clusters ({}) and values ({}) to be the "
        "same length.".format(len(clusters), len(values)),
        scprep.utils.sort_clusters_by_values,
        clusters,
        values,
    )
