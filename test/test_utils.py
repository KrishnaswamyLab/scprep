from tools import data, matrix, utils
import scprep
from sklearn.utils.testing import assert_raise_message, assert_warns_message
import numpy as np
import pandas as pd


def test_with_pkg():
    @scprep.utils._with_pkg(pkg="invalid")
    def invalid():
        pass
    assert_raise_message(ImportError,
                         "invalid not found. Please install it with e.g. "
                         "`pip install --user invalid`",
                         invalid)


def test_with_pkg_version_none():
    @scprep.utils._with_pkg(pkg="pandas")
    def test():
        return True
    assert test()


def test_with_pkg_version_exact():
    major, minor = [int(v) for v in pd.__version__.split(".")[:2]]

    @scprep.utils._with_pkg(pkg="pandas", min_version="{}.{}".format(major, minor))
    def test():
        return True
    assert test()


def test_with_pkg_version_exact_no_minor():
    major, minor = [int(v) for v in pd.__version__.split(".")[:2]]

    @scprep.utils._with_pkg(pkg="pandas", min_version=major)
    def test():
        return True
    assert test()


def test_with_pkg_version_pass_major():
    major, minor = [int(v) for v in pd.__version__.split(".")[:2]]

    @scprep.utils._with_pkg(pkg="pandas", min_version=major - 1)
    def test():
        return True
    assert test()


def test_with_pkg_version_pass_minor():
    major, minor = [int(v) for v in pd.__version__.split(".")[:2]]

    @scprep.utils._with_pkg(pkg="pandas", min_version="{}.{}".format(major, minor - 1))
    def test():
        return True
    assert test()


def test_with_pkg_version_fail_major():
    major, minor = [int(v) for v in pd.__version__.split(".")[:2]]

    @scprep.utils._with_pkg(pkg="pandas", min_version=major + 1)
    def test():
        return True
    assert_raise_message(ImportError,
                         "scprep requires pandas>={0} (installed: {1}). "
                         "Please upgrade it with e.g."
                         " `pip install --user --upgrade pandas".format(
                             major + 1, pd.__version__),
                         test)


def test_with_pkg_version_fail_minor():
    major, minor = [int(v) for v in pd.__version__.split(".")[:2]]

    @scprep.utils._with_pkg(pkg="pandas", min_version="{}.{}".format(major, minor + 1))
    def test():
        return True
    assert_raise_message(ImportError,
                         "scprep requires pandas>={0}.{1} (installed: {2}). "
                         "Please upgrade it with e.g."
                         " `pip install --user --upgrade pandas".format(
                             major, minor + 1, pd.__version__),
                         test)


def test_with_pkg_version_memoize():
    major, minor = [int(v) for v in pd.__version__.split(".")[:2]]
    min_version = "{}.{}".format(major, minor + 1)

    @scprep.utils._with_pkg(pkg="pandas", min_version=min_version)
    def test():
        return True
    true_version = pd.__version__
    pd.__version__ = min_version
    # should pass
    assert test()
    pd.__version__ = true_version
    # should fail, but already memoized
    assert test()


def test_try_import():
    assert scprep.utils._try_import("invalid") is None
    assert scprep.utils._try_import("numpy") is np


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
                         "Expected array-like. Got ",
                         scprep.utils.toarray,
                         "hello")


def test_toarray_list_of_strings():
    X = ['hello', 'world', [1, 2, 3]]
    X = scprep.utils.toarray(X)
    assert isinstance(X[2], np.ndarray)


def test_matrix_sum():
    X = data.generate_positive_sparse_matrix(shape=(50, 100))
    sums = np.array(X.sum(0)).flatten()
    matrix.test_all_matrix_types(X, utils.assert_transform_equals, Y=sums,
                                 transform=scprep.utils.matrix_sum, axis=0,
                                 check=utils.assert_all_close)
    matrix.test_numpy_matrix(X, utils.assert_transform_equals, Y=sums,
                             transform=scprep.utils.matrix_sum, axis=0,
                             check=utils.assert_all_close)

    sums = np.array(X.sum(1)).flatten()
    matrix.test_all_matrix_types(X, utils.assert_transform_equals, Y=sums,
                                 transform=scprep.utils.matrix_sum, axis=1,
                                 check=utils.assert_all_close)
    matrix.test_numpy_matrix(X, utils.assert_transform_equals, Y=sums,
                             transform=scprep.utils.matrix_sum, axis=1,
                             check=utils.assert_all_close)

    sums = np.array(X.sum(None)).flatten()
    matrix.test_all_matrix_types(X, utils.assert_transform_equals, Y=sums,
                                 transform=scprep.utils.matrix_sum, axis=None,
                                 check=utils.assert_all_close)
    matrix.test_numpy_matrix(X, utils.assert_transform_equals, Y=sums,
                             transform=scprep.utils.matrix_sum, axis=None,
                             check=utils.assert_all_close)

    assert_raise_message(ValueError,
                         "Expected axis in [0, 1, None]. Got 5",
                         scprep.utils.matrix_sum,
                         data,
                         5)


def test_matrix_elementwise_multiply_row():
    X = data.generate_positive_sparse_matrix(shape=(50, 100))
    x = X[:, 0] + 1
    Y = pd.DataFrame(X).mul(x, axis=0)
    matrix.test_all_matrix_types(
        X, utils.assert_transform_equivalent, Y=Y,
        transform=scprep.utils.matrix_vector_elementwise_multiply,
        check=utils.assert_all_close,
        axis=0, multiplier=x)


def test_matrix_elementwise_multiply_col():
    X = data.generate_positive_sparse_matrix(shape=(50, 100))
    x = X[0] + 1
    Y = pd.DataFrame(X).mul(x, axis=1)
    matrix.test_all_matrix_types(
        X, utils.assert_transform_equivalent, Y=Y,
        transform=scprep.utils.matrix_vector_elementwise_multiply,
        check=utils.assert_all_close,
        axis=1, multiplier=x)


def test_matrix_elementwise_multiply_guess_row():
    X = data.generate_positive_sparse_matrix(shape=(50, 100))
    x = X[:, 0] + 1
    Y = pd.DataFrame(X).mul(x, axis=0)
    matrix.test_all_matrix_types(
        X, utils.assert_transform_equivalent, Y=Y,
        transform=scprep.utils.matrix_vector_elementwise_multiply,
        check=utils.assert_all_close,
        axis=None, multiplier=x)


def test_matrix_elementwise_multiply_guess_col():
    X = data.generate_positive_sparse_matrix(shape=(50, 100))
    x = X[0] + 1
    Y = pd.DataFrame(X).mul(x, axis=1)
    matrix.test_all_matrix_types(
        X, utils.assert_transform_equivalent, Y=Y,
        transform=scprep.utils.matrix_vector_elementwise_multiply,
        check=utils.assert_all_close,
        axis=None, multiplier=x)


def test_matrix_elementwise_multiply_square_guess():
    X = data.generate_positive_sparse_matrix(shape=(50, 50))
    assert_raise_message(
        RuntimeError,
        "`data` is square, cannot guess axis from input. Please provide "
        "`axis=0` to multiply along rows or "
        "`axis=1` to multiply along columns.",
        scprep.utils.matrix_vector_elementwise_multiply,
        X, X[0])


def test_matrix_elementwise_multiply_row_wrong_size():
    X = data.generate_positive_sparse_matrix(shape=(50, 100))
    assert_raise_message(
        ValueError,
        "Expected `multiplier` to be a vector of length `data.shape[0]` (50)."
        " Got (100,)",
        scprep.utils.matrix_vector_elementwise_multiply,
        X, X[0], axis=0)


def test_matrix_elementwise_multiply_col_wrong_size():
    X = data.generate_positive_sparse_matrix(shape=(50, 100))
    assert_raise_message(
        ValueError,
        "Expected `multiplier` to be a vector of length `data.shape[1]` (100)."
        " Got (50,)",
        scprep.utils.matrix_vector_elementwise_multiply,
        X, X[:, 0], axis=1)


def test_matrix_elementwise_multiply_guess_wrong_size():
    X = data.generate_positive_sparse_matrix(shape=(50, 100))
    assert_raise_message(
        ValueError,
        "Expected `multiplier` to be a vector of length `data.shape[0]` (50) "
        "or `data.shape[1]` (100). Got (10,)",
        scprep.utils.matrix_vector_elementwise_multiply,
        X, X[0, :10])


def test_matrix_elementwise_multiply_invalid_axis():
    X = data.generate_positive_sparse_matrix(shape=(50, 100))
    assert_raise_message(
        ValueError,
        "Expected axis in [0, 1, None]. Got 5",
        scprep.utils.matrix_vector_elementwise_multiply,
        X, X[0], axis=5)


def test_deprecated():
    X = data.load_10X()
    assert_warns_message(FutureWarning,
                         "`scprep.utils.select_cols` is deprecated. Use "
                         "`scprep.select.select_cols` instead.",
                         scprep.utils.select_cols,
                         X,
                         [1, 2, 3])
    assert_warns_message(FutureWarning,
                         "`scprep.utils.select_rows` is deprecated. Use "
                         "`scprep.select.select_rows` instead.",
                         scprep.utils.select_rows,
                         X,
                         [1, 2, 3])
    assert_warns_message(FutureWarning,
                         "`scprep.utils.get_gene_set` is deprecated. Use "
                         "`scprep.select.get_gene_set` instead.",
                         scprep.utils.get_gene_set,
                         X,
                         starts_with="D")
    assert_warns_message(FutureWarning,
                         "`scprep.utils.get_cell_set` is deprecated. Use "
                         "`scprep.select.get_cell_set` instead.",
                         scprep.utils.get_cell_set,
                         X,
                         starts_with="A")
    assert_warns_message(FutureWarning,
                         "`scprep.utils.subsample` is deprecated. Use "
                         "`scprep.select.subsample` instead.",
                         scprep.utils.subsample,
                         X,
                         n=10)
