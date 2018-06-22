import numpy as np
from scipy import sparse
import pandas as pd
from functools import partial
from nose.tools import assert_raises
import warnings


def to_array(X):
    if sparse.issparse(X):
        X = X.toarray()
    elif isinstance(X, pd.SparseDataFrame):
        X = X.to_dense().values
    elif isinstance(X, pd.DataFrame):
        X = X.values
    return X


def all_equal(X, Y):
    X = to_array(X)
    Y = to_array(Y)
    return np.all(X == Y)


def all_close(X, Y):
    X = to_array(X)
    Y = to_array(Y)
    return np.allclose(X, Y)


def check_matrix_types(X, test_fun, matrix_funs, *args, **kwargs):
    for fun in matrix_funs:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=sparse.SparseEfficiencyWarning,
                message="Constructing a DIA matrix with [0-9]*"
                " diagonals is inefficient")
            Y = fun(X.copy())
        try:
            test_fun(Y, *args, **kwargs)
        except Exception as e:
            raise type(e)(str(type(Y)))


def check_dense_matrix_types(X, test_fun, *args, **kwargs):
    check_matrix_types(X, test_fun, *args, matrix_funs=[
        np.array,
        pd.DataFrame],
        **kwargs)


def check_sparse_matrix_types(X, test_fun, *args, **kwargs):
    check_matrix_types(X, test_fun, *args, matrix_funs=[
        sparse.csr_matrix,
        sparse.csc_matrix,
        sparse.bsr_matrix,
        sparse.csc_matrix,
        sparse.lil_matrix,
        sparse.dok_matrix,
        sparse.dia_matrix,
        partial(pd.SparseDataFrame, default_fill_value=0.0)],
        **kwargs)


def check_all_matrix_types(X, test_fun, *args, **kwargs):
    check_dense_matrix_types(X, test_fun, *args, **kwargs)
    check_sparse_matrix_types(X, test_fun, *args, **kwargs)


def check_output_equivalent(X, Y, transform, check=all_equal):
    try:
        Y2 = transform(X)
    except Exception as e:
        print("transformation failed on {}".format(type(X).__name__))
        raise(e)
    assert check(Y, Y2), "{} failed on {}".format(
        transform,
        type(X).__name__)
    return Y2


def check_transform_equivalent(X, Y, transform, check=all_equal):
    Y2 = check_output_equivalent(X, Y, transform, check=check)
    assert matrix_class_equivalent(X, Y2), \
        "{} produced inconsistent matrix output".format(
        type(X).__name__)


def check_transform_raises(X, transform, exception=ValueError):
    assert_raises(exception, transform, X)


def matrix_class_equivalent(X, Y):
    assert X.shape == Y.shape
    if sparse.issparse(X):
        assert sparse.issparse(Y)
    if isinstance(X, pd.DataFrame):
        assert np.all(X.columns == Y.columns)
        assert np.all(X.index == Y.index)
    if isinstance(X, pd.SparseDataFrame):
        assert X.density == Y.density
    return True


def generate_positive_sparse_matrix(shape=[500, 500], seed=42):
    np.random.seed(seed)
    X = np.random.normal(0, 1, shape) * \
        np.random.poisson(0.1, shape)
    X = np.abs(X)
    return X
