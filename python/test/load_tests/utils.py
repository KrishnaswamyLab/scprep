import numpy as np
from scipy import sparse
import pandas as pd
from nose.tools import assert_raises


def to_array(X):
    if sparse.issparse(X):
        X = X.toarray()
    elif isinstance(X, pd.SparseDataFrame):
        X = X.to_dense().values
    elif isinstance(X, pd.DataFrame):
        X = X.values
    return X


def assert_all_equal(X, Y):
    X = to_array(X)
    Y = to_array(Y)
    np.testing.assert_array_equal(X, Y)


def assert_all_close(X, Y, rtol=1e-05, atol=1e-08):
    X = to_array(X)
    Y = to_array(Y)
    np.testing.assert_allclose(X, Y, rtol=rtol, atol=atol)


def check_output_equivalent(X, Y, transform, check=assert_all_equal, **kwargs):
    try:
        Y2 = transform(X, **kwargs)
    except Exception as e:
        print("transformation failed on {}".format(type(X).__name__))
        raise(e)
    check(Y, Y2), "{} failed on {}".format(
        transform,
        type(X).__name__)
    return Y2


def check_output_unchanged(X, transform, check=assert_all_equal, **kwargs):
    try:
        Y = transform(X, **kwargs)
    except Exception as e:
        print("transformation failed on {}".format(type(X).__name__))
        raise(e)
    check(Y, X), "{} failed on {}".format(
        transform,
        type(X).__name__)
    assert matrix_class_equivalent(X, Y), \
        "{} produced inconsistent matrix output".format(
            type(X).__name__)


def check_transform_equivalent(X, Y, transform, check=assert_all_equal,
                               matrix_form_unchanged=True,
                               **kwargs):
    Y2 = check_output_equivalent(X, Y, transform, check=check, **kwargs)
    if matrix_form_unchanged:
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
