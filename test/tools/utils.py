import numpy as np
from scipy import sparse
import pandas as pd
from nose.tools import assert_raises
from scprep.utils import toarray, is_SparseDataFrame
from . import matrix
from nose.tools import assert_raises_regex, assert_warns_regex
import re


def assert_warns_message(expected_warning, expected_message, *args, **kwargs):
    expected_regex = re.escape(expected_message)
    return assert_warns_regex(expected_warning, expected_regex, *args, **kwargs)


def assert_raises_message(expected_warning, expected_message, *args, **kwargs):
    expected_regex = re.escape(expected_message)
    return assert_raises_regex(expected_warning, expected_regex, *args, **kwargs)


def assert_all_equal(X, Y):
    """Assert all values of two matrices are the same"""
    X = toarray(X)
    Y = toarray(Y)
    np.testing.assert_array_equal(X, Y)


def assert_all_close(X, Y, rtol=1e-05, atol=1e-08):
    """Assert all values of two matrices are similar

    Parameters
    ----------
    rtol : relative (multiplicative) tolerance of error
    atol : absolute (additive) tolerance of error
    """
    X = toarray(X)
    Y = toarray(Y)
    np.testing.assert_allclose(X, Y, rtol=rtol, atol=atol)


def assert_transform_equals(X, Y, transform, check=assert_all_equal, **kwargs):
    """Check that transform(X, **kwargs) == Y

    Parameters
    ----------
    X : input
    Y : output
    transform : function to apply to X
    check : function which asserts equivalence of X and Y
    **kwargs : additional arguments for transform

    Returns
    -------
    Y2 : returned value of transform(X, **kwargs)
    """
    Y2 = transform(X, **kwargs)
    check(Y, Y2), "{} failed on {}".format(transform, matrix._typename(X))
    return Y2


def assert_transform_unchanged(X, transform, check=assert_all_equal, **kwargs):
    """Check that transform(X, **kwargs) == X

    Parameters
    ----------
    X : input
    transform : function to apply to X
    check : function which asserts equivalence of X and Y
    **kwargs : additional arguments for transform

    Returns
    -------
    Y2 : returned value of transform(X, **kwargs)
    """
    assert_transform_equals(X, X, transform, check=check, **kwargs)


def assert_transform_equivalent(X, Y, transform, check=assert_all_equal, **kwargs):
    """Check the output of transform(X, **kwargs) == Y and transform(X, **kwargs) gives the same kind of matrix as X

    Parameters
    ----------
    X : input
    Y : output
    transform : function to apply to X
    check : function which asserts equivalence of X and Y

    **kwargs : additional arguments for transform

    Returns
    -------
    Y2 : returned value of transform(X, **kwargs)
    """
    Y2 = assert_transform_equals(X, Y, transform, check=check, **kwargs)
    assert assert_matrix_class_equivalent(
        X, Y2
    ), "{} produced inconsistent matrix output".format(_typename(X))


def assert_transform_raises(X, transform, exception=ValueError, **kwargs):
    """Check that transform(X) raises exception

    Parameters
    ----------
    X : input
    transform : function to apply to X
    exception : expected exception class
    """
    assert_raises(exception, transform, X, **kwargs)


def _is_sparse_dataframe(X):
    return is_SparseDataFrame(X) or (
        isinstance(X, pd.DataFrame) and hasattr(X, "sparse")
    )


def _sparse_dataframe_density(X):
    try:
        return X.sparse.density
    except AttributeError:
        return X.density


def assert_matrix_class_equivalent(X, Y):
    """Check the format of X and Y are the same

    We expect:
        * shape hasn't changed
        * sparsity hasn't changed
        * class hasn't changed (except perhaps which kind of spmatrix)
        * column and index names haven't changed
    """
    assert X.shape == Y.shape
    if sparse.issparse(X):
        assert sparse.issparse(Y)
        assert X.tocoo().nnz == Y.tocoo().nnz
    elif is_SparseDataFrame(X):
        assert _is_sparse_dataframe(Y)
    else:
        assert type(X) == type(Y), (type(X), type(Y))
    if _is_sparse_dataframe(X):
        assert _sparse_dataframe_density(X) == _sparse_dataframe_density(Y)
        assert _sparse_dataframe_density(X) == _sparse_dataframe_density(Y)
    if isinstance(X, pd.DataFrame):
        assert np.all(X.columns == Y.columns)
        assert np.all(X.index == Y.index)
    return True
