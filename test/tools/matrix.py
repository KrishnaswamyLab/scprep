import numpy as np
import pandas as pd
import warnings
from scipy import sparse
from functools import partial


def _no_warning_dia_matrix(*args, **kwargs):
    """Helper function to silently create diagonal matrix"""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=sparse.SparseEfficiencyWarning,
            message="Constructing a DIA matrix with [0-9]*"
            " diagonals is inefficient")
        return sparse.dia_matrix(*args, **kwargs)

SparseDataFrame = partial(pd.SparseDataFrame, default_fill_value=0.0)

_scipy_matrix_types = [
    sparse.csr_matrix,
    sparse.csc_matrix,
    sparse.bsr_matrix,
    sparse.lil_matrix,
    sparse.dok_matrix,
    _no_warning_dia_matrix,
]

_numpy_matrix_types = [
    np.array,
]

_pandas_dense_matrix_types = [
    pd.DataFrame,
]

_pandas_sparse_matrix_types = [
    SparseDataFrame,
]

_pandas_matrix_types = [
    pd.DataFrame,
    SparseDataFrame,
]

_indexable_matrix_types = [
    sparse.csr_matrix,
    sparse.csc_matrix,
    sparse.lil_matrix,
    sparse.dok_matrix,
    np.array,
    pd.DataFrame,
    SparseDataFrame
]


def test_matrix_types(X, test_fun, matrix_types, *args, **kwargs):
    """Test a function across a range of matrix types

    Parameters
    ----------
    X : matrix input
    test_fun : Function(X, *args, **kwargs) for testing
    matrix_types : List of functions (typically class constructors) converting X to desired matrix formats
    *args : positional arguments for test_fun
    **kwargs : keyword arguments for test_fun
    """
    for fun in matrix_types:
        Y = fun(X.copy())
        try:
            test_fun(Y, *args, **kwargs)
        except Exception as e:
            raise RuntimeError("{} with {} input to {}\n{}".format(
                type(e).__name__, type(Y).__name__, test_fun.__name__,
                str(e)))


def test_dense_matrix_types(X, test_fun, *args, **kwargs):
    """Test a function across all dense matrix types

    Parameters
    ----------
    X : matrix input
    test_fun : Function(X, *args, **kwargs) for testing
    *args : positional arguments for test_fun
    **kwargs : keyword arguments for test_fun
    """
    test_matrix_types(
        X, test_fun, _numpy_matrix_types + _pandas_dense_matrix_types,
        *args, **kwargs)


def test_sparse_matrix_types(X, test_fun, *args, **kwargs):
    """Test a function across all sparse matrix types

    Parameters
    ----------
    X : matrix input
    test_fun : Function(X, *args, **kwargs) for testing
    *args : positional arguments for test_fun
    **kwargs : keyword arguments for test_fun
    """
    test_matrix_types(
        X, test_fun, _scipy_matrix_types + _pandas_sparse_matrix_types,
        *args, **kwargs)


def test_all_matrix_types(X, test_fun, *args, **kwargs):
    """Test a function across all matrix types

    Parameters
    ----------
    X : matrix input
    test_fun : Function(X, *args, **kwargs) for testing
    *args : positional arguments for test_fun
    **kwargs : keyword arguments for test_fun
    """
    test_dense_matrix_types(X, test_fun, *args, **kwargs)
    test_sparse_matrix_types(X, test_fun, *args, **kwargs)


def test_pandas_matrix_types(X, test_fun, *args, **kwargs):
    """Test a function across all dense matrix types

    Parameters
    ----------
    X : matrix input
    test_fun : Function(X, *args, **kwargs) for testing
    *args : positional arguments for test_fun
    **kwargs : keyword arguments for test_fun
    """
    test_matrix_types(
        X, test_fun, _pandas_matrix_types,
        *args, **kwargs)


def test_numpy_matrix(X, test_fun, *args, **kwargs):
    """Test a function for np.matrix

    Parameters
    ----------
    X : matrix input
    test_fun : Function(X, *args, **kwargs) for testing
    *args : positional arguments for test_fun
    **kwargs : keyword arguments for test_fun
    """
    test_matrix_types(
        X, test_fun, [np.matrix],
        *args, **kwargs)
