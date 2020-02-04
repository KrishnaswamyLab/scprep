import numpy as np
import pandas as pd
import warnings
from scipy import sparse
from functools import partial
from scprep.utils import is_SparseDataFrame
from packaging import version


def _ignore_pandas_sparse_warning():
    warnings.filterwarnings("ignore", category=FutureWarning, message="SparseSeries")
    warnings.filterwarnings("ignore", category=FutureWarning, message="SparseDataFrame")
    warnings.filterwarnings("error", category=pd.errors.PerformanceWarning)


def _reset_warnings():
    warnings.filterwarnings("error", category=FutureWarning, message="SparseSeries")
    warnings.filterwarnings("error", category=FutureWarning, message="SparseDataFrame")
    warnings.filterwarnings("error", category=pd.errors.PerformanceWarning)


_reset_warnings()


def _no_warning_dia_matrix(*args, **kwargs):
    """Helper function to silently create diagonal matrix"""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=sparse.SparseEfficiencyWarning,
            message="Constructing a DIA matrix with [0-9]*" " diagonals is inefficient",
        )
        return sparse.dia_matrix(*args, **kwargs)


def SparseDataFrame_deprecated(X, default_fill_value=0.0):
    return pd.SparseDataFrame(X, default_fill_value=default_fill_value)


def SparseSeries(X, default_fill_value=0.0):
    return pd.Series(X).astype(pd.SparseDtype(float, fill_value=default_fill_value))


def SparseSeries_deprecated(X, default_fill_value=0.0):
    return pd.SparseSeries(X, fill_value=default_fill_value)


def SparseDataFrame(X, default_fill_value=0.0):
    if sparse.issparse(X):
        X = pd.DataFrame.sparse.from_spmatrix(X)
        X.sparse.fill_value = default_fill_value
    elif is_SparseDataFrame(X) or not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    return X.astype(pd.SparseDtype(float, fill_value=default_fill_value))


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

_pandas_vector_types = [pd.Series, SparseSeries]

_pandas_0 = version.parse(pd.__version__) < version.parse("1.0.0")

if _pandas_0:
    _pandas_sparse_matrix_types.append(SparseDataFrame_deprecated)
    _pandas_vector_types.append(SparseSeries_deprecated)

_pandas_matrix_types = _pandas_dense_matrix_types + _pandas_sparse_matrix_types

_scipy_indexable_matrix_types = [
    sparse.csr_matrix,
    sparse.csc_matrix,
    sparse.lil_matrix,
    sparse.dok_matrix,
]

_indexable_matrix_types = (
    _scipy_indexable_matrix_types + _numpy_matrix_types + _pandas_matrix_types
)


def _typename(X):
    if (
        isinstance(X, pd.DataFrame)
        and not is_SparseDataFrame(X)
        and hasattr(X, "sparse")
    ):
        return "DataFrame[SparseArray]"
    else:
        return type(X).__name__


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
        if fun is SparseDataFrame_deprecated or fun is SparseSeries_deprecated:
            _ignore_pandas_sparse_warning()
        Y = fun(X.copy())
        try:
            test_fun(Y, *args, **kwargs)
        except Exception as e:
            raise RuntimeError(
                "{} with {} input to {}\n{}".format(
                    type(e).__name__, _typename(Y), test_fun.__name__, str(e)
                )
            )
        finally:
            _reset_warnings()


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
        X, test_fun, _numpy_matrix_types + _pandas_dense_matrix_types, *args, **kwargs
    )


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
        X, test_fun, _scipy_matrix_types + _pandas_sparse_matrix_types, *args, **kwargs
    )


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
    test_matrix_types(X, test_fun, _pandas_matrix_types, *args, **kwargs)


def test_numpy_matrix(X, test_fun, *args, **kwargs):
    """Test a function for np.matrix

    Parameters
    ----------
    X : matrix input
    test_fun : Function(X, *args, **kwargs) for testing
    *args : positional arguments for test_fun
    **kwargs : keyword arguments for test_fun
    """
    test_matrix_types(X, test_fun, [np.matrix], *args, **kwargs)
