import numpy as np
import pandas as pd
import warnings
from scipy import sparse
from functools import partial


_scipy_matrix_types = [
    sparse.csr_matrix,
    sparse.csc_matrix,
    sparse.bsr_matrix,
    sparse.lil_matrix,
    sparse.dok_matrix,
    sparse.dia_matrix,
]

_numpy_matrix_types = [
    np.array,
]

_pandas_dense_matrix_types = [
    pd.DataFrame,
]

_pandas_sparse_matrix_types = [
    partial(pd.SparseDataFrame, default_fill_value=0.0),
]

_indexable_matrix_types = [
    sparse.csr_matrix,
    sparse.csc_matrix,
    sparse.lil_matrix,
    sparse.dok_matrix,
    np.array,
    pd.DataFrame,
    pd.SparseDataFrame
]

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
            print("{} with {} input to {}".format(
                type(e).__name__, type(Y).__name__, test_fun.__name__))
            raise e


def check_dense_matrix_types(X, test_fun, *args, **kwargs):
    check_matrix_types(
        X, test_fun, *args,
        matrix_funs=_numpy_matrix_types + _pandas_dense_matrix_types,
        **kwargs)


def check_sparse_matrix_types(X, test_fun, *args, **kwargs):
    check_matrix_types(
        X, test_fun, *args,
        matrix_funs=_scipy_matrix_types + _pandas_sparse_matrix_types,
        **kwargs)


def check_all_matrix_types(X, test_fun, *args, **kwargs):
    check_dense_matrix_types(X, test_fun, *args, **kwargs)
    check_sparse_matrix_types(X, test_fun, *args, **kwargs)
