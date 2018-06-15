import numpy as np
from scipy import sparse
import pandas as pd
from functools import partial
from preprocessing import transformation


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


def any(condition):
    """np.any doesn't handle data frames
    """
    return np.sum(np.any(condition)) > 0


def test_transform(transform, lambda_transform, seed=42, check=all_equal):
    np.random.seed(seed)
    X = np.random.normal(0, 1, [1000, 1000]) * \
        np.random.poisson(0.1, [1000, 1000])
    matrix_funs = [
        np.array,
        sparse.csr_matrix,
        sparse.csc_matrix,
        sparse.bsr_matrix,
        sparse.csc_matrix,
        sparse.lil_matrix,
        sparse.dok_matrix,
        sparse.dia_matrix,
        pd.DataFrame,
        partial(pd.SparseDataFrame, default_fill_value=0.0)]
    X = np.abs(X)
    transform_X = lambda_transform(X)
    for fun in matrix_funs:
        Y = transform(fun(X.copy()))
        assert check(Y, transform_X), "{} failed on {}".format(
            transform.__name__,
            fun.__name__)
        if sparse.issparse(X):
            assert sparse.issparse(Y)
        if isinstance(X, pd.DataFrame):
            assert X.columns == Y.columns
            assert X.index == Y.index
        if isinstance(X, pd.SparseDataFrame):
            assert X.density == Y.density


test_transform(transformation.sqrt_transform, lambda X: np.sqrt(X))
test_transform(transformation.log_transform, lambda X: np.log(X + 1))
test_transform(transformation.arcsinh_transform,
               lambda X: np.arcsinh(X / 5), check=all_close)
