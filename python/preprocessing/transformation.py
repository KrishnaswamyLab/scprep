# author: Scott Gigante <scott.gigante@yale.edu>
# (C) 2018 Krishnaswamy Lab GPLv2

import numpy as np
from scipy import sparse
import pandas as pd
import warnings


def _transform(X, fun):
    if sparse.issparse(X):
        if isinstance(X, (sparse.lil_matrix, sparse.dok_matrix)):
            X = X.tocsr()
        X.data = fun(X.data)
    else:
        X = fun(X)
    return X


def sqrt_transform(X):
    if any(X < 0):
        raise ValueError("Cannot square root transform negative values")
    return _transform(X, np.sqrt)


def log_transform(X, pseudocount=1):
    if any(X < 0):
        raise ValueError("Cannot log transform negative values")
    if pseudocount != 1 and (sparse.issparse(X) or
                             isinstance(X, pd.SparseDataFrame)):
        warnings.warn("log transform on sparse data requires pseudocount=1",
                      RuntimeWarning)
        pseudocount = 1
    return _transform(X, lambda X: np.log(X + 1))


def arcsinh_transform(X, cofactor=5):
    if cofactor <= 0:
        raise ValueError("Expected cofactor > 0 or None. "
                         "Got {}".format(cofactor))
    if cofactor is not None:
        X = X / cofactor
    return _transform(X, np.arcsinh)
