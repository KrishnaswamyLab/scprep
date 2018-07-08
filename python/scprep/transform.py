# author: Scott Gigante <scott.gigante@yale.edu>
# (C) 2018 Krishnaswamy Lab GPLv2

import numpy as np
from scipy import sparse
import pandas as pd
import warnings
from .utils import matrix_any


def _transform(data, fun):
    """Perform a numerical transformation to data

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    fun : callable
        Numerical transformation function, `np.ufunc` or similar.

    Returns
    -------
    data : array-like, shape=[n_samples, n_features]
        Transformed output data
    """
    if sparse.issparse(data):
        if isinstance(data, (sparse.lil_matrix, sparse.dok_matrix)):
            data = data.tocsr()
        else:
            # avoid modifying in place
            data = data.copy()
        data.data = fun(data.data)
    else:
        data = fun(data)
    return data


def sqrt(data):
    """Square root transform

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data

    Returns
    -------
    data : array-like, shape=[n_samples, n_features]
        Square root transformed output data

    Raises
    ------
    ValueError : if data has negative values
    """
    if matrix_any(data < 0):
        raise ValueError("Cannot square root transform negative values")
    return _transform(data, np.sqrt)


def log(data, pseudocount=1, base=10):
    """Log transform

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    pseudocount : int, optional (default: 1)
        Pseudocount to add to values before log transform.
        If data is sparse, pseudocount must be 1 such that
        log(0 + pseudocount) = 0
    base : {2, 'e', 10}, optional (default: 10)
        Logarithm base.

    Returns
    -------
    data : array-like, shape=[n_samples, n_features]
        Log transformed output data

    Raises
    ------
    ValueError : if data has zero or negative values
    RuntimeWarning : if data is sparse and pseudocount != 1
    """
    if matrix_any(data < 0):
        raise ValueError("Cannot log transform negative values")
    if pseudocount != 1 and (sparse.issparse(data) or
                             isinstance(data, pd.SparseDataFrame)):
        warnings.warn("log transform on sparse data requires pseudocount=1",
                      RuntimeWarning)
        pseudocount = 1
    if base == 2:
        log = np.log2
    elif base == 'e':
        log = np.log
    elif base == 10:
        log = np.log10
    else:
        raise ValueError("Expected base in [2, 'e', 10]. Got {}".format(base))
    return _transform(data, lambda data: log(data + pseudocount))


def arcsinh(data, cofactor=5):
    """Inverse hyperbolic sine transform

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    cofactor : float or None, optional (default: 5)
        Factor by which to divide data before arcsinh transform

    Returns
    -------
    data : array-like, shape=[n_samples, n_features]
        Inverse hyperbolic sine transformed output data

    Raises
    ------
    ValueError : if cofactor <= 0
    """
    if cofactor <= 0:
        raise ValueError("Expected cofactor > 0 or None. "
                         "Got {}".format(cofactor))
    if cofactor is not None:
        data = data / cofactor
    return _transform(data, np.arcsinh)


def sqrt_transform(*args, **kwargs):
    warnings.warn("scprep.transform.sqrt_transform is deprecated. Please use "
                  "scprep.transform.sqrt in future.", FutureWarning)


def log_transform(*args, **kwargs):
    warnings.warn("scprep.transform.log_transform is deprecated. Please use "
                  "scprep.transform.log in future.", FutureWarning)


def arcsinh_transform(*args, **kwargs):
    warnings.warn("scprep.transform.arcsinh_transform is deprecated. Please "
                  "use scprep.transform.arcsinh in future.", FutureWarning)
