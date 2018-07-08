import numpy as np
import pandas as pd
import numbers
from scipy import sparse
import warnings


def matrix_any(condition):
    """Check if a condition is true anywhere in a data matrix

    np.any doesn't handle matrices of type pd.DataFrame

    Parameters
    ----------
    condition : array-like
        Boolean matrix

    Returns
    -------
    any : bool
        True if condition contains any True values, False otherwise
    """
    return np.sum(np.sum(condition)) > 0


def select_cols(data, idx):
    """Select columns from a data matrix

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    idx : list-like, shape=[m_features]
        Integer indices or string column names to be selected

    Returns
    -------
    data : array-like, shape=[n_samples, m_features]
        Subsetted output data

    Raises
    ------
    UserWarning : if no columns are selected
    """
    if isinstance(data, pd.DataFrame):
        try:
            data = data.loc[:, idx]
        except KeyError:
            if isinstance(idx, numbers.Integral) or \
                    issubclass(np.array(idx).dtype.type, numbers.Integral):
                data = data.loc[:, np.array(data.columns)[idx]]
            else:
                raise
    else:
        if isinstance(data, (sparse.coo_matrix,
                             sparse.bsr_matrix,
                             sparse.lil_matrix,
                             sparse.dia_matrix)):
            data = data.tocsr()
        data = data[:, idx]
    if data.shape[1] == 0:
        warnings.warn("Selecting 0 columns.", UserWarning)
    return data


def select_rows(data, idx):
    """Select rows from a data matrix

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    idx : list-like, shape=[m_samples]
        Integer indices or string index names to be selected

    Returns
    -------
    data : array-like, shape=[m_samples, r_features]
        Subsetted output data

    Raises
    ------
    UserWarning : if no rows are selected
    """
    if isinstance(data, pd.DataFrame):
        data = data.loc[idx]
    else:
        if isinstance(data, (sparse.coo_matrix,
                             sparse.bsr_matrix,
                             sparse.dia_matrix)):
            data = data.tocsr()
        data = data[idx, :]
    if data.shape[0] == 0:
        warnings.warn("Selecting 0 rows.", UserWarning)
    return data
