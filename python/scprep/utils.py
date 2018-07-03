import numpy as np
import pandas as pd
import numbers
from scipy import sparse
import warnings


def matrix_any(condition):
    """np.any doesn't handle data frames
    """
    return np.sum(np.sum(condition)) > 0


def select_cols(data, idx):
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
