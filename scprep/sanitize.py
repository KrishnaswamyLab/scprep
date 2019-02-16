# author: Scott Gigante <scott.gigante@yale.edu>
# (C) 2018 Krishnaswamy Lab GPLv2

import pandas as pd


def check_numeric(data, dtype='float', copy=None):
    """Check a matrix contains only numeric data

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    dtype : str or `np.dtype`, optional (default: 'float')
        Data type to which to coerce the data
    copy : bool or None, optional (default: None)
        Copy the data before coercion. If None, default to
        False for all datatypes except pandas.SparseDataFrame

    Returns
    -------
    data : array-like, shape=[n_samples, n_features]
        Output data as numeric type

    Raises
    ------
    TypeError : if `data` cannot be coerced to `dtype`
    """
    if copy is None:
        copy = isinstance(data, pd.SparseDataFrame)
    try:
        return data.astype(dtype, copy=copy)
    except TypeError as e:
        if isinstance(data, pd.SparseDataFrame):
            if not copy:
                raise TypeError("pd.SparseDataFrame does not support "
                                "copy=False. Please use copy=True.")
            else:
                return data.astype(dtype)
        else:
            raise e
