import numpy as np
import pandas as pd
import numbers
from scipy import sparse
import warnings
import re


def _is_1d(data):
    try:
        return len(data.shape) == 1
    except AttributeError:
        return True


def _get_columns(data):
    return data.columns if isinstance(data, pd.DataFrame) else data.index


def _get_column_length(data):
    try:
        return data.shape[1]
    except (IndexError, AttributeError):
        return len(data)


def _get_row_length(data):
    try:
        return data.shape[0]
    except (IndexError, AttributeError):
        return len(data)


def _check_columns_compatible(*data):
    for d in data:
        if not _get_column_length(d) == _get_column_length(data[0]):
            raise ValueError(
                "Expected all data to have the same number of "
                "columns. Got {}".format(
                    [_get_column_length(d) for d in data]))
        if isinstance(d, (pd.DataFrame, pd.Series)) and \
                isinstance(data[0], (pd.DataFrame, pd.Series)):
            if not np.all(_get_columns(data[0]) == _get_columns(d)):
                raise ValueError(
                    "Expected all pandas inputs to have the same columns. "
                    "Fix with "
                    "`scprep.select.select_cols(extra_data, data.columns)`")


def _check_rows_compatible(*data):
    for d in data:
        if not _get_row_length(d) == _get_row_length(data[0]):
            raise ValueError(
                "Expected all data to have the same number of "
                "rows. Got {}".format(
                    [d.shape[0] for d in data]))
        if isinstance(d, (pd.DataFrame, pd.Series)) and \
                isinstance(data[0], (pd.DataFrame, pd.Series)):
            if not np.all(data[0].index == d.index):
                raise ValueError(
                    "Expected all pandas inputs to have the same index. "
                    "Fix with "
                    "`scprep.select.select_rows(extra_data, data.index)`")


def _convert_dataframe_1d(idx):
    if (not _is_1d(idx)) and np.prod(idx.shape) != np.max(idx.shape):
        raise ValueError(
            "Expected idx to be 1D. Got shape {}".format(idx.shape))
    idx = idx.iloc[:, 0] if idx.shape[1] == 1 else idx.iloc[0, :]
    return idx


def _get_string_subset(data, starts_with=None, ends_with=None, regex=None):
    """Get a subset from a string array

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features] or [n_features]
        Input pd.DataFrame, or list of names
    starts_with : str or None, optional (default: None)
        If not None, only return names that start with this prefix
    ends_with : str or None, optional (default: None)
        If not None, only return names that end with this suffix
    regex : str or None, optional (default: None)
        If not None, only return names that match this regular expression

    Returns
    -------
    data : list-like, shape<=[n_features]
        List of matching strings
    """
    mask = np.full_like(data, True, dtype=bool)
    if starts_with is not None:
        start_match = np.vectorize(lambda x: x.startswith(starts_with))
        mask = np.logical_and(mask, start_match(data))
    if ends_with is not None:
        end_match = np.vectorize(lambda x: x.endswith(ends_with))
        mask = np.logical_and(mask, end_match(data))
    if regex is not None:
        regex = re.compile(regex)
        regex_match = np.vectorize(lambda x: bool(regex.search(x)))
        mask = np.logical_and(mask, regex_match(data))
    return data[mask]


def get_gene_set(data, starts_with=None, ends_with=None, regex=None):
    """Get a list of genes from data

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features] or [n_features]
        Input pd.DataFrame, or list of gene names
    starts_with : str or None, optional (default: None)
        If not None, only return gene names that start with this prefix
    ends_with : str or None, optional (default: None)
        If not None, only return gene names that end with this suffix
    regex : str or None, optional (default: None)
        If not None, only return gene names that match this regular expression

    Returns
    -------
    genes : list-like, shape<=[n_features]
        List of matching genes
    """
    if not _is_1d(data):
        try:
            data = data.columns.values
        except AttributeError:
            raise TypeError("data must be a list of gene names or a pandas "
                            "DataFrame. Got {}".format(type(data).__name__))
    if starts_with is None and ends_with is None and regex is None:
        warnings.warn("No selection conditions provided. "
                      "Returning all genes.", UserWarning)
    return _get_string_subset(data, starts_with=starts_with,
                              ends_with=ends_with, regex=regex)


def get_cell_set(data, starts_with=None, ends_with=None, regex=None):
    """Get a list of cells from data

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features] or [n_samples]
        Input pd.DataFrame, or list of cell names
    starts_with : str or None, optional (default: None)
        If not None, only return cell names that start with this prefix
    ends_with : str or None, optional (default: None)
        If not None, only return cell names that end with this suffix
    regex : str or None, optional (default: None)
        If not None, only return cell names that match this regular expression

    Returns
    -------
    cells : list-like, shape<=[n_features]
        List of matching cells
    """
    if not _is_1d(data):
        try:
            data = data.index.values
        except AttributeError:
            raise TypeError("data must be a list of cell names or a pandas "
                            "DataFrame. Got {}".format(type(data).__name__))
    if starts_with is None and ends_with is None and regex is None:
        warnings.warn("No selection conditions provided. "
                      "Returning all cells.", UserWarning)
    return _get_string_subset(data, starts_with=starts_with,
                              ends_with=ends_with, regex=regex)


def select_cols(data, *extra_data, idx=None,
                starts_with=None, ends_with=None, regex=None):
    """Select columns from a data matrix

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    extra_data : array-like, shape=[*, n_features], optional
        Optional additional data objects from which to select the same rows
    idx : list-like, shape=[m_features]
        Integer indices or string column names to be selected
    starts_with : str or None, optional (default: None)
        If not None, select columns that start with this prefix
    ends_with : str or None, optional (default: None)
        If not None, select columns that end with this suffix
    regex : str or None, optional (default: None)
        If not None, select columns that match this regular expression

    Returns
    -------
    data : array-like, shape=[n_samples, m_features]
        Subsetted output data.
    extra_data : array-like, shape=[*, m_features]
        Subsetted extra data, if passed.

    Examples
    --------
    data_subset = scprep.select.select_cols(data, idx=np.random.choice([True, False], data.shape[1]))
    data_subset, metadata_subset = scprep.select.select_cols(data, metadata, starts_with="MT")

    Raises
    ------
    UserWarning : if no columns are selected
    """
    if len(extra_data) > 0:
        _check_columns_compatible(data, *extra_data)
    if idx is None and starts_with is None and ends_with is None and regex is None:
        warnings.warn("No selection conditions provided. "
                      "Returning all columns.", UserWarning)
        return tuple([data] + list(extra_data)) if len(extra_data) > 0 else data
    if idx is None:
        if not isinstance(data, pd.DataFrame):
            raise ValueError(
                "Can only select based on column names with DataFrame input. "
                "Please set `idx` to select specific columns.")
        idx = get_gene_set(data, starts_with=starts_with,
                           ends_with=ends_with, regex=regex)

    if isinstance(idx, pd.DataFrame):
        idx = _convert_dataframe_1d(idx)

    if isinstance(data, pd.DataFrame):
        try:
            data = data.loc[:, idx]
        except (KeyError, TypeError):
            if isinstance(idx, numbers.Integral) or \
                    issubclass(np.array(idx).dtype.type, numbers.Integral):
                data = data.loc[:, np.array(data.columns)[idx]]
            else:
                raise
    elif isinstance(data, pd.Series):
        try:
            data = data.loc[idx]
        except (KeyError, TypeError):
            if isinstance(idx, numbers.Integral) or \
                    issubclass(np.array(idx).dtype.type, numbers.Integral):
                data = data.loc[np.array(data.index)[idx]]
            else:
                raise
    elif _is_1d(data):
        if isinstance(data, list):
            # can't numpy index a list
            data = np.array(data)
        data = data[idx]
    else:
        if isinstance(data, (sparse.coo_matrix,
                             sparse.bsr_matrix,
                             sparse.lil_matrix,
                             sparse.dia_matrix)):
            data = data.tocsr()
        data = data[:, idx]
    if _get_column_length(data) == 0:
        warnings.warn("Selecting 0 columns.", UserWarning)
    if len(extra_data) > 0:
        data = [data]
        for d in extra_data:
            data.append(select_cols(d, idx=idx))
        data = tuple(data)
    return data


def select_rows(data, *extra_data, idx=None,
                starts_with=None, ends_with=None, regex=None):
    """Select rows from a data matrix

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    extra_data : array-like, shape=[n_samples, *], optional
        Optional additional data objects from which to select the same rows
    idx : list-like, shape=[m_samples], optional (default: None)
        Integer indices or string index names to be selected
    starts_with : str or None, optional (default: None)
        If not None, select rows that start with this prefix
    ends_with : str or None, optional (default: None)
        If not None, select rows that end with this suffix
    regex : str or None, optional (default: None)
        If not None, select rows that match this regular expression

    Returns
    -------
    data : array-like, shape=[m_samples, n_features]
        Subsetted output data
    extra_data : array-like, shape=[m_samples, *]
        Subsetted extra data, if passed.

    Examples
    --------
    data_subset = scprep.select.select_rows(data, idx=np.random.choice([True, False], data.shape[0]))
    data_subset, labels_subset = scprep.select.select_rows(data, labels, end_with="batch1")

    Raises
    ------
    UserWarning : if no rows are selected
    """
    if len(extra_data) > 0:
        _check_rows_compatible(data, *extra_data)
    if idx is None and starts_with is None and ends_with is None and regex is None:
        warnings.warn("No selection conditions provided. "
                      "Returning all rows.", UserWarning)
        return tuple([data] + list(extra_data)) if len(extra_data) > 0 else data
    if idx is None:
        if not isinstance(data, pd.DataFrame):
            raise ValueError(
                "Can only select based on row names with DataFrame input. "
                "Please set `idx` to select specific rows.")
        idx = get_cell_set(data, starts_with=starts_with,
                           ends_with=ends_with, regex=regex)
    if isinstance(idx, pd.DataFrame):
        idx = _convert_dataframe_1d(idx)
    if isinstance(data, (pd.DataFrame, pd.Series)):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "error", "Passing list-likes to .loc")
                data = data.loc[idx]
        except (KeyError, TypeError, FutureWarning):
            if isinstance(idx, numbers.Integral) or \
                    issubclass(np.array(idx).dtype.type, numbers.Integral):
                data = data.iloc[idx]
            else:
                raise
    elif _is_1d(data):
        if isinstance(data, list):
            # can't numpy index a list
            data = np.array(data)
        data = data[idx]
    else:
        if isinstance(data, (sparse.coo_matrix,
                             sparse.bsr_matrix,
                             sparse.dia_matrix)):
            data = data.tocsr()
        data = data[idx, :]
    if _get_row_length(data) == 0:
        warnings.warn("Selecting 0 rows.", UserWarning)
    if len(extra_data) > 0:
        data = [data]
        for d in extra_data:
            data.append(select_rows(d, idx=idx))
        data = tuple(data)
    return data


def subsample(*data, n=10000, seed=None):
    """Subsample the number of points in a dataset

    Selects a random subset of (optionally multiple) datasets.
    Helpful for plotting, or for methods with computational
    constraints.

    Parameters
    ----------
    data : array-like, shape=[n_samples, *]
        Input data. Any number of datasets can be passed at once,
        so long as `n_samples` remains the same.
    n : int, optional (default: 10000)
        Number of samples to retain. Must be less than `n_samples`.
    seed : int, optional (default: None)
        Random seed

    Examples
    --------
    data_subsample, labels_subsample = scprep.utils.subsample(data, labels, n=1000)
    """
    N = data[0].shape[0]
    if len(data) > 1:
        _check_rows_compatible(*data)
    if N < n:
        raise ValueError("Expected n ({}) <= n_samples ({})".format(n, N))
    np.random.seed(seed)
    select_idx = np.random.choice(N, n, replace=False)
    data = [select_rows(d, idx=select_idx) for d in data]
    return tuple(data) if len(data) > 1 else data[0]
