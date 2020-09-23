import numpy as np
import pandas as pd
import numbers
from scipy import sparse
import warnings
import re
import sys

from . import utils

if int(sys.version.split(".")[1]) < 7:
    _re_pattern = type(re.compile(""))
else:
    _re_pattern = re.Pattern


def _is_1d(data):
    try:
        return len(data.shape) == 1
    except AttributeError:
        return True


def _check_idx_1d(idx, silent=False):
    if (not _is_1d(idx)) and np.prod(idx.shape) != np.max(idx.shape):
        if silent:
            return False
        else:
            raise ValueError("Expected idx to be 1D. Got shape {}".format(idx.shape))
    else:
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
                "Expected `data` and `extra_data` to have the same number of "
                "columns. Got {}".format([_get_column_length(d) for d in data])
            )
        if isinstance(d, (pd.DataFrame, pd.Series)) and isinstance(
            data[0], (pd.DataFrame, pd.Series)
        ):
            if not np.all(_get_columns(data[0]) == _get_columns(d)):
                raise ValueError(
                    "Expected `data` and `extra_data` pandas inputs to have "
                    "the same column names. Fix with "
                    "`scprep.select.select_cols(*extra_data, idx=data.columns)`"
                )


def _check_rows_compatible(*data):
    for d in data:
        if not _get_row_length(d) == _get_row_length(data[0]):
            raise ValueError(
                "Expected `data` and `extra_data` to have the same number of "
                "rows. Got {}".format([d.shape[0] for d in data])
            )
        if isinstance(d, (pd.DataFrame, pd.Series)) and isinstance(
            data[0], (pd.DataFrame, pd.Series)
        ):
            if not np.all(data[0].index == d.index):
                raise ValueError(
                    "Expected `data` and `extra_data` pandas inputs to have "
                    "the same index. Fix with "
                    "`scprep.select.select_rows(*extra_data, idx=data.index)`"
                )


def _convert_dataframe_1d(idx, silent=False):
    if _check_idx_1d(idx, silent=silent):
        idx = idx.iloc[:, 0] if idx.shape[1] == 1 else idx.iloc[0, :]
    return idx


def _string_vector_match(data, match, fun, dtype=str):
    """Get a boolean match array from a vector

    Parameters
    ----------
    data : list-like
        Vector to be matched against
    match : `dtype` or list-like
        Match criteria
    fun : callable(x, match)
        Function that returns True if `match` matches `x`
    dtype : type, optional (default: str)
        Expected type(match) (if not list-like)

    Returns
    -------
    data_match : list-like, dtype=bool
    """
    if isinstance(match, dtype):
        fun = np.vectorize(fun)
        return fun(data, match)
    else:
        return np.any(
            [_string_vector_match(data, m, fun, dtype=dtype) for m in match], axis=0
        )


def _exact_word_regex(word):
    allowed_chars = ["\\(", "\\)", "\\[", "\\]", "\\.", ",", "!", "\\?", " ", "^", "$"]
    wildcard = "(" + "|".join(allowed_chars) + ")+"
    return "{wildcard}{word}{wildcard}".format(wildcard=wildcard, word=re.escape(word))


def _get_string_subset_mask(
    data, starts_with=None, ends_with=None, exact_word=None, regex=None
):
    """Get a subset from a string array

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features] or [n_features]
        Input pd.DataFrame, or list of names
    starts_with : str, list-like or None, optional (default: None)
        If not None, only return names that start with this prefix.
    ends_with : str, list-like or None, optional (default: None)
        If not None, only return names that end with this suffix.
    exact_word : str, list-like or None, optional (default: None)
        If not None, only return names that contain this exact word.
    regex : str, list-like or None, optional (default: None)
        If not None, only return names that match this regular expression.

    Returns
    -------
    data : list-like, shape<=[n_features]
        List of matching strings
    """
    mask = np.full_like(data, True, dtype=bool)
    if starts_with is not None:
        start_match = _string_vector_match(
            data, starts_with, lambda x, match: x.startswith(match)
        )
        mask = np.logical_and(mask, start_match)
    if ends_with is not None:
        end_match = _string_vector_match(
            data, ends_with, lambda x, match: x.endswith(match)
        )
        mask = np.logical_and(mask, end_match)
    if exact_word is not None:
        if not isinstance(exact_word, str):
            exact_word = [_exact_word_regex(w) for w in exact_word]
        else:
            exact_word = _exact_word_regex(exact_word)
        exact_word_match = _get_string_subset_mask(data, regex=exact_word)
        mask = np.logical_and(mask, exact_word_match)
    if regex is not None:
        if not isinstance(regex, str):
            regex = [re.compile(r) for r in regex]
        else:
            regex = re.compile(regex)
        regex_match = _string_vector_match(
            data, regex, lambda x, match: bool(match.search(x)), dtype=_re_pattern
        )
        mask = np.logical_and(mask, regex_match)
    return mask


def _get_string_subset(
    data, starts_with=None, ends_with=None, exact_word=None, regex=None
):
    """Get a subset from a string array

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features] or [n_features]
        Input pd.DataFrame, or list of names
    starts_with : str, list-like or None, optional (default: None)
        If not None, only return names that start with this prefix.
    ends_with : str, list-like or None, optional (default: None)
        If not None, only return names that end with this suffix.
    exact_word : str, list-like or None, optional (default: None)
        If not None, only return names that contain this exact word.
    regex : str, list-like or None, optional (default: None)
        If not None, only return names that match this regular expression.

    Returns
    -------
    data : list-like, shape<=[n_features]
        List of matching strings
    """
    data = utils.toarray(data)
    mask = _get_string_subset_mask(
        data,
        starts_with=starts_with,
        ends_with=ends_with,
        exact_word=exact_word,
        regex=regex,
    )
    return data[mask]


def get_gene_set(data, starts_with=None, ends_with=None, exact_word=None, regex=None):
    """Get a list of genes from data

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features] or [n_features]
        Input pd.DataFrame, or list of gene names
    starts_with : str, list-like or None, optional (default: None)
        If not None, only return gene names that start with this prefix.
    ends_with : str, list-like or None, optional (default: None)
        If not None, only return gene names that end with this suffix.
    exact_word : str, list-like or None, optional (default: None)
        If not None, only return gene names that contain this exact word.
    regex : str, list-like or None, optional (default: None)
        If not None, only return gene names that match this regular expression.

    Returns
    -------
    genes : list-like, shape<=[n_features]
        List of matching genes
    """
    if not _is_1d(data):
        try:
            data = data.columns.to_numpy()
        except AttributeError:
            raise TypeError(
                "data must be a list of gene names or a pandas "
                "DataFrame. Got {}".format(type(data).__name__)
            )
    if (
        starts_with is None
        and ends_with is None
        and regex is None
        and exact_word is None
    ):
        warnings.warn(
            "No selection conditions provided. " "Returning all genes.", UserWarning
        )
    return _get_string_subset(
        data,
        starts_with=starts_with,
        ends_with=ends_with,
        exact_word=exact_word,
        regex=regex,
    )


def get_cell_set(data, starts_with=None, ends_with=None, exact_word=None, regex=None):
    """Get a list of cells from data

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features] or [n_samples]
        Input pd.DataFrame, or list of cell names
    starts_with : str, list-like or None, optional (default: None)
        If not None, only return cell names that start with this prefix.
    ends_with : str, list-like or None, optional (default: None)
        If not None, only return cell names that end with this suffix.
    exact_word : str, list-like or None, optional (default: None)
        If not None, only return cell names that contain this exact word.
    regex : str, list-like or None, optional (default: None)
        If not None, only return cell names that match this regular expression.

    Returns
    -------
    cells : list-like, shape<=[n_features]
        List of matching cells
    """
    if not _is_1d(data):
        try:
            data = data.index.to_numpy()
        except AttributeError:
            raise TypeError(
                "data must be a list of cell names or a pandas "
                "DataFrame. Got {}".format(type(data).__name__)
            )
    if (
        starts_with is None
        and ends_with is None
        and regex is None
        and exact_word is None
    ):
        warnings.warn(
            "No selection conditions provided. Returning all cells.", UserWarning
        )
    return _get_string_subset(
        data,
        starts_with=starts_with,
        ends_with=ends_with,
        exact_word=exact_word,
        regex=regex,
    )


def select_cols(
    data,
    *extra_data,
    idx=None,
    starts_with=None,
    ends_with=None,
    exact_word=None,
    regex=None
):
    """Select columns from a data matrix

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    extra_data : array-like, shape=[any, n_features], optional
        Optional additional data objects from which to select the same rows
    idx : list-like, shape=[m_features]
        Integer indices or string column names to be selected
    starts_with : str, list-like or None, optional (default: None)
        If not None, select columns that start with this prefix.
    ends_with : str, list-like or None, optional (default: None)
        If not None, select columns that end with this suffix.
    exact_word : str, list-like or None, optional (default: None)
        If not None, select columns that contain this exact word.
    regex : str, list-like or None, optional (default: None)
        If not None, select columns that match this regular expression.

    Returns
    -------
    data : array-like, shape=[n_samples, m_features]
        Subsetted output data.
    extra_data : array-like, shape=[any, m_features]
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
    if (
        idx is None
        and starts_with is None
        and ends_with is None
        and exact_word is None
        and regex is None
    ):
        warnings.warn(
            "No selection conditions provided. Returning all columns.", UserWarning
        )
        return tuple([data] + list(extra_data)) if len(extra_data) > 0 else data
    if idx is None:
        if not isinstance(data, pd.DataFrame):
            raise ValueError(
                "Can only select based on column names with DataFrame input. "
                "Please set `idx` to select specific columns."
            )
        idx = get_gene_set(
            data,
            starts_with=starts_with,
            ends_with=ends_with,
            exact_word=exact_word,
            regex=regex,
        )

    if isinstance(idx, pd.DataFrame):
        idx = _convert_dataframe_1d(idx)
    elif not isinstance(idx, (numbers.Integral, str)):
        idx = utils.toarray(idx)
        _check_idx_1d(idx)
        idx = idx.flatten()

    if utils.is_SparseDataFrame(data):
        # evil deprecated dataframe; get rid of it
        data = utils.SparseDataFrame(data)

    input_1d = _is_1d(data)
    if isinstance(data, pd.DataFrame):
        try:
            if isinstance(idx, (numbers.Integral, str)):
                data = data.loc[:, idx]
            else:
                if np.issubdtype(idx.dtype, np.dtype(bool).type):
                    # temporary workaround for pandas error
                    raise TypeError
                data = data.loc[:, idx]
        except (KeyError, TypeError):
            if isinstance(idx, str):
                raise
            if (
                isinstance(idx, numbers.Integral)
                or np.issubdtype(idx.dtype, np.dtype(int))
                or np.issubdtype(idx.dtype, np.dtype(bool))
            ):
                data = data.loc[:, np.array(data.columns)[idx]]
            else:
                raise
    elif isinstance(data, pd.Series):
        try:
            if np.issubdtype(idx.dtype, np.dtype(bool).type):
                # temporary workaround for pandas error
                raise TypeError
            data = data.loc[idx]
        except (KeyError, TypeError):
            if (
                isinstance(idx, numbers.Integral)
                or np.issubdtype(idx.dtype, np.dtype(int))
                or np.issubdtype(idx.dtype, np.dtype(bool))
            ):
                data = data.loc[np.array(data.index)[idx]]
            else:
                raise
    elif _is_1d(data):
        if isinstance(data, list):
            # can't numpy index a list
            data = np.array(data)
        data = data[idx]
    else:
        if isinstance(
            data,
            (
                sparse.coo_matrix,
                sparse.bsr_matrix,
                sparse.lil_matrix,
                sparse.dia_matrix,
            ),
        ):
            data = data.tocsr()
        if isinstance(idx, pd.Series):
            idx = utils.toarray(idx)
        data = data[:, idx]
    if _get_column_length(data) == 0:
        warnings.warn("Selecting 0 columns.", UserWarning)
    elif isinstance(data, pd.DataFrame) and not input_1d:
        # convert to series if possible
        data = _convert_dataframe_1d(data, silent=True)
    if len(extra_data) > 0:
        data = [data]
        for d in extra_data:
            data.append(select_cols(d, idx=idx))
        data = tuple(data)
    return data


def select_rows(
    data,
    *extra_data,
    idx=None,
    starts_with=None,
    ends_with=None,
    exact_word=None,
    regex=None
):
    """Select rows from a data matrix

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    extra_data : array-like, shape=[n_samples, any], optional
        Optional additional data objects from which to select the same rows
    idx : list-like, shape=[m_samples], optional (default: None)
        Integer indices or string index names to be selected
    starts_with : str, list-like or None, optional (default: None)
        If not None, select rows that start with this prefix.
    ends_with : str, list-like or None, optional (default: None)
        If not None, select rows that end with this suffix.
    exact_word : str, list-like or None, optional (default: None)
        If not None, select rows that contain this exact word.
    regex : str, list-like or None, optional (default: None)
        If not None, select rows that match this regular expression.

    Returns
    -------
    data : array-like, shape=[m_samples, n_features]
        Subsetted output data
    extra_data : array-like, shape=[m_samples, any]
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
    if (
        idx is None
        and starts_with is None
        and ends_with is None
        and exact_word is None
        and regex is None
    ):
        warnings.warn(
            "No selection conditions provided. " "Returning all rows.", UserWarning
        )
        return tuple([data] + list(extra_data)) if len(extra_data) > 0 else data
    if idx is None:
        if not isinstance(data, pd.DataFrame):
            raise ValueError(
                "Can only select based on row names with DataFrame input. "
                "Please set `idx` to select specific rows."
            )
        idx = get_cell_set(
            data,
            starts_with=starts_with,
            ends_with=ends_with,
            exact_word=exact_word,
            regex=regex,
        )

    if isinstance(idx, pd.DataFrame):
        idx = _convert_dataframe_1d(idx)
    elif not isinstance(idx, (numbers.Integral, str)):
        idx = utils.toarray(idx)
        _check_idx_1d(idx)
        idx = idx.flatten()

    if utils.is_SparseDataFrame(data):
        # evil deprecated dataframe; get rid of it
        data = utils.SparseDataFrame(data)

    input_1d = _is_1d(data)
    if isinstance(data, (pd.DataFrame, pd.Series)):
        try:
            if isinstance(idx, (numbers.Integral, str)):
                data = data.loc[idx]
            else:
                if np.issubdtype(idx.dtype, np.dtype(bool).type):
                    # temporary workaround for pandas error
                    raise TypeError
                with warnings.catch_warnings():
                    warnings.filterwarnings("error", "Passing list-likes to .loc")
                    data = data.loc[idx]
        except (KeyError, TypeError, FutureWarning):
            if isinstance(idx, str):
                raise
            if (
                isinstance(idx, numbers.Integral)
                or np.issubdtype(idx.dtype, np.dtype(int))
                or np.issubdtype(idx.dtype, np.dtype(bool))
            ):
                data = data.loc[np.array(data.index)[idx]]
            else:
                raise
    elif _is_1d(data):
        if isinstance(data, list):
            # can't numpy index a list
            data = np.array(data)
        data = data[idx]
    else:
        if isinstance(data, (sparse.coo_matrix, sparse.bsr_matrix, sparse.dia_matrix)):
            data = data.tocsr()
        if isinstance(idx, pd.Series):
            idx = utils.toarray(idx)
        data = data[idx, :]
    if _get_row_length(data) == 0:
        warnings.warn("Selecting 0 rows.", UserWarning)
    elif isinstance(data, pd.DataFrame) and not input_1d:
        # convert to series if possible
        data = _convert_dataframe_1d(data, silent=True)
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
    data : array-like, shape=[n_samples, any]
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
    select_idx = np.isin(np.arange(N), np.random.choice(N, n, replace=False))
    data = [select_rows(d, idx=select_idx) for d in data]
    return tuple(data) if len(data) > 1 else data[0]


def highly_variable_genes(
    data, *extra_data, kernel_size=0.05, smooth=5, cutoff=None, percentile=80
):
    """Select genes with high variability

    Variability is computed as the deviation from a loess fit
    to the rolling median of the mean-variance curve

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    extra_data : array-like, shape=[any, n_features], optional
        Optional additional data objects from which to select the same rows
    kernel_size : float or int, optional (default: 0.005)
        Width of rolling median window. If a float between 0 and 1, the width is given by
        kernel_size * data.shape[1]. Otherwise should be an odd integer
    smooth : int, optional (default: 5)
        Amount of smoothing to apply to the median filter
    cutoff : float, optional (default: None)
        Variability above which expression is deemed significant
    percentile : int, optional (Default: 80)
        Percentile above or below which to remove genes.
        Must be an integer between 0 and 100. Only one of `cutoff`
        and `percentile` should be specified.

    Returns
    -------
    data : array-like, shape=[n_samples, m_features]
        Filtered output data, where m_features <= n_features
    extra_data : array-like, shape=[any, m_features]
        Filtered extra data, if passed.
    """
    from . import measure

    var_genes = measure.gene_variability(data, kernel_size=kernel_size, smooth=smooth)
    keep_cells_idx = utils._get_filter_idx(
        var_genes, cutoff, percentile, keep_cells="above"
    )
    return select_cols(data, *extra_data, idx=keep_cells_idx)
