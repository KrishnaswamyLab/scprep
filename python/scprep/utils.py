import numpy as np
import pandas as pd
import numbers
from scipy import sparse
import warnings
import re


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
    data : array-like, shape=[m_samples, n_features]
        Subsetted output data

    Raises
    ------
    UserWarning : if no rows are selected
    """
    if isinstance(data, pd.DataFrame):
        try:
            data = data.loc[idx]
        except KeyError:
            if isinstance(idx, numbers.Integral) or \
                    issubclass(np.array(idx).dtype.type, numbers.Integral):
                data = data.iloc[idx]
            else:
                raise
    else:
        if isinstance(data, (sparse.coo_matrix,
                             sparse.bsr_matrix,
                             sparse.dia_matrix)):
            data = data.tocsr()
        data = data[idx, :]
    if data.shape[0] == 0:
        warnings.warn("Selecting 0 rows.", UserWarning)
    return data


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
    if len(data.shape) > 1:
        try:
            data = data.columns
        except AttributeError:
            raise TypeError("data must be a list of gene names or a pandas "
                            "DataFrame. Got {}".format(type(data).__name__))
    return _get_string_subset(data, starts_with=starts_with,
                              ends_with=ends_with, regex=regex)


def get_cell_set(data, starts_with=None, ends_with=None, regex=None):
    """Get a list of cells from data

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features] or [n_features]
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
    if len(data.shape) > 1:
        try:
            data = data.index
        except AttributeError:
            raise TypeError("data must be a list of cell names or a pandas "
                            "DataFrame. Got {}".format(type(data).__name__))
    return _get_string_subset(data, starts_with=starts_with,
                              ends_with=ends_with, regex=regex)


def combine_batches(data, batch_labels, append_to_cell_names=False):
    """Combine data matrices from multiple batches and store a batch label

    Parameters
    ----------
    data : list of array-like, shape=[n_batch]
        All matrices must be of the same format and have the same number of
        columns (or genes.)
    batch_labels : list of `str`, shape=[n_batch]
        List of names assigned to each batch
    append_to_cell_names : bool, optional (default: False)
        If input is a pandas dataframe, add the batch label corresponding to
        each cell to its existing index (or cell name / barcode.)

    Returns
    -------
    data : data matrix, shape=[n_samples, n_features]
        Number of samples is the sum of numbers of samples of all batches.
        Number of features is the same as each of the batches.
    sample_labels : list-like, shape=[n_samples]
        Batch labels corresponding to each sample
    """
    if not len(data) == len(batch_labels):
        raise ValueError("Expected data ({}) and batch_labels ({}) to be the "
                         "same length.".format(len(data), len(batch_labels)))
    matrix_type = type(data[0])
    if not issubclass(matrix_type, (np.ndarray,
                                    pd.DataFrame,
                                    sparse.spmatrix)):
        raise ValueError("Expected data to contain pandas DataFrames, "
                         "scipy sparse matrices or numpy arrays. "
                         "Got {}".format(matrix_type.__name__))

    matrix_shape = data[0].shape[1]
    for d in data[1:]:
        if not isinstance(d, matrix_type):
            types = ", ".join([type(d).__name__ for d in data])
            raise TypeError("Expected data all of the same class. "
                            "Got {}".format(types))

    if not d.shape[1] == matrix_shape:
        shapes = ", ".join([str(d.shape[1]) for d in data])
        raise ValueError("Expected data all with the same number of "
                         "columns. Got {}".format(shapes))

    if append_to_cell_names and not issubclass(matrix_type, pd.DataFrame):
        warnings.warn("append_to_cell_names only valid for pd.DataFrame input."
                      " Got {}".format(matrix_type.__name__), UserWarning)

    sample_labels = np.concatenate([np.repeat(batch_labels[i], d.shape[0])
                                    for i, d in enumerate(data)])
    if issubclass(matrix_type, pd.DataFrame):
        if append_to_cell_names:
            index = np.concatenate(
                [np.core.defchararray.add(np.array(d.index, dtype=str),
                                          "_" + str(batch_labels[i]))
                 for i, d in enumerate(data)])
        data = pd.concat(data)
        if append_to_cell_names:
            data.index = index
    elif issubclass(matrix_type, sparse.spmatrix):
        data = sparse.vstack(data)
    elif issubclass(matrix_type, np.ndarray):
        data = np.vstack(data)

    return data, sample_labels
