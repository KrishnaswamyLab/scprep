# author: Daniel Burkhardt <daniel.burkhardt@yale.edu>
# (C) 2018 Krishnaswamy Lab GPLv2

from sklearn.preprocessing import normalize
import numpy as np
from scipy import sparse
import pandas as pd
import numbers
import warnings
from . import measure, utils


def _get_scaled_libsize(data, rescale=10000, return_library_size=False):
    if return_library_size or rescale in ["median", "mean"]:
        libsize = measure.library_size(data)
    else:
        libsize = None
    if rescale == "median":
        rescale = np.median(utils.toarray(libsize))
        if rescale == 0:
            warnings.warn(
                "Median library size is zero. " "Rescaling to mean instead.",
                UserWarning,
            )
            rescale = np.mean(utils.toarray(libsize))
    elif rescale == "mean":
        rescale = np.mean(utils.toarray(libsize))
    elif isinstance(rescale, numbers.Number):
        pass
    elif rescale is None:
        rescale = 1
    else:
        raise ValueError(
            "Expected rescale in ['median', 'mean'], a number "
            "or `None`. Got {}".format(rescale)
        )
    return rescale, libsize


def library_size_normalize(data, rescale=10000, return_library_size=False):
    """Performs L1 normalization on input data
    Performs L1 normalization on input data such that the sum of expression
    values for each cell sums to 1
    then returns normalized matrix to the metric space using median UMI count
    per cell effectively scaling all cells as if they were sampled evenly.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    rescale : {'mean', 'median'}, float or `None`, optional (default: 10000)
        Rescaling strategy. If 'mean' or 'median', normalized cells are scaled
        back up to the mean or median expression value. If a float,
        normalized cells are scaled up to the given value. If `None`, no
        rescaling is done and all cells will have normalized library size of 1.
    return_library_size : bool, optional (default: False)
        If True, also return the library size pre-normalization

    Returns
    -------
    data_norm : array-like, shape=[n_samples, n_features]
        Library size normalized output data
    filtered_library_size : list-like, shape=[m_samples]
        Library size of cells pre-normalization,
        returned only if return_library_size is True
    """
    # pandas support
    columns, index = None, None
    if isinstance(data, pd.DataFrame):
        columns, index = data.columns, data.index
        if utils.is_sparse_dataframe(data):
            data = data.sparse.to_coo()
        elif utils.is_SparseDataFrame(data):
            data = data.to_coo()
        else:
            # dense data
            data = data.to_numpy()

    calc_libsize = sparse.issparse(data) and (return_library_size or data.nnz > 2 ** 31)
    rescale, libsize = _get_scaled_libsize(data, rescale, calc_libsize)

    if libsize is not None:
        divisor = utils.toarray(libsize)
        data_norm = utils.matrix_vector_elementwise_multiply(
            data, 1 / np.where(divisor == 0, 1, divisor), axis=0
        )
    else:
        if return_library_size:
            data_norm, libsize = normalize(data, norm="l1", axis=1, return_norm=True)
        else:
            data_norm = normalize(data, norm="l1", axis=1)
    data_norm = data_norm * rescale

    if columns is not None:
        # pandas dataframe
        if sparse.issparse(data_norm):
            data_norm = utils.SparseDataFrame(data_norm, default_fill_value=0.0)
        else:
            data_norm = pd.DataFrame(data_norm)
        data_norm.columns = columns
        data_norm.index = index
        libsize = pd.Series(libsize, index=index, name="library_size", dtype="float64")
    if return_library_size:
        return data_norm, libsize
    else:
        return data_norm


def batch_mean_center(data, sample_idx=None):
    """Performs batch mean-centering on the data

    The features of the data are all centered such that
    the column means are zero. Each batch is centered separately.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    sample_idx : list-like, optional
        Batch indices. If `None`, data is assumed to be a single batch

    Returns
    -------
    data : array-like, shape=[n_samples, n_features]
        Batch mean-centered output data.
    """
    if (
        sparse.issparse(data)
        or utils.is_SparseDataFrame(data)
        or utils.is_sparse_dataframe(data)
    ):
        raise ValueError(
            "Cannot mean center sparse data. " "Convert to dense matrix first."
        )
    if sample_idx is None:
        sample_idx = np.ones(len(data))
    else:
        sample_idx = utils.toarray(sample_idx).flatten()
    for sample in np.unique(sample_idx):
        idx = sample_idx == sample
        if isinstance(data, pd.DataFrame):
            feature_means = data.iloc[idx].mean(axis=0)
            data.iloc[idx] = data.iloc[idx] - feature_means
        else:
            feature_means = np.mean(data[idx], axis=0)
            data[idx] = data[idx] - feature_means[None, :]
    return data
