import numpy as np
import pandas as pd
import warnings
import numbers
import scipy.signal
from scipy import sparse

from . import utils, select


def library_size(data):
    """Measure the library size of each cell.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data

    Returns
    -------
    library_size : list-like, shape=[n_samples]
        Sum over all genes for each cell
    """
    library_size = utils.matrix_sum(data, axis=1)
    if isinstance(library_size, pd.Series):
        library_size.name = "library_size"
    return library_size


def gene_set_expression(
    data,
    genes=None,
    library_size_normalize=False,
    starts_with=None,
    ends_with=None,
    exact_word=None,
    regex=None,
):
    """Measure the expression of a set of genes in each cell.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    genes : list-like, shape<=[n_features], optional (default: None)
        Integer column indices or string gene names included in gene set
    library_size_normalize : bool, optional (default: False)
        Divide gene set expression by library size
    starts_with : str or None, optional (default: None)
        If not None, select genes that start with this prefix
    ends_with : str or None, optional (default: None)
        If not None, select genes that end with this suffix
    exact_word : str, list-like or None, optional (default: None)
        If not None, select genes that contain this exact word.
    regex : str or None, optional (default: None)
        If not None, select genes that match this regular expression

    Returns
    -------
    gene_set_expression : list-like, shape=[n_samples]
        Sum over genes for each cell
    """
    if library_size_normalize:
        from .normalize import library_size_normalize

        data = library_size_normalize(data)
    gene_data = select.select_cols(
        data,
        idx=genes,
        starts_with=starts_with,
        ends_with=ends_with,
        exact_word=exact_word,
        regex=regex,
    )
    if len(gene_data.shape) > 1:
        gene_set_expression = library_size(gene_data)
    else:
        gene_set_expression = gene_data
    if isinstance(gene_set_expression, pd.Series):
        gene_set_expression.name = "expression"
    return gene_set_expression


def gene_variability(data, kernel_size=0.005, smooth=5, return_means=False):
    """Measure the variability of each gene in a dataset

    Variability is computed as the deviation from
    the rolling median of the mean-variance curve

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    kernel_size : float or int, optional (default: 0.005)
        Width of rolling median window. If a float between 0 and 1, the width is given by
        kernel_size * data.shape[1]. Otherwise should be an odd integer
    smooth : int, optional (default: 5)
        Amount of smoothing to apply to the median filter
    return_means : boolean, optional (default: False)
        If True, return the gene means

    Returns
    -------
    variability : list-like, shape=[n_samples]
        Variability for each gene
    """
    columns = data.columns if isinstance(data, pd.DataFrame) else None
    data = utils.to_array_or_spmatrix(data)
    if isinstance(data, sparse.dia_matrix):
        data = data.tocsc()
    data_std = utils.matrix_std(data, axis=0) ** 2
    data_mean = utils.toarray(data.mean(axis=0)).flatten()

    if kernel_size < 1:
        kernel_size = 2 * (int(kernel_size * len(data_std)) // 2) + 1

    order = np.argsort(data_mean)
    data_std_med = np.empty_like(data_std)
    data_std_order = data_std[order]
    # handle overhang with reflection
    data_std_order = np.r_[
        data_std_order[kernel_size::-1],
        data_std_order,
        data_std_order[:-kernel_size:-1],
    ]
    medfilt = scipy.signal.medfilt(data_std_order, kernel_size=kernel_size)[
        kernel_size:-kernel_size
    ]

    # apply a little smoothing
    for i in range(smooth):
        medfilt = np.r_[(medfilt[1:] + medfilt[:-1]) / 2, medfilt[-1]]

    data_std_med[order] = medfilt
    result = data_std - data_std_med

    if columns is not None:
        result = pd.Series(result, index=columns, name="variability")
        data_mean = pd.Series(data_mean, index=columns, name="mean")
    if return_means:
        result = result, data_mean
    return result


def gene_capture_count(data, cutoff=0):
    """Measure the number of cells in which each gene has non-negligible counts

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    cutoff : float, optional (default: 0)
        Number of counts above which expression is deemed non-negligible

    Returns
    -------
    capture-count : list-like, shape=[m_features]
        Capture count for each gene
    """
    gene_sums = np.array(utils.matrix_sum(data > cutoff, axis=0)).reshape(-1)
    if isinstance(data, pd.DataFrame):
        gene_sums = pd.Series(gene_sums, index=data.columns, name="capture_count")
    return gene_sums
