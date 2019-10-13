import numpy as np
import pandas as pd
import warnings
import numbers
import scipy.signal

from . import utils, select
from ._lazyload import statsmodels


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
        library_size.name = 'library_size'
    return library_size


def gene_set_expression(data, genes=None, library_size_normalize=False,
                        starts_with=None, ends_with=None,
                        exact_word=None, regex=None):
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
    gene_data = select.select_cols(data, idx=genes, starts_with=starts_with,
                                   ends_with=ends_with,
                                   exact_word=exact_word, regex=regex)
    if len(gene_data.shape) > 1:
        gene_set_expression = library_size(gene_data)
    else:
        gene_set_expression = gene_data
    if isinstance(gene_set_expression, pd.Series):
        gene_set_expression.name = 'expression'
    return gene_set_expression


@utils._with_pkg(pkg="statsmodels")
def gene_variability(data, span=0.7, interpolate=0.2, kernel_size=0.05, return_means=False):
    """Measure the variability of each gene in a dataset

    Variability is computed as the deviation from a loess fit
    to the rolling median of the mean-variance curve

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    span : float, optional (default: 0.7)
        Fraction of genes to use when computing the loess estimate at each point
    interpolate : float, optional (default: 0.2)
        Multiple of the standard deviation of variances at which to interpolate
        linearly in order to reduce computation time.
    kernel_size : float or int, optional (default: 0.05)
        Width of rolling median window. If a float between 0 and 1, the width is given by
        kernel_size * data.shape[1]. Otherwise should be an odd integer
    return_means : boolean, optional (default: False)
        If True, return the gene means

    Returns
    -------
    variability : list-like, shape=[n_samples]
        Variability for each gene
    """
    columns = data.columns if isinstance(data, pd.DataFrame) else None
    data = utils.to_array_or_spmatrix(data)
    data_std = utils.matrix_std(data, axis=0) ** 2
    if kernel_size < 1:
        kernel_size = 2*(int(kernel_size * len(data_std))//2)+1
    order = np.argsort(data_std)
    data_std_med = np.empty_like(data_std)
    data_std_med[order] = scipy.signal.medfilt(data_std[order], kernel_size=kernel_size)
    data_mean = utils.toarray(np.mean(data, axis=0)).flatten()
    delta = np.std(data_std_med) * interpolate
    lowess = statsmodels.nonparametric.smoothers_lowess.lowess(
        data_std_med, data_mean,
        delta=delta, frac=span, return_sorted=False)
    result = data_std - lowess
    if columns is not None:
        result = pd.Series(result, index=columns, name='variability')
        data_mean = pd.Series(data_mean, index=columns, name='mean')
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
        gene_sums = pd.Series(gene_sums, index=data.columns, name='capture_count')
    return gene_sums
