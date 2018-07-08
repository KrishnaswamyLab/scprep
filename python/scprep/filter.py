# author: Scott Gigante <scott.gigante@yale.edu>
# (C) 2018 Krishnaswamy Lab GPLv2

import numpy as np

from .utils import select_rows, select_cols
from .measure import library_size, gene_set_expression, _get_percentile_cutoff


def remove_empty_genes(data):
    """Remove all genes with zero counts across all cells

    This is equivalent to `remove_rare_genes(data, cutoff=0, min_cells=1)`
    but should be faster.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data

    Returns
    -------
    data : array-like, shape=[n_samples, m_features]
        Filtered output data, where m_features <= n_features
    """
    gene_sums = np.array(data.sum(axis=0)).reshape(-1)
    keep_genes_idx = gene_sums > 0
    data = select_cols(data, keep_genes_idx)
    return data


def remove_rare_genes(data, cutoff=0, min_cells=5):
    """Remove all genes with negligible counts in all but a few cells

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    cutoff : float, optional (default: 0)
        Number of counts above which expression is deemed non-negligible
    min_cells : int, optional (default: 5)
        Minimum number of cells above `cutoff` in order to retain a gene

    Returns
    -------
    data : array-like, shape=[n_samples, m_features]
        Filtered output data, where m_features <= n_features
    """
    gene_sums = np.array((data > cutoff).sum(axis=0)).reshape(-1)
    keep_genes_idx = gene_sums >= min_cells
    data = select_cols(data, keep_genes_idx)
    return data


def remove_empty_cells(data):
    """Remove all cells with zero library size

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data

    Returns
    -------
    data : array-like, shape=[m_samples, n_features]
        Filtered output data, where m_samples <= n_samples
    """
    cell_sums = library_size(data)
    keep_cells_idx = cell_sums > 0
    data = select_rows(data, keep_cells_idx)
    return data


def filter_library_size(data, cutoff=2000):
    """Remove all cells with library size below a certain value

    It is recommended to use :func:`~scprep.plot.plot_library_size` to
    choose a cutoff prior to filtering.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    cutoff : float, optional (default: 2000)
        Minimum library size required to retain a cell

    Returns
    -------
    data : array-like, shape=[m_samples, n_features]
        Filtered output data, where m_samples <= n_samples
    """
    cell_sums = library_size(data)
    keep_cells_idx = cell_sums > cutoff
    data = select_rows(data, keep_cells_idx)
    return data


def filter_gene_set_expression(data, genes,
                               cutoff=None, percentile=None,
                               keep_cells='below'):
    """Remove cells with total expression of a gene set below a certain value

    It is recommended to use :func:`~scprep.plot.plot_gene_set_expression` to
    choose a cutoff prior to filtering.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    genes : list-like
        Integer column indices or string gene names included in gene set
    cutoff : float, optional (default: 2000)
        Value above or below which to remove cells. Only one of `cutoff`
        and `percentile` should be specified.
    percentile : int (Default: None)
        Percentile above or below which to remove cells.
        Must be an integer between 0 and 100. Only one of `cutoff`
        and `percentile` should be specified.
    keep_cells : {'above', 'below'}
        Keep cells above or below the cutoff
    """
    cell_sums = gene_set_expression(data, genes)
    cutoff = _get_percentile_cutoff(
        cell_sums, cutoff, percentile, required=True)
    if keep_cells == 'above':
        keep_cells_idx = cell_sums > cutoff
    elif keep_cells == 'below':
        keep_cells_idx = cell_sums < cutoff
    else:
        raise ValueError("Expected `keep_cells in ['above', 'below']. "
                         "Got {}".format(keep_cells))
    data = select_rows(data, keep_cells_idx)
    return data
