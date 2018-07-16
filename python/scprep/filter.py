# author: Scott Gigante <scott.gigante@yale.edu>
# (C) 2018 Krishnaswamy Lab GPLv2

import numpy as np

from . import utils, measure


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
    data = utils.select_cols(data, keep_genes_idx)
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
    data = utils.select_cols(data, keep_genes_idx)
    return data


def remove_empty_cells(data, sample_labels=None):
    """Remove all cells with zero library size

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    sample_labels : list-like or None, optional, shape=[n_samples] (default: None)
        Labels associated with the rows of `data`. If provided, these
        will be filtered such that they retain a one-to-one mapping
        with the rows of the output data.

    Returns
    -------
    data : array-like, shape=[m_samples, n_features]
        Filtered output data, where m_samples <= n_samples
    """
    cell_sums = measure.library_size(data)
    keep_cells_idx = cell_sums > 0
    data = utils.select_rows(data, keep_cells_idx)
    if sample_labels is not None:
        sample_labels = sample_labels[keep_cells_idx]
        data = data, sample_labels
    return data


def filter_library_size(data, cutoff=None, percentile=None,
                        keep_cells='above', sample_labels=None):
    """Remove all cells with library size below a certain value

    It is recommended to use :func:`~scprep.plot.plot_library_size` to
    choose a cutoff prior to filtering.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    cutoff : float, optional (default: None)
        Minimum library size required to retain a cell. Only one of `cutoff`
        and `percentile` should be specified.
    percentile : int, optional (Default: None)
        Percentile above or below which to remove cells.
        Must be an integer between 0 and 100. Only one of `cutoff`
        and `percentile` should be specified.
    keep_cells : {'above', 'below'}, optional (default: 'above')
        Keep cells above or below the cutoff
    sample_labels : list-like or None, optional, shape=[n_samples] (default: None)
        Labels associated with the rows of `data`. If provided, these
        will be filtered such that they retain a one-to-one mapping
        with the rows of the output data.

    Returns
    -------
    data : array-like, shape=[m_samples, n_features]
        Filtered output data, where m_samples <= n_samples
    sample_labels : list-like, shape=[m_samples]
        Filtered sample labels, if provided
    """
    cell_sums = measure.library_size(data)
    cutoff = measure._get_percentile_cutoff(
        cell_sums, cutoff, percentile, required=True)
    if keep_cells == 'above':
        keep_cells_idx = cell_sums > cutoff
    elif keep_cells == 'below':
        keep_cells_idx = cell_sums < cutoff
    else:
        raise ValueError("Expected `keep_cells` in ['above', 'below']. "
                         "Got {}".format(keep_cells))
    data = utils.select_rows(data, keep_cells_idx)
    if sample_labels is not None:
        sample_labels = sample_labels[keep_cells_idx]
        data = data, sample_labels
    return data


def filter_gene_set_expression(data, genes,
                               cutoff=None, percentile=None,
                               library_size_normalize=True,
                               keep_cells='below',
                               sample_labels=None):
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
    percentile : int, optional (Default: None)
        Percentile above or below which to remove cells.
        Must be an integer between 0 and 100. Only one of `cutoff`
        and `percentile` should be specified.
    library_size_normalize : bool, optional (default: True)
        Divide gene set expression by library size
    keep_cells : {'above', 'below'}, optional (default: 'below')
        Keep cells above or below the cutoff
    sample_labels : list-like or None, optional, shape=[n_samples] (default: None)
        Labels associated with the rows of `data`. If provided, these
        will be filtered such that they retain a one-to-one mapping
        with the rows of the output data.
    """
    cell_sums = measure.gene_set_expression(
        data, genes,
        library_size_normalize=library_size_normalize)
    cutoff = measure._get_percentile_cutoff(
        cell_sums, cutoff, percentile, required=True)
    if keep_cells == 'above':
        keep_cells_idx = cell_sums > cutoff
    elif keep_cells == 'below':
        keep_cells_idx = cell_sums < cutoff
    else:
        raise ValueError("Expected `keep_cells` in ['above', 'below']. "
                         "Got {}".format(keep_cells))
    data = utils.select_rows(data, keep_cells_idx)
    if sample_labels is not None:
        sample_labels = sample_labels[keep_cells_idx]
        data = data, sample_labels
    return data
