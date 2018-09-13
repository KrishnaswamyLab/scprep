# author: Scott Gigante <scott.gigante@yale.edu>
# (C) 2018 Krishnaswamy Lab GPLv2

import numpy as np
import pandas as pd

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
    gene_sums = np.array(utils.matrix_sum(data, axis=0)).reshape(-1)
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
    gene_sums = np.array(utils.matrix_sum(data > cutoff, axis=0)).reshape(-1)
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


def _get_filter_idx(data, values,
                    cutoff, percentile,
                    keep_cells):
    cutoff = measure._get_percentile_cutoff(
        values, cutoff, percentile, required=True)
    if keep_cells == 'above':
        keep_cells_idx = values > cutoff
    elif keep_cells == 'below':
        keep_cells_idx = values < cutoff
    else:
        raise ValueError("Expected `keep_cells` in ['above', 'below']. "
                         "Got {}".format(keep_cells))
    return keep_cells_idx


def filter_values(data, values,
                  cutoff=None, percentile=None,
                  keep_cells='above', sample_labels=None,
                  filter_per_sample=False):
    """Remove all cells with `values` above or below a certain threshold

    It is recommended to use :func:`~scprep.plot.histogram` to
    choose a cutoff prior to filtering.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    values : list-like, shape=[n_samples]
        Value upon which to filter
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
    filter_per_sample : bool, optional (default: False)
        If True, filters separately for each unique sample label. Only used
        if `sample_labels` is not `None` and `percentile` is given.

    Returns
    -------
    data : array-like, shape=[m_samples, n_features]
        Filtered output data, where m_samples <= n_samples
    sample_labels : list-like, shape=[m_samples]
        Filtered sample labels, if provided
    """
    if filter_per_sample and percentile is not None and \
            sample_labels is not None:
        # filter separately and combine
        sample_labels_array = utils.toarray(sample_labels).flatten()
        keep_cells_idx = np.full_like(
            sample_labels_array, True,
            dtype=bool)
        for label in np.unique(sample_labels_array):
            sample_idx = sample_labels_array == label
            keep_cells_idx[sample_idx] = _get_filter_idx(
                utils.select_rows(data, sample_idx),
                values[sample_idx],
                cutoff, percentile, keep_cells)
            keep_cells_idx = keep_cells_idx.flatten()
    else:
        keep_cells_idx = _get_filter_idx(data, values,
                                         cutoff, percentile,
                                         keep_cells)
    data = utils.select_rows(data, keep_cells_idx)
    if sample_labels is not None:
        sample_labels = sample_labels[keep_cells_idx]
        data = data, sample_labels
    return data


def filter_library_size(data, cutoff=None, percentile=None,
                        keep_cells='above', sample_labels=None,
                        filter_per_sample=False):
    """Remove all cells with library size above or below a certain threshold

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
    filter_per_sample : bool, optional (default: False)
        If True, filters separately for each unique sample label.

    Returns
    -------
    data : array-like, shape=[m_samples, n_features]
        Filtered output data, where m_samples <= n_samples
    sample_labels : list-like, shape=[m_samples]
        Filtered sample labels, if provided
    """
    cell_sums = measure.library_size(data)
    return filter_values(data, cell_sums,
                         cutoff=cutoff, percentile=percentile,
                         keep_cells=keep_cells,
                         sample_labels=sample_labels,
                         filter_per_sample=filter_per_sample)


def filter_gene_set_expression(data, genes,
                               cutoff=None, percentile=None,
                               library_size_normalize=True,
                               keep_cells='below',
                               sample_labels=None,
                               filter_per_sample=False):
    """Remove cells with total expression of a gene set above or below a certain threshold

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
    filter_per_sample : bool, optional (default: False)
        If True, filters separately for each unique sample label.
    """
    cell_sums = measure.gene_set_expression(
        data, genes,
        library_size_normalize=library_size_normalize)
    return filter_values(data, cell_sums,
                         cutoff=cutoff, percentile=percentile,
                         keep_cells=keep_cells,
                         sample_labels=sample_labels,
                         filter_per_sample=filter_per_sample)
