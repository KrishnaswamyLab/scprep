# author: Scott Gigante <scott.gigante@yale.edu>
# (C) 2018 Krishnaswamy Lab GPLv2

import numpy as np
import pandas as pd
from scipy import sparse

import warnings
import numbers

from . import utils, measure, select


def remove_empty_genes(data, *extra_data):
    warnings.warn(
        "`scprep.filter.remove_empty_genes` is deprecated. "
        "Use `scprep.filter.filter_empty_genes` instead.",
        DeprecationWarning,
    )
    return filter_empty_genes(data, *extra_data)


def remove_rare_genes(data, *extra_data, cutoff=0, min_cells=5):
    warnings.warn(
        "`scprep.filter.remove_rare_genes` is deprecated. "
        "Use `scprep.filter.filter_rare_genes` instead.",
        DeprecationWarning,
    )
    return filter_rare_genes(data, *extra_data, cutoff=cutoff, min_cells=min_cells)


def remove_empty_cells(data, *extra_data, sample_labels=None):
    warnings.warn(
        "`scprep.filter.remove_empty_cells` is deprecated. "
        "Use `scprep.filter.filter_empty_cells` instead.",
        DeprecationWarning,
    )
    return filter_empty_cells(data, *extra_data, sample_labels=sample_labels)


def remove_duplicates(data, *extra_data, sample_labels=None):
    warnings.warn(
        "`scprep.filter.remove_duplicates` is deprecated. "
        "Use `scprep.filter.filter_duplicates` instead.",
        DeprecationWarning,
    )
    return filter_duplicates(data, *extra_data, sample_labels=sample_labels)


def filter_empty_genes(data, *extra_data):
    """Filter all genes with zero counts across all cells

    This is equivalent to `filter_rare_genes(data, cutoff=0, min_cells=1)`
    but should be faster.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    extra_data : array-like, shape=[any, n_features], optional
        Optional additional data objects from which to select the same genes

    Returns
    -------
    data : array-like, shape=[n_samples, m_features]
        Filtered output data, where m_features <= n_features
    extra_data : array-like, shape=[any, m_features]
        Filtered extra data, if passed.
    """
    gene_sums = np.array(utils.matrix_sum(data, axis=0)).reshape(-1)
    keep_genes_idx = gene_sums > 0
    data = select.select_cols(data, *extra_data, idx=keep_genes_idx)
    return data


def filter_rare_genes(data, *extra_data, cutoff=0, min_cells=5):
    """Filter all genes with negligible counts in all but a few cells

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    extra_data : array-like, shape=[any, n_features], optional
        Optional additional data objects from which to select the same rows
    cutoff : float, optional (default: 0)
        Number of counts above which expression is deemed non-negligible
    min_cells : int, optional (default: 5)
        Minimum number of cells above `cutoff` in order to retain a gene

    Returns
    -------
    data : array-like, shape=[n_samples, m_features]
        Filtered output data, where m_features <= n_features
    extra_data : array-like, shape=[any, m_features]
        Filtered extra data, if passed.
    """
    gene_sums = measure.gene_capture_count(data, cutoff=cutoff)
    keep_genes_idx = gene_sums >= min_cells
    data = select.select_cols(data, *extra_data, idx=keep_genes_idx)
    return data


def filter_empty_cells(data, *extra_data, sample_labels=None):
    """Remove all cells with zero library size

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    extra_data : array-like, shape=[n_samples, any], optional
        Optional additional data objects from which to select the same rows
    sample_labels : Deprecated

    Returns
    -------
    data : array-like, shape=[m_samples, n_features]
        Filtered output data, where m_samples <= n_samples
    extra_data : array-like, shape=[m_samples, any]
        Filtered extra data, if passed.
    """
    if sample_labels is not None:
        warnings.warn(
            "`sample_labels` is deprecated. "
            "Passing `sample_labels` as `extra_data`.",
            DeprecationWarning,
        )
        extra_data = list(extra_data) + [sample_labels]
    cell_sums = measure.library_size(data)
    keep_cells_idx = cell_sums > 0
    data = select.select_rows(data, *extra_data, idx=keep_cells_idx)
    return data


def filter_values(
    data,
    *extra_data,
    values=None,
    cutoff=None,
    percentile=None,
    keep_cells="above",
    return_values=False,
    sample_labels=None,
    filter_per_sample=None
):
    """Remove all cells with `values` above or below a certain threshold

    It is recommended to use :func:`~scprep.plot.histogram` to
    choose a cutoff prior to filtering.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    extra_data : array-like, shape=[n_samples, any], optional
        Optional additional data objects from which to select the same rows
    values : list-like, shape=[n_samples]
        Value upon which to filter
    cutoff : float or tuple of floats, optional (default: None)
        Value above or below which to retain cells. Only one of `cutoff`
        and `percentile` should be specified.
    percentile : int or tuple of ints, optional (Default: None)
        Percentile above or below which to retain cells.
        Must be an integer between 0 and 100. Only one of `cutoff`
        and `percentile` should be specified.
    keep_cells : {'above', 'below', 'between'} or None, optional (default: None)
        Keep cells above, below or between the cutoff.
        If None, defaults to 'above' when a single cutoff is given and
        'between' when two cutoffs are given.
    return_values : bool, optional (default: False)
        If True, also return the values corresponding to the retained cells
    sample_labels : Deprecated
    filter_per_sample : Deprecated

    Returns
    -------
    data : array-like, shape=[m_samples, n_features]
        Filtered output data, where m_samples <= n_samples
    filtered_values : list-like, shape=[m_samples]
        Values corresponding to retained samples,
        returned only if return_values is True
    extra_data : array-like, shape=[m_samples, any]
        Filtered extra data, if passed.
    """
    if sample_labels is not None:
        warnings.warn(
            "`sample_labels` is deprecated. "
            "Passing `sample_labels` as `extra_data`.",
            DeprecationWarning,
        )
        extra_data = list(extra_data) + [sample_labels]
    if filter_per_sample is not None:
        warnings.warn(
            "`filter_per_sample` is deprecated. " "Filtering as a single sample.",
            DeprecationWarning,
        )
    assert values is not None
    keep_cells_idx = utils._get_filter_idx(values, cutoff, percentile, keep_cells)
    if return_values:
        extra_data = [values] + list(extra_data)
    data = select.select_rows(data, *extra_data, idx=keep_cells_idx)
    return data


def filter_library_size(
    data,
    *extra_data,
    cutoff=None,
    percentile=None,
    keep_cells=None,
    return_library_size=False,
    sample_labels=None,
    filter_per_sample=None
):
    """Remove all cells with library size above or below a certain threshold

    It is recommended to use :func:`~scprep.plot.plot_library_size` to
    choose a cutoff prior to filtering.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    extra_data : array-like, shape=[n_samples, any], optional
        Optional additional data objects from which to select the same rows
    cutoff : float or tuple of floats, optional (default: None)
        Library size above or below which to retain a cell. Only one of `cutoff`
        and `percentile` should be specified.
    percentile : int or tuple of ints, optional (Default: None)
        Percentile above or below which to retain a cell.
        Must be an integer between 0 and 100. Only one of `cutoff`
        and `percentile` should be specified.
    keep_cells : {'above', 'below', 'between'} or None, optional (default: None)
        Keep cells above, below or between the cutoff.
        If None, defaults to 'above' when a single cutoff is given and
        'between' when two cutoffs are given.
    return_library_size : bool, optional (default: False)
        If True, also return the library sizes corresponding to the retained cells
    sample_labels : Deprecated
    filter_per_sample : Deprecated

    Returns
    -------
    data : array-like, shape=[m_samples, n_features]
        Filtered output data, where m_samples <= n_samples
    filtered_library_size : list-like, shape=[m_samples]
        Library sizes corresponding to retained samples,
        returned only if return_library_size is True
    extra_data : array-like, shape=[m_samples, any]
        Filtered extra data, if passed.
    """
    cell_sums = measure.library_size(data)
    return filter_values(
        data,
        *extra_data,
        values=cell_sums,
        cutoff=cutoff,
        percentile=percentile,
        keep_cells=keep_cells,
        return_values=return_library_size,
        sample_labels=sample_labels,
        filter_per_sample=filter_per_sample,
    )


def filter_gene_set_expression(
    data,
    *extra_data,
    genes=None,
    starts_with=None,
    ends_with=None,
    exact_word=None,
    regex=None,
    cutoff=None,
    percentile=None,
    library_size_normalize=False,
    keep_cells=None,
    return_expression=False,
    sample_labels=None,
    filter_per_sample=None
):
    """Remove cells with total expression of a gene set above or below a certain threshold

    It is recommended to use :func:`~scprep.plot.plot_gene_set_expression` to
    choose a cutoff prior to filtering.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    extra_data : array-like, shape=[n_samples, any], optional
        Optional additional data objects from which to select the same rows
    genes : list-like, optional (default: None)
        Integer column indices or string gene names included in gene set
    starts_with : str or None, optional (default: None)
        If not None, select genes that start with this prefix
    ends_with : str or None, optional (default: None)
        If not None, select genes that end with this suffix
    exact_word : str, list-like or None, optional (default: None)
        If not None, select genes that contain this exact word.
    regex : str or None, optional (default: None)
        If not None, select genes that match this regular expression
    cutoff : float or tuple of floats, optional (default: None)
        Expression value above or below which to remove cells. Only one of `cutoff`
        and `percentile` should be specified.
    percentile : int or tuple of ints, optional (Default: None)
        Percentile above or below which to retain a cell.
        Must be an integer between 0 and 100. Only one of `cutoff`
        and `percentile` should be specified.
    library_size_normalize : bool, optional (default: False)
        Divide gene set expression by library size
    keep_cells : {'above', 'below', 'between'} or None, optional (default: None)
        Keep cells above or below the cutoff. If None, defaults to
        'below' for one cutoff and 'between' for two.
    return_expression : bool, optional (default: False)
        If True, also return the values corresponding to the retained cells
    sample_labels : Deprecated
    filter_per_sample : Deprecated

    Returns
    -------
    data : array-like, shape=[m_samples, n_features]
        Filtered output data, where m_samples <= n_samples
    filtered_expression : list-like, shape=[m_samples]
        Gene set expression corresponding to retained samples,
        returned only if return_expression is True
    extra_data : array-like, shape=[m_samples, any]
        Filtered extra data, if passed.
    """
    if keep_cells is None:
        if isinstance(cutoff, numbers.Number) or isinstance(percentile, numbers.Number):
            keep_cells = "below"
    cell_sums = measure.gene_set_expression(
        data,
        genes=genes,
        starts_with=starts_with,
        ends_with=ends_with,
        exact_word=exact_word,
        regex=regex,
        library_size_normalize=library_size_normalize,
    )
    return filter_values(
        data,
        *extra_data,
        values=cell_sums,
        cutoff=cutoff,
        percentile=percentile,
        keep_cells=keep_cells,
        return_values=return_expression,
        sample_labels=sample_labels,
        filter_per_sample=filter_per_sample,
    )


def _find_unique_cells(data):
    """Identify unique cells

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data

    Returns
    -------
    unique_idx : np.ndarray
        Sorted array of unique element indices
    """
    if utils.is_SparseDataFrame(data):
        unique_idx = _find_unique_cells(data.to_coo())
    elif utils.is_sparse_dataframe(data):
        unique_idx = _find_unique_cells(data.sparse.to_coo())
    elif isinstance(data, pd.DataFrame):
        unique_idx = ~data.duplicated()
    elif isinstance(data, np.ndarray):
        _, unique_idx = np.unique(data, axis=0, return_index=True)
        unique_idx = np.sort(unique_idx)
    elif sparse.issparse(data):
        _, unique_data = np.unique(data.tolil().data, return_index=True)
        _, unique_index = np.unique(data.tolil().rows, return_index=True)
        unique_idx = np.sort(list(set(unique_index).union(set(unique_data))))
    return unique_idx


def filter_duplicates(data, *extra_data, sample_labels=None):
    """Filter all duplicate cells

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    extra_data : array-like, shape=[n_samples, any], optional
        Optional additional data objects from which to select the same rows
    sample_labels : Deprecated

    Returns
    -------
    data : array-like, shape=[m_samples, n_features]
        Filtered output data, where m_samples <= n_samples
    extra_data : array-like, shape=[m_samples, any]
        Filtered extra data, if passed.
    """
    if sample_labels is not None:
        warnings.warn(
            "`sample_labels` is deprecated. "
            "Passing `sample_labels` as `extra_data`.",
            DeprecationWarning,
        )
        extra_data = list(extra_data) + [sample_labels]
    unique_idx = _find_unique_cells(data)
    data = select.select_rows(data, *extra_data, idx=unique_idx)
    return data
