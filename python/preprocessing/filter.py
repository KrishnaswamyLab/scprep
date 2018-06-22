# author: Scott Gigante <scott.gigante@yale.edu>
# (C) 2018 Krishnaswamy Lab GPLv2

import pandas as pd
import numpy as np
from scipy import sparse
import warnings
import numbers
try:
    import matplotlib.pyplot as plt
except ImportError:
    pass


def _select_cols(data, idx):
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
    return data


def _select_rows(data, idx):
    if isinstance(data, pd.DataFrame):
        data = data.loc[idx]
    else:
        if isinstance(data, (sparse.coo_matrix,
                             sparse.bsr_matrix,
                             sparse.dia_matrix)):
            data = data.tocsr()
        data = data[idx, :]
    return data


def remove_empty_genes(data):
    gene_sums = np.array(data.sum(axis=0)).reshape(-1)
    keep_genes_idx = gene_sums > 0
    data = _select_cols(data, keep_genes_idx)
    return data


def remove_rare_genes(data, cutoff=0, min_cells=10):
    gene_sums = np.array((data > cutoff).sum(axis=0)).reshape(-1)
    keep_genes_idx = gene_sums > min_cells
    data = _select_cols(data, keep_genes_idx)
    return data


def remove_empty_cells(data):
    cell_sums = library_size(data)
    keep_cells_idx = cell_sums > 0
    data = _select_rows(data, keep_cells_idx)
    return data


def library_size(data):
    if isinstance(data, pd.SparseDataFrame):
        # densifies matrix if you take the sum
        cell_sums = pd.Series(
            np.array(data.to_coo().sum(axis=1)).reshape(-1),
            index=data.index)
    else:
        cell_sums = data.sum(axis=1)
    if isinstance(cell_sums, np.matrix):
        cell_sums = np.array(cell_sums).reshape(-1)
    return cell_sums


def plot_library_size(data, bins=30, cutoff=None, log=True):
    try:
        plt
    except NameError:
        raise ImportError(
            "matplotlib not found. "
            "Please install it with e.g. `pip install --user matplotlib`")
    cell_sums = library_size(data)
    if log:
        bins = np.logspace(np.log10(max(np.min(cell_sums), 1)),
                           np.log10(np.max(cell_sums)),
                           bins)
    plt.hist(cell_sums, bins=bins)
    if log:
        plt.xscale('log')
        plt.yscale('log')
    if cutoff is not None:
        plt.axvline(cutoff, color='red')
    plt.show(block=False)


def filter_library_size(data, cutoff=2000):
    cell_sums = library_size(data)
    keep_cells_idx = cell_sums > cutoff
    data = _select_rows(data, keep_cells_idx)
    return data


def gene_set_expression(data, genes):
    gene_data = _select_cols(data, genes)
    return library_size(gene_data)


def _get_percentile_cutoff(data, cutoff, percentile, required=False):
    if percentile is not None:
        if cutoff is not None:
            warnings.warn(
                "Only one of `cutoff` and `percentile` should be given.",
                UserWarning)
        if percentile < 1:
            warnings.warn(
                "`percentile` expects values between 0 and 100. "
                "Got {}. Did you mean {}?".format(percentile,
                                                  percentile * 100),
                UserWarning)
        cutoff = np.percentile(np.array(data).reshape(-1), percentile)
    elif cutoff is None and required:
        raise ValueError(
            "One of either `cutoff` or `percentile` must be given.")
    return cutoff


def plot_gene_set_expression(data, genes, bins=100,
                             cutoff=None, percentile=None):
    """
    Parameters
    ----------
    genes : list-like, dtype=`str` or `int`
        Column names or indices of genes to be summed and showed
    cutoff : float (default: None)
        Absolute value at which to draw a cutoff line.
        Overridden by percentile.
    percentile : int (Default: None)
        Integer between 0 and 100.
        Percentile at which to draw a cutoff line. Overrides cutoff.
    """
    try:
        plt
    except NameError:
        print("matplotlib not found. "
              "Please install it with e.g. `pip install --user matplotlib`")
    cell_sums = gene_set_expression(data, genes)
    cutoff = _get_percentile_cutoff(
        cell_sums, cutoff, percentile, required=False)
    plt.hist(cell_sums, bins=bins)
    if cutoff is not None:
        plt.vline(cutoff, color='red')
    plt.show(block=False)


def filter_gene_set_expression(data, genes,
                               cutoff=None, percentile=None,
                               keep_cells='below'):
    """
    cutoff : float (default: None)
        Absolute value at which to draw a cutoff line.
        Overridden by percentile.
    percentile : int (Default: None)
        Integer between 0 and 100.
        Percentile at which to draw a cutoff line. Overrides cutoff.
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
    data = _select_rows(data, keep_cells_idx)
    return data
