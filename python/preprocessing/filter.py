# author: Scott Gigante <scott.gigante@yale.edu>
# (C) 2018 Krishnaswamy Lab GPLv2

import numpy as np

from .utils import select_rows, select_cols
from .measure import library_size, gene_set_expression, _get_percentile_cutoff


def remove_empty_genes(data):
    gene_sums = np.array(data.sum(axis=0)).reshape(-1)
    keep_genes_idx = gene_sums > 0
    data = select_cols(data, keep_genes_idx)
    return data


def remove_rare_genes(data, cutoff=0, min_cells=10):
    gene_sums = np.array((data > cutoff).sum(axis=0)).reshape(-1)
    keep_genes_idx = gene_sums > min_cells
    data = select_cols(data, keep_genes_idx)
    return data


def remove_empty_cells(data):
    cell_sums = library_size(data)
    keep_cells_idx = cell_sums > 0
    data = select_rows(data, keep_cells_idx)
    return data


def filter_library_size(data, cutoff=2000):
    cell_sums = library_size(data)
    keep_cells_idx = cell_sums > cutoff
    data = select_rows(data, keep_cells_idx)
    return data


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
    data = select_rows(data, keep_cells_idx)
    return data
