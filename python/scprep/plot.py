import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    pass

from .measure import library_size, gene_set_expression, _get_percentile_cutoff


def with_matplotlib(fun):
    def wrapped_fun(*args, **kwargs):
        try:
            plt
        except NameError:
            raise ImportError(
                "matplotlib not found. "
                "Please install it with e.g. `pip install --user matplotlib`")
        return fun(*args, **kwargs)
    return wrapped_fun


@with_matplotlib
def plot_library_size(data, bins=30, cutoff=None, log=True):
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


@with_matplotlib
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
    cell_sums = gene_set_expression(data, genes)
    cutoff = _get_percentile_cutoff(
        cell_sums, cutoff, percentile, required=False)
    plt.hist(cell_sums, bins=bins)
    if cutoff is not None:
        plt.axvline(cutoff, color='red')
    plt.show(block=False)
