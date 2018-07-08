import numpy as np
from decorator import decorator
try:
    import matplotlib.pyplot as plt
except ImportError:
    pass

from .measure import library_size, gene_set_expression, _get_percentile_cutoff


def with_matplotlib(fun):
    @decorator(fun)
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
def plot_library_size(data, cutoff=None, bins=100, log=True):
    """Plot the library size histogram.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    cutoff : float or `None`, optional (default: `None`)
        Absolute cutoff at which to draw a vertical line.
    bins : int, optional (default: 100)
        Number of bins to draw in the histogram
    log : bool, or {'x', 'y'}, optional (default: True)
        If True, plot both axes on a log scale. If 'x' or 'y',
        only plot the given axis on a log scale. If False,
        plot both axes on a linear scale.
    """
    libsize = library_size(data)
    if log:
        bins = np.logspace(np.log10(max(np.min(libsize), 1)),
                           np.log10(np.max(libsize)),
                           bins)
    plt.hist(libsize, bins=bins)
    if log == 'x' or log is True:
        plt.xscale('log')
    if log == 'y' or log is True:
        plt.yscale('log')
    if cutoff is not None:
        plt.axvline(cutoff, color='red')
    plt.show(block=False)


@with_matplotlib
def plot_gene_set_expression(data, genes, bins=100,
                             cutoff=None, percentile=None):
    """Plot the hsitogram of the expression of a gene set.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    genes : list-like, dtype=`str` or `int`
        Integer column indices or string gene names included in gene set
    cutoff : float or `None`, optional (default: `None`)
        Absolute cutoff at which to draw a vertical line.
        Only one of `cutoff` and `percentile` may be given.
    percentile : float or `None`, optional (default: `None`)
        Percentile between 0 and 100 at which to draw a vertical line.
        Only one of `cutoff` and `percentile` may be given.
    bins : int, optional (default: 100)
        Number of bins to draw in the histogram
    log : bool, or {'x', 'y'}, optional (default: True)
        If True, plot both axes on a log scale. If 'x' or 'y',
        only plot the given axis on a log scale. If False,
        plot both axes on a linear scale.
    """
    cell_sums = gene_set_expression(data, genes)
    cutoff = _get_percentile_cutoff(
        cell_sums, cutoff, percentile, required=False)
    plt.hist(cell_sums, bins=bins)
    if cutoff is not None:
        plt.axvline(cutoff, color='red')
    plt.show(block=False)
