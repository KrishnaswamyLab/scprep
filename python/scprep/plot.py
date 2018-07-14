import numpy as np
from decorator import decorator
try:
    import matplotlib.pyplot as plt
except ImportError:
    pass

from .measure import library_size, gene_set_expression, _get_percentile_cutoff


@decorator
def _with_matplotlib(fun, *args, **kwargs):
    try:
        plt
    except NameError:
        raise ImportError(
            "matplotlib not found. "
            "Please install it with e.g. `pip install --user matplotlib`")
    return fun(*args, **kwargs)


@_with_matplotlib
def histogram(data,
              bins=100, log=True,
              cutoff=None, percentile=None,
              ax=None, figsize=None):
    """Plot a histogram.

    Parameters
    ----------
    data : array-like, shape=[n_samples]
        Input data
    bins : int, optional (default: 100)
        Number of bins to draw in the histogram
    log : bool, or {'x', 'y'}, optional (default: True)
        If True, plot both axes on a log scale. If 'x' or 'y',
        only plot the given axis on a log scale. If False,
        plot both axes on a linear scale.
    cutoff : float or `None`, optional (default: `None`)
        Absolute cutoff at which to draw a vertical line.
        Only one of `cutoff` and `percentile` may be given.
    percentile : float or `None`, optional (default: `None`)
        Percentile between 0 and 100 at which to draw a vertical line.
        Only one of `cutoff` and `percentile` may be given.
    ax : `matplotlib.Axes` or None, optional (default: None)
        Axis to plot on. If None, a new axis will be created.
    figsize : tuple or None, optional (default: None)
        If not None, sets the figure size (width, height)
    """
    if ax is not None:
        fig, ax = plt.subplots(figsize=figsize)
    if log:
        bins = np.logspace(np.log10(max(np.min(data), 1)),
                           np.log10(np.max(data)),
                           bins)
    plt.hist(data, bins=bins)

    if log == 'x' or log is True:
        plt.xscale('log')
    if log == 'y' or log is True:
        plt.yscale('log')

    cutoff = _get_percentile_cutoff(
        data, cutoff, percentile, required=False)
    if cutoff is not None:
        plt.axvline(cutoff, color='red')
    plt.show(block=False)


@_with_matplotlib
def plot_library_size(data,
                      bins=100, log=True,
                      cutoff=None, percentile=None,
                      ax=None, figsize=None):
    """Plot the library size histogram.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    bins : int, optional (default: 100)
        Number of bins to draw in the histogram
    log : bool, or {'x', 'y'}, optional (default: True)
        If True, plot both axes on a log scale. If 'x' or 'y',
        only plot the given axis on a log scale. If False,
        plot both axes on a linear scale.
    cutoff : float or `None`, optional (default: `None`)
        Absolute cutoff at which to draw a vertical line.
        Only one of `cutoff` and `percentile` may be given.
    percentile : float or `None`, optional (default: `None`)
        Percentile between 0 and 100 at which to draw a vertical line.
        Only one of `cutoff` and `percentile` may be given.
    ax : `matplotlib.Axes` or None, optional (default: None)
        Axis to plot on. If None, a new axis will be created.
    figsize : tuple or None, optional (default: None)
        If not None, sets the figure size (width, height)
    """
    histogram(library_size(data),
              cutoff=cutoff, percentile=percentile,
              bins=bins, log=log, ax=ax, figsize=figsize)


@_with_matplotlib
def plot_gene_set_expression(data, genes,
                             bins=100, log=False,
                             cutoff=None, percentile=None,
                             ax=None, figsize=None):
    """Plot the hsitogram of the expression of a gene set.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    genes : list-like, dtype=`str` or `int`
        Integer column indices or string gene names included in gene set
    bins : int, optional (default: 100)
        Number of bins to draw in the histogram
    log : bool, or {'x', 'y'}, optional (default: True)
        If True, plot both axes on a log scale. If 'x' or 'y',
        only plot the given axis on a log scale. If False,
        plot both axes on a linear scale.
    cutoff : float or `None`, optional (default: `None`)
        Absolute cutoff at which to draw a vertical line.
        Only one of `cutoff` and `percentile` may be given.
    percentile : float or `None`, optional (default: `None`)
        Percentile between 0 and 100 at which to draw a vertical line.
        Only one of `cutoff` and `percentile` may be given.
    ax : `matplotlib.Axes` or None, optional (default: None)
        Axis to plot on. If None, a new axis will be created.
    figsize : tuple or None, optional (default: None)
        If not None, sets the figure size (width, height)
    """
    histogram(gene_set_expression(data, genes),
              cutoff=cutoff, percentile=percentile,
              bins=bins, log=log, ax=ax, figsize=figsize)
