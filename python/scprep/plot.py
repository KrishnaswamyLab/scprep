import numpy as np
from decorator import decorator
import os
try:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
except ImportError:
    pass

from . import measure


@decorator
def _with_matplotlib(fun, *args, **kwargs):
    try:
        plt
    except NameError:
        raise ImportError(
            "matplotlib not found. "
            "Please install it with e.g. `pip install --user matplotlib`")
    return fun(*args, **kwargs)


def _mpl_is_gui_backend():
    backend = mpl.get_backend()
    if backend in ['module://ipykernel.pylab.backend_inline', 'agg']:
        return False
    else:
        return True


@_with_matplotlib
def show(fig):
    """Show a matplotlib Figure correctly, regardless of platform

    If running a Jupyter notebook, we avoid running `fig.show`. If running
    in Windows, it is necessary to run `plt.show` rather than `fig.show`.

    Parameters
    ----------
    fig : matplotlib.Figure
        Figure to show
    """
    if _mpl_is_gui_backend():
        if os.platform == "Windows":
            plt.show(block=True)
        else:
            fig.show()


@_with_matplotlib
def histogram(data,
              bins=100, log=True,
              cutoff=None, percentile=None,
              ax=None, figsize=None, **kwargs):
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
    **kwargs : additional arguments for `matplotlib.pyplot.hist`
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        try:
            fig = ax.get_figure()
        except AttributeError as e:
            if not isinstance(ax, mpl.axes.Axes):
                raise TypeError("Expected ax as a matplotlib.axes.Axes. "
                                "Got {}".format(type(ax)))
            else:
                raise e
    if log == 'x' or log is True:
        bins = np.logspace(np.log10(max(np.min(data), 1)),
                           np.log10(np.max(data)),
                           bins)
    ax.hist(data, bins=bins, **kwargs)

    if log == 'x' or log is True:
        ax.set_xscale('log')
    if log == 'y' or log is True:
        ax.set_yscale('log')

    cutoff = measure._get_percentile_cutoff(
        data, cutoff, percentile, required=False)
    if cutoff is not None:
        ax.axvline(cutoff, color='red')
    if _mpl_is_gui_backend():
        fig.show()


@_with_matplotlib
def plot_library_size(data,
                      bins=100, log=True,
                      cutoff=None, percentile=None,
                      ax=None, figsize=None,
                      **kwargs):
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
    **kwargs : additional arguments for `matplotlib.pyplot.hist`
    """
    histogram(measure.library_size(data),
              cutoff=cutoff, percentile=percentile,
              bins=bins, log=log, ax=ax, figsize=figsize, **kwargs)


@_with_matplotlib
def plot_gene_set_expression(data, genes,
                             bins=100, log=False,
                             cutoff=None, percentile=None,
                             library_size_normalize=True,
                             ax=None, figsize=None,
                             **kwargs):
    """Plot the histogram of the expression of a gene set.

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
    library_size_normalize : bool, optional (default: True)
        Divide gene set expression by library size
    ax : `matplotlib.Axes` or None, optional (default: None)
        Axis to plot on. If None, a new axis will be created.
    figsize : tuple or None, optional (default: None)
        If not None, sets the figure size (width, height)
    **kwargs : additional arguments for `matplotlib.pyplot.hist`
    """
    histogram(measure.gene_set_expression(
        data, genes, library_size_normalize=library_size_normalize),
        cutoff=cutoff, percentile=percentile,
        bins=bins, log=log, ax=ax, figsize=figsize, **kwargs)
