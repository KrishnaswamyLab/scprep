import numpy as np
import pandas as pd

from .. import measure, utils
from .utils import (_get_figure, show,
                    temp_fontsize, parse_fontsize)
from .tools import label_axis


@utils._with_pkg(pkg="matplotlib", min_version=3)
def histogram(data,
              bins=100, log=False,
              cutoff=None, percentile=None,
              ax=None, figsize=None,
              xlabel=None,
              ylabel='Number of cells',
              title=None,
              fontsize=None,
              **kwargs):
    """Plot a histogram.

    Parameters
    ----------
    data : array-like, shape=[n_samples]
        Input data. Multiple datasets may be given as a list of array-likes.
    bins : int, optional (default: 100)
        Number of bins to draw in the histogram
    log : bool, or {'x', 'y'}, optional (default: False)
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
    [x,y]label : str, optional
        Labels to display on the x and y axis.
    title : str or None, optional (default: None)
        Axis title.
    fontsize : float or None (default: None)
        Base font size.
    **kwargs : additional arguments for `matplotlib.pyplot.hist`

    Returns
    -------
    ax : `matplotlib.Axes`
        axis on which plot was drawn
    """
    with temp_fontsize(fontsize):
        fig, ax, show_fig = _get_figure(ax, figsize)
        data = utils.toarray(data).squeeze()
        if len(data.shape) > 1 or data.dtype.type is np.object_:
            # top level must be list
            data = [d for d in data]
            xmin = np.min([np.min(d) for d in data])
            xmax = np.max([np.max(d) for d in data])
        else:
            xmin = np.min(data)
            xmax = np.max(data)
        if log == 'x' or log is True:
            bins = np.logspace(np.log10(max(xmin, 1)),
                               np.log10(xmax),
                               bins)
        ax.hist(data, bins=bins, **kwargs)

        if log == 'x' or log is True:
            ax.set_xscale('log')
        if log == 'y' or log is True:
            ax.set_yscale('log')

        label_axis(ax.xaxis, label=xlabel)
        label_axis(ax.yaxis, label=ylabel)

        if title is not None:
            ax.set_title(title, fontsize=parse_fontsize(None, 'xx-large'))

        cutoff = measure._get_percentile_cutoff(
            data, cutoff, percentile, required=False)
        if cutoff is not None:
            ax.axvline(cutoff, color='red')
        if show_fig:
            show(fig)
    return ax


@utils._with_pkg(pkg="matplotlib", min_version=3)
def plot_library_size(data,
                      bins=100, log=True,
                      cutoff=None, percentile=None,
                      ax=None, figsize=None,
                      xlabel='Library size',
                      title=None,
                      fontsize=None,
                      **kwargs):
    """Plot the library size histogram.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data. Multiple datasets may be given as a list of array-likes.
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
    [x,y]label : str, optional
        Labels to display on the x and y axis.
    title : str or None, optional (default: None)
        Axis title.
    fontsize : float or None (default: None)
        Base font size.
    **kwargs : additional arguments for `matplotlib.pyplot.hist`

    Returns
    -------
    ax : `matplotlib.Axes`
        axis on which plot was drawn
    """
    data = utils.toarray(data)
    if len(data.shape) > 2 or data.dtype.type is np.object_:
        # top level must be list
        libsize = [measure.library_size(d)
                   for d in data]
    else:
        libsize = measure.library_size(data)
    return histogram(libsize,
                     cutoff=cutoff, percentile=percentile,
                     bins=bins, log=log, ax=ax, figsize=figsize,
                     xlabel=xlabel, title=title, fontsize=fontsize, **kwargs)


@utils._with_pkg(pkg="matplotlib", min_version=3)
def plot_gene_set_expression(data, genes=None,
                             starts_with=None, ends_with=None, regex=None,
                             bins=100, log=False,
                             cutoff=None, percentile=None,
                             library_size_normalize=False,
                             ax=None, figsize=None,
                             xlabel='Gene expression',
                             title=None,
                             fontsize=None,
                             **kwargs):
    """Plot the histogram of the expression of a gene set.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data. Multiple datasets may be given as a list of array-likes.
    genes : list-like, optional (default: None)
        Integer column indices or string gene names included in gene set
    starts_with : str or None, optional (default: None)
        If not None, select genes that start with this prefix
    ends_with : str or None, optional (default: None)
        If not None, select genes that end with this suffix
    regex : str or None, optional (default: None)
        If not None, select genes that match this regular expression
    bins : int, optional (default: 100)
        Number of bins to draw in the histogram
    log : bool, or {'x', 'y'}, optional (default: False)
        If True, plot both axes on a log scale. If 'x' or 'y',
        only plot the given axis on a log scale. If False,
        plot both axes on a linear scale.
    cutoff : float or `None`, optional (default: `None`)
        Absolute cutoff at which to draw a vertical line.
        Only one of `cutoff` and `percentile` may be given.
    percentile : float or `None`, optional (default: `None`)
        Percentile between 0 and 100 at which to draw a vertical line.
        Only one of `cutoff` and `percentile` may be given.
    library_size_normalize : bool, optional (default: False)
        Divide gene set expression by library size
    ax : `matplotlib.Axes` or None, optional (default: None)
        Axis to plot on. If None, a new axis will be created.
    figsize : tuple or None, optional (default: None)
        If not None, sets the figure size (width, height)
    [x,y]label : str, optional
        Labels to display on the x and y axis.
    title : str or None, optional (default: None)
        Axis title.
    fontsize : float or None (default: None)
        Base font size.
    **kwargs : additional arguments for `matplotlib.pyplot.hist`

    Returns
    -------
    ax : `matplotlib.Axes`
        axis on which plot was drawn
    """
    if not isinstance(data, pd.DataFrame) and isinstance(data[0], pd.DataFrame):
        # top level must be list
        expression = [measure.gene_set_expression(
            d, genes=genes,
            starts_with=starts_with, ends_with=ends_with, regex=regex,
            library_size_normalize=library_size_normalize)
            for d in data]
    else:
        expression = measure.gene_set_expression(
            data, genes=genes,
            starts_with=starts_with, ends_with=ends_with, regex=regex,
            library_size_normalize=library_size_normalize)
    return histogram(expression,
                     cutoff=cutoff, percentile=percentile,
                     bins=bins, log=log, ax=ax, figsize=figsize,
                     xlabel=xlabel, title=title, fontsize=fontsize, **kwargs)
