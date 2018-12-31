import numpy as np
from decorator import decorator
import platform
import numbers
import pandas as pd
import warnings
try:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    pass

from . import measure, utils


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


def _get_figure(ax=None, figsize=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        show_fig = True
    else:
        try:
            fig = ax.get_figure()
        except AttributeError as e:
            if not isinstance(ax, mpl.axes.Axes):
                raise TypeError("Expected ax as a matplotlib.axes.Axes. "
                                "Got {}".format(type(ax)))
            else:
                raise e
        show_fig = False
    return fig, ax, show_fig


def _is_color_array(c):
    return np.all([mpl.colors.is_color_like(val) for val in c])


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
        plt.tight_layout()
        if platform.system() == "Windows":
            plt.show(block=True)
        else:
            fig.show()


@_with_matplotlib
def histogram(data,
              bins=100, log=True,
              cutoff=None, percentile=None,
              ax=None, figsize=None,
              xlabel=None,
              ylabel='Number of cells',
              **kwargs):
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
    [x,y]label : str, optional
        Labels to display on the x and y axis.
    **kwargs : additional arguments for `matplotlib.pyplot.hist`
    """
    fig, ax, show_fig = _get_figure(ax, figsize)
    if log == 'x' or log is True:
        bins = np.logspace(np.log10(max(np.min(data), 1)),
                           np.log10(np.max(data)),
                           bins)
    ax.hist(data, bins=bins, **kwargs)

    if log == 'x' or log is True:
        ax.set_xscale('log')
    if log == 'y' or log is True:
        ax.set_yscale('log')

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    cutoff = measure._get_percentile_cutoff(
        data, cutoff, percentile, required=False)
    if cutoff is not None:
        ax.axvline(cutoff, color='red')
    if show_fig:
        show(fig)


@_with_matplotlib
def plot_library_size(data,
                      bins=100, log=True,
                      cutoff=None, percentile=None,
                      ax=None, figsize=None,
                      xlabel='Library size',
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
    [x,y]label : str, optional
        Labels to display on the x and y axis.
    **kwargs : additional arguments for `matplotlib.pyplot.hist`
    """
    histogram(measure.library_size(data),
              cutoff=cutoff, percentile=percentile,
              bins=bins, log=log, ax=ax, figsize=figsize,
              xlabel=xlabel, **kwargs)


@_with_matplotlib
def plot_gene_set_expression(data, genes,
                             bins=100, log=False,
                             cutoff=None, percentile=None,
                             library_size_normalize=True,
                             ax=None, figsize=None,
                             xlabel='Gene expression',
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
    [x,y]label : str, optional
        Labels to display on the x and y axis.
    **kwargs : additional arguments for `matplotlib.pyplot.hist`
    """
    histogram(measure.gene_set_expression(
        data, genes, library_size_normalize=library_size_normalize),
        cutoff=cutoff, percentile=percentile,
        bins=bins, log=log, ax=ax, figsize=figsize,
        xlabel=xlabel, **kwargs)


@_with_matplotlib
def scree_plot(singular_values, cumulative=False, ax=None, figsize=None,
               xlabel='Principal Component', ylabel='Explained Variance (%)',
               **kwargs):
    """Plot the explained variance of each principal component

    Parameters
    ----------
    singular_values : list-like, shape=[n_components]
        Singular values returned by `scprep.reduce.pca(data, return_singular_values=True)`
    cumulative : bool, optional (default=False)
        If True, plot the cumulative explained variance
    ax : `matplotlib.Axes` or None, optional (default: None)
        Axis to plot on. If None, a new axis will be created.
    figsize : tuple or None, optional (default: None)
        If not None, sets the figure size (width, height)
    [x,y]label : str, optional
        Labels to display on the x and y axis.
    **kwargs : additional arguments for `matplotlib.pyplot.plot`
    """
    explained_variance = singular_values ** 2
    explained_variance = explained_variance / explained_variance.sum()
    if cumulative:
        explained_variance = np.cumsum(explained_variance)
    fig, ax, show_fig = _get_figure(ax, figsize)
    ax.plot(np.arange(len(explained_variance)), explained_variance, **kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if show_fig:
        show(fig)


def _scatter_params(x, y, z=None, c=None, discrete=None,
                    cmap=None, legend=True):
    """Automatically select nice parameters for a scatter plot
    """
    if len(x) != len(y) or (z is not None and len(x) != len(z)):
        raise ValueError(
            "Expected all axis of data to have the same length"
            ". Got {}".format([
                len(d) for d in ([x, y, z] if z is not None else [x, y])]))
    if c is None or mpl.colors.is_color_like(c) or _is_color_array(c):
        # no legend
        labels = None
        if legend is True:
            if _is_color_array(c):
                warnings.warn(
                    "`c` is a color array and cannot be used to create a "
                    "legend. To interpret these values as labels instead, "
                    "provide a `cmap` dictionary with label-color pairs.",
                    UserWarning)
            else:
                warnings.warn(
                    "Cannot create a legend with `c={}`".format(c),
                    UserWarning)
        legend = False
    else:
        # label/value color array
        c = utils.toarray(c)
        if not len(c) == len(x):
            raise ValueError("Expected c of length {} or 1. Got {}".format(
                len(x), len(c)))
        # check discreteness
        if discrete is None:
            if isinstance(cmap, dict) or \
                    not np.all([isinstance(x, numbers.Number) for x in c]):
                # cmap dictionary or non-numeric values force discrete
                discrete = True
            else:
                # guess based on number of unique elements
                discrete = len(np.unique(c)) <= 20
        if discrete:
            c, labels = pd.factorize(c, sort=True)
            # choose cmap if not given
            if cmap is None and len(np.unique(c)) <= 10:
                cmap = mpl.colors.ListedColormap(
                    mpl.cm.tab10.colors[:len(np.unique(c))])
            elif cmap is None:
                cmap = 'tab20'
        else:
            if not np.all([isinstance(x, numbers.Number) for x in c]):
                raise ValueError(
                    "Cannot treat non-numeric data as continuous.")
            labels = None
            # choose cmap if not given
            if cmap is None:
                cmap = 'inferno'

    if isinstance(cmap, dict):
        # dictionary cmap
        if c is None or mpl.colors.is_color_like(c):
            raise ValueError("Expected list-like `c` with dictionary cmap. "
                             "Got {}".format(type(c)))
        elif not discrete:
            raise ValueError("Cannot use dictionary cmap with "
                             "continuous data.")
        elif np.any([l not in cmap for l in labels]):
            missing = set(labels).difference(cmap.keys())
            raise ValueError(
                "Dictionary cmap requires a color "
                "for every unique entry in `c`. "
                "Missing colors for [{}]".format(
                    ", ".join([str(l) for l in missing])))
        else:
            cmap = mpl.colors.ListedColormap(
                [mpl.colors.to_rgba(cmap[l]) for l in labels])
    elif hasattr(cmap, '__len__') and not isinstance(cmap, str):
        # list-like cmap
        if c is None or mpl.colors.is_color_like(c):
            raise ValueError("Expected list-like `c` with list cmap. "
                             "Got {}".format(type(c)))
        elif not discrete:
            cmap = mpl.colors.LinearSegmentedColormap(
                'scprep_custom_continuous_cmap',
                [mpl.colors.to_rgba(col) for col in cmap])
        elif len(cmap) != len(labels):
            raise ValueError(
                "List cmap requires a color "
                "for every unique entry in `c` ({}). "
                "Got {}".format(len(labels), len(cmap)))
        else:
            cmap = mpl.colors.ListedColormap(
                [mpl.colors.to_rgba(col) for col in cmap])

    if z is not None:
        subplot_kw = {'projection': '3d'}
    else:
        subplot_kw = {}
    return c, labels, discrete, cmap, legend, subplot_kw


@_with_matplotlib
def generate_legend(cmap, ax, title=None, marker='o', markersize=10,
                    loc='best', fontsize=14, title_size=14, max_rows=10,
                    **kwargs):
    '''Generates a legend on an axis using a dictionary cmap.'''
    handles = [mpl.lines.Line2D([], [], marker=marker, color=color,
                                label=label, markersize=markersize)
               for label, color in cmap.items()]
    legend = ax.legend(handles=handles, title=title,
                       loc=loc, fontsize=fontsize,
                       ncol=max(1, len(cmap) // max_rows), **kwargs)
    plt.setp(legend.get_title(), fontsize=title_size)
    return legend


@_with_matplotlib
def generate_colorbar(cmap, ax, vmin=0, vmax=1, title=None,
                      title_size=14, **kwargs):
    '''Generates a colorbar on an axis.'''
    try:
        plot_axis = ax[0]
    except TypeError:
        # not a list
        plot_axis = ax
    fig, _, _ = _get_figure(plot_axis)
    xmin, xmax = plot_axis.get_xlim()
    ymin, ymax = plot_axis.get_ylim()
    im = plot_axis.imshow(np.linspace(vmin, vmax, 10).reshape(-1, 1),
                          vmin=vmin, vmax=vmax, cmap=cmap,
                          aspect='auto', origin='lower',
                          extent=[xmin, xmax, ymin, ymax])
    im.remove()
    colorbar = fig.colorbar(im, ax=ax, **kwargs)
    colorbar.set_label(title, rotation=270)
    return colorbar


def _label_axis(axis, ticks=True, ticklabels=True, label=None):
    if not ticks:
        axis.set_ticks([])
    elif ticks is True:
        pass
    else:
        axis.set_ticks(ticks)
    if not ticklabels:
        axis.set_ticklabels([])
    elif ticklabels is True:
        pass
    else:
        axis.set_ticklabels(ticklabels)
    if label is not None:
        axis.set_label(label)


@_with_matplotlib
def scatter(x, y, z=None,
            c=None, cmap=None, s=1, discrete=None,
            ax=None, legend=True, figsize=None,
            xticks=True,
            yticks=True,
            zticks=True,
            xticklabels=True,
            yticklabels=True,
            zticklabels=True,
            label_prefix=None,
            xlabel=None,
            ylabel=None,
            zlabel=None,
            title=None,
            legend_title=None,
            legend_loc='best',
            filename=None,
            dpi=None,
            **plot_kwargs):
    """Create a scatter plot

    Builds upon `matplotlib.pyplot.scatter` with nice defaults
    and handles categorical colors / legends better. For easy access, use
    `scatter2d` or `scatter3d`.

    Parameters
    ----------
    x : list-like
        data for x axis
    y : list-like
        data for y axis
    z : list-like, optional (default: None)
        data for z axis
    c : list-like or None, optional (default: None)
        Color vector. Can be a single color value (RGB, RGBA, or named
        matplotlib colors), an array of these of length n_samples, or a list of
        discrete or continuous values of any data type. If `c` is not a single
        or list of matplotlib colors, the values in `c` will be used to
        populate the legend / colorbar with colors from `cmap`
    cmap : `matplotlib` colormap, str, dict or None, optional (default: None)
        matplotlib colormap. If None, uses `tab20` for discrete data and
        `inferno` for continuous data. If a dictionary, expects one key
        for every unique value in `c`, where values are valid matplotlib colors
        (hsv, rbg, rgba, or named colors)
    s : float, optional (default: 1)
        Point size.
    discrete : bool or None, optional (default: None)
        If True, the legend is categorical. If False, the legend is a colorbar.
        If None, discreteness is detected automatically. Data containing
        non-numeric `c` is always discrete, and numeric data with 20 or less
        unique values is discrete.
    ax : `matplotlib.Axes` or None, optional (default: None)
        axis on which to plot. If None, an axis is created
    legend : bool, optional (default: True)
        States whether or not to create a legend. If data is continuous,
        the legend is a colorbar.
    figsize : tuple, optional (default: None)
        Tuple of floats for creation of new `matplotlib` figure. Only used if
        `ax` is None.
    {x,y,z}ticks : True, False, or list-like (default: True)
        If True, keeps default axis ticks. If False, removes axis ticks.
        If a list, sets custom axis ticks
    {x,y,z}ticklabels : True, False, or list-like (default: True)
        If True, keeps default axis tick labels. If False, removes axis tick labels.
        If a list, sets custom axis tick labels
    label_prefix : str or None (default: None)
        Prefix for all axis labels. Axes will be labelled `label_prefix`1,
        `label_prefix`2, etc. Can be overriden by setting `xlabel`,
        `ylabel`, and `zlabel`.
    {x,y,z}label : str or None (default : None)
        Axis labels. Overrides the automatic label given by
        label_prefix. If None and label_prefix is None, no label is set.
    title : str or None (default: None)
        axis title. If None, no title is set.
    legend_title : str (default: None)
        title for the colorbar of legend
    legend_loc : int or string or pair of floats, default: 'best'
        Matplotlib legend location. Only used for discrete data.
        See <https://matplotlib.org/api/_as_gen/matplotlib.pyplot.legend.html>
        for details.
    filename : str or None (default: None)
        file to which the output is saved
    dpi : int or None, optional (default: None)
        The resolution in dots per inch. If None it will default to the value
        savefig.dpi in the matplotlibrc file. If 'figure' it will set the dpi
        to be the value of the figure. Only used if filename is not None.
    **plot_kwargs : keyword arguments
        Extra arguments passed to `matplotlib.pyplot.scatter`.

    Returns
    -------
    ax : `matplotlib.Axes`
        axis on which plot was drawn

    Examples
    --------
    >>> import phate
    >>> import matplotlib.pyplot as plt
    >>> ###
    >>> # Running PHATE
    >>> ###
    >>> tree_data, tree_clusters = phate.tree.gen_dla(n_dim=100, n_branch=20,
    ...                                               branch_length=100)
    >>> tree_data.shape
    (2000, 100)
    >>> phate_operator = phate.PHATE(k=5, a=20, t=150)
    >>> tree_phate = phate_operator.fit_transform(tree_data)
    >>> tree_phate.shape
    (2000, 2)
    >>> ###
    >>> # Plotting using phate.plot
    >>> ###
    >>> phate.plot.scatter2d(tree_phate, c=tree_clusters)
    >>> # You can also pass the PHATE operator instead of data
    >>> phate.plot.scatter2d(phate_operator, c=tree_clusters)
    >>> phate.plot.scatter3d(phate_operator, c=tree_clusters)
    >>> ###
    >>> # Using a cmap dictionary
    >>> ###
    >>> import numpy as np
    >>> X = np.random.normal(0,1,[1000,2])
    >>> c = np.random.choice(['a','b'], 1000, replace=True)
    >>> X[c=='a'] += 10
    >>> phate.plot.scatter2d(X, c=c, cmap={'a' : [1,0,0,1], 'b' : 'xkcd:sky blue'})
    """
    # convert to 1D numpy arrays
    x = utils.toarray(x).flatten()
    y = utils.toarray(y).flatten()
    if z is not None:
        z = utils.toarray(z).flatten()
        if ax is not None and not isinstance(ax, Axes3D):
            raise TypeError("Expected ax with projection='3d'. "
                            "Got 2D axis instead.")

    c, labels, discrete, cmap, legend, subplot_kw = _scatter_params(
        x, y, z, c, discrete, cmap, legend)

    fig, ax, show_fig = _get_figure(ax, figsize)

    # randomize point order
    plot_idx = np.random.permutation(len(x))
    if c is not None and not mpl.colors.is_color_like(c):
        c = c[plot_idx]
    # plot!
    sc = ax.scatter(
        *[d[plot_idx] for d in ([x, y] if z is None else [x, y, z])],
        c=c, cmap=cmap, s=s, **plot_kwargs)

    # automatic axis labels
    if label_prefix is not None:
        if xlabel is None:
            xlabel = label_prefix + "1" if xlabel is None else xlabel
        if ylabel is None:
            ylabel = label_prefix + "2"
        if zlabel is None:
            zlabel = label_prefix + "3"

    # label axes
    _label_axis(ax.xaxis, xticks, xticklabels, xlabel)
    _label_axis(ax.yaxis, yticks, yticklabels, ylabel)
    if z is not None:
        _label_axis(ax.zaxis, zticks, zticklabels, zlabel)

    # generate legend
    if legend:
        if discrete:
            generate_legend({labels[i]: sc.cmap(sc.norm(i))
                             for i in range(len(labels))}, ax=ax)
        else:
            generate_colorbar(cmap, ax, vmin=np.min(c), vmax=np.max(c),
                              title=legend_title)

    # save and show
    if filename is not None:
        fig.savefig(filename, dpi=dpi)
    if show_fig:
        show(fig)
    return ax


def scatter2d(data,
              c=None, cmap=None, s=1, discrete=None,
              ax=None, legend=True, figsize=None,
              xticks=True,
              yticks=True,
              xticklabels=True,
              yticklabels=True,
              label_prefix=None,
              xlabel=None,
              ylabel=None,
              title=None,
              legend_title=None,
              legend_loc='best',
              filename=None,
              dpi=None,
              **plot_kwargs):
    """Create a 2D scatter plot

    Builds upon `matplotlib.pyplot.scatter` with nice defaults
    and handles categorical colors / legends better.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data. Only the first two components will be used.
    c : list-like or None, optional (default: None)
        Color vector. Can be a single color value (RGB, RGBA, or named
        matplotlib colors), an array of these of length n_samples, or a list of
        discrete or continuous values of any data type. If `c` is not a single
        or list of matplotlib colors, the values in `c` will be used to
        populate the legend / colorbar with colors from `cmap`
    cmap : `matplotlib` colormap, str, dict or None, optional (default: None)
        matplotlib colormap. If None, uses `tab20` for discrete data and
        `inferno` for continuous data. If a dictionary, expects one key
        for every unique value in `c`, where values are valid matplotlib colors
        (hsv, rbg, rgba, or named colors)
    s : float, optional (default: 1)
        Point size.
    discrete : bool or None, optional (default: None)
        If True, the legend is categorical. If False, the legend is a colorbar.
        If None, discreteness is detected automatically. Data containing
        non-numeric `c` is always discrete, and numeric data with 20 or less
        unique values is discrete.
    ax : `matplotlib.Axes` or None, optional (default: None)
        axis on which to plot. If None, an axis is created
    legend : bool, optional (default: True)
        States whether or not to create a legend. If data is continuous,
        the legend is a colorbar.
    figsize : tuple, optional (default: None)
        Tuple of floats for creation of new `matplotlib` figure. Only used if
        `ax` is None.
    {x,y}ticks : True, False, or list-like (default: True)
        If True, keeps default axis ticks. If False, removes axis ticks.
        If a list, sets custom axis ticks
    {x,y}ticklabels : True, False, or list-like (default: True)
        If True, keeps default axis tick labels. If False, removes axis tick labels.
        If a list, sets custom axis tick labels
    label_prefix : str or None (default: None)
        Prefix for all axis labels. Axes will be labelled `label_prefix`1,
        `label_prefix`2, etc. Can be overriden by setting `xlabel`,
        `ylabel`, and `zlabel`.
    {x,y}label : str or None (default : None)
        Axis labels. Overrides the automatic label given by
        label_prefix. If None and label_prefix is None, no label is set.
    title : str or None (default: None)
        axis title. If None, no title is set.
    legend_title : str (default: None)
        title for the colorbar of legend
    legend_loc : int or string or pair of floats, default: 'best'
        Matplotlib legend location. Only used for discrete data.
        See <https://matplotlib.org/api/_as_gen/matplotlib.pyplot.legend.html>
        for details.
    filename : str or None (default: None)
        file to which the output is saved
    dpi : int or None, optional (default: None)
        The resolution in dots per inch. If None it will default to the value
        savefig.dpi in the matplotlibrc file. If 'figure' it will set the dpi
        to be the value of the figure. Only used if filename is not None.
    **plot_kwargs : keyword arguments
        Extra arguments passed to `matplotlib.pyplot.scatter`.

    Returns
    -------
    ax : `matplotlib.Axes`
        axis on which plot was drawn

    Examples
    --------
    >>> import phate
    >>> import matplotlib.pyplot as plt
    >>> ###
    >>> # Running PHATE
    >>> ###
    >>> tree_data, tree_clusters = phate.tree.gen_dla(n_dim=100, n_branch=20,
    ...                                               branch_length=100)
    >>> tree_data.shape
    (2000, 100)
    >>> phate_operator = phate.PHATE(k=5, a=20, t=150)
    >>> tree_phate = phate_operator.fit_transform(tree_data)
    >>> tree_phate.shape
    (2000, 2)
    >>> ###
    >>> # Plotting using phate.plot
    >>> ###
    >>> phate.plot.scatter2d(tree_phate, c=tree_clusters)
    >>> # You can also pass the PHATE operator instead of data
    >>> phate.plot.scatter2d(phate_operator, c=tree_clusters)
    >>> phate.plot.scatter3d(phate_operator, c=tree_clusters)
    >>> ###
    >>> # Using a cmap dictionary
    >>> ###
    >>> import numpy as np
    >>> X = np.random.normal(0,1,[1000,2])
    >>> c = np.random.choice(['a','b'], 1000, replace=True)
    >>> X[c=='a'] += 10
    >>> phate.plot.scatter2d(X, c=c, cmap={'a' : [1,0,0,1], 'b' : 'xkcd:sky blue'})
    """
    return scatter(x=utils.select_cols(data, 0),
                   y=utils.select_cols(data, 1),
                   c=c, cmap=cmap, s=s, discrete=discrete,
                   ax=ax, legend=legend, figsize=figsize,
                   xticks=xticks,
                   yticks=yticks,
                   xticklabels=xticklabels,
                   yticklabels=yticklabels,
                   label_prefix=label_prefix,
                   xlabel=xlabel,
                   ylabel=ylabel,
                   title=title,
                   legend_title=legend_title,
                   legend_loc=legend_loc,
                   filename=filename,
                   dpi=dpi,
                   **plot_kwargs)


def scatter3d(data,
              c=None, cmap=None, s=1, discrete=None,
              ax=None, legend=True, figsize=None,
              xticks=True,
              yticks=True,
              zticks=True,
              xticklabels=True,
              yticklabels=True,
              zticklabels=True,
              label_prefix=None,
              xlabel=None,
              ylabel=None,
              zlabel=None,
              title=None,
              legend_title=None,
              legend_loc='best',
              filename=None,
              dpi=None,
              **plot_kwargs):
    """Create a 3D scatter plot

    Builds upon `matplotlib.pyplot.scatter` with nice defaults
    and handles categorical colors / legends better.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data. Only the first two components will be used.
    c : list-like or None, optional (default: None)
        Color vector. Can be a single color value (RGB, RGBA, or named
        matplotlib colors), an array of these of length n_samples, or a list of
        discrete or continuous values of any data type. If `c` is not a single
        or list of matplotlib colors, the values in `c` will be used to
        populate the legend / colorbar with colors from `cmap`
    cmap : `matplotlib` colormap, str, dict or None, optional (default: None)
        matplotlib colormap. If None, uses `tab20` for discrete data and
        `inferno` for continuous data. If a dictionary, expects one key
        for every unique value in `c`, where values are valid matplotlib colors
        (hsv, rbg, rgba, or named colors)
    s : float, optional (default: 1)
        Point size.
    discrete : bool or None, optional (default: None)
        If True, the legend is categorical. If False, the legend is a colorbar.
        If None, discreteness is detected automatically. Data containing
        non-numeric `c` is always discrete, and numeric data with 20 or less
        unique values is discrete.
    ax : `matplotlib.Axes` or None, optional (default: None)
        axis on which to plot. If None, an axis is created
    legend : bool, optional (default: True)
        States whether or not to create a legend. If data is continuous,
        the legend is a colorbar.
    figsize : tuple, optional (default: None)
        Tuple of floats for creation of new `matplotlib` figure. Only used if
        `ax` is None.
    {x,y,z}ticks : True, False, or list-like (default: True)
        If True, keeps default axis ticks. If False, removes axis ticks.
        If a list, sets custom axis ticks
    {x,y,z}ticklabels : True, False, or list-like (default: True)
        If True, keeps default axis tick labels. If False, removes axis tick labels.
        If a list, sets custom axis tick labels
    label_prefix : str or None (default: None)
        Prefix for all axis labels. Axes will be labelled `label_prefix`1,
        `label_prefix`2, etc. Can be overriden by setting `xlabel`,
        `ylabel`, and `zlabel`.
    {x,y,z}label : str or None (default : None)
        Axis labels. Overrides the automatic label given by
        label_prefix. If None and label_prefix is None, no label is set.
    title : str or None (default: None)
        axis title. If None, no title is set.
    legend_title : str (default: None)
        title for the colorbar of legend
    legend_loc : int or string or pair of floats, default: 'best'
        Matplotlib legend location. Only used for discrete data.
        See <https://matplotlib.org/api/_as_gen/matplotlib.pyplot.legend.html>
        for details.
    filename : str or None (default: None)
        file to which the output is saved
    dpi : int or None, optional (default: None)
        The resolution in dots per inch. If None it will default to the value
        savefig.dpi in the matplotlibrc file. If 'figure' it will set the dpi
        to be the value of the figure. Only used if filename is not None.
    **plot_kwargs : keyword arguments
        Extra arguments passed to `matplotlib.pyplot.scatter`.

    Returns
    -------
    ax : `matplotlib.Axes`
        axis on which plot was drawn

    Examples
    --------
    >>> import phate
    >>> import matplotlib.pyplot as plt
    >>> ###
    >>> # Running PHATE
    >>> ###
    >>> tree_data, tree_clusters = phate.tree.gen_dla(n_dim=100, n_branch=20,
    ...                                               branch_length=100)
    >>> tree_data.shape
    (2000, 100)
    >>> phate_operator = phate.PHATE(k=5, a=20, t=150)
    >>> tree_phate = phate_operator.fit_transform(tree_data)
    >>> tree_phate.shape
    (2000, 2)
    >>> ###
    >>> # Plotting using phate.plot
    >>> ###
    >>> phate.plot.scatter2d(tree_phate, c=tree_clusters)
    >>> # You can also pass the PHATE operator instead of data
    >>> phate.plot.scatter2d(phate_operator, c=tree_clusters)
    >>> phate.plot.scatter3d(phate_operator, c=tree_clusters)
    >>> ###
    >>> # Using a cmap dictionary
    >>> ###
    >>> import numpy as np
    >>> X = np.random.normal(0,1,[1000,2])
    >>> c = np.random.choice(['a','b'], 1000, replace=True)
    >>> X[c=='a'] += 10
    >>> phate.plot.scatter2d(X, c=c, cmap={'a' : [1,0,0,1], 'b' : 'xkcd:sky blue'})
    """
    return scatter(x=utils.select_cols(data, 0),
                   y=utils.select_cols(data, 1),
                   z=utils.select_cols(data, 2),
                   c=c, cmap=cmap, s=s, discrete=discrete,
                   ax=ax, legend=legend, figsize=figsize,
                   xticks=xticks,
                   yticks=yticks,
                   zticks=zticks,
                   xticklabels=xticklabels,
                   yticklabels=yticklabels,
                   zticklabels=zticklabels,
                   label_prefix=label_prefix,
                   xlabel=xlabel,
                   ylabel=ylabel,
                   zlabel=zlabel,
                   title=title,
                   legend_title=legend_title,
                   legend_loc=legend_loc,
                   filename=filename,
                   dpi=dpi,
                   **plot_kwargs)


def rotate_scatter3d(data,
                     filename=None,
                     elev=30,
                     rotation_speed=30,
                     fps=10,
                     ax=None,
                     figsize=None,
                     ipython_html="jshtml",
                     **kwargs):
    """Create a rotating 3D scatter plot

    Builds upon `matplotlib.pyplot.scatter` with nice defaults
    and handles categorical colors / legends better.

    Parameters
    ----------
    data : array-like, `phate.PHATE` or `scanpy.AnnData`
        Input data. Only the first three dimensions are used.
    filename : str, optional (default: None)
        If not None, saves a .gif or .mp4 with the output
    elev : float, optional (default: 30)
        Elevation of viewpoint from horizontal, in degrees
    rotation_speed : float, optional (default: 30)
        Speed of axis rotation, in degrees per second
    fps : int, optional (default: 10)
        Frames per second. Increase this for a smoother animation
    ax : `matplotlib.Axes` or None, optional (default: None)
        axis on which to plot. If None, an axis is created
    figsize : tuple, optional (default: None)
        Tuple of floats for creation of new `matplotlib` figure. Only used if
        `ax` is None.
    ipython_html : {'html5', 'jshtml'}
        which html writer to use if using a Jupyter Notebook
    **kwargs : keyword arguments
        See :~func:`phate.plot.scatter3d`.

    Returns
    -------
    ani : `matplotlib.animation.FuncAnimation`
        animation object

    Examples
    --------
    >>> import phate
    >>> import matplotlib.pyplot as plt
    >>> tree_data, tree_clusters = phate.tree.gen_dla(n_dim=100, n_branch=20,
    ...                                               branch_length=100)
    >>> tree_data.shape
    (2000, 100)
    >>> phate_operator = phate.PHATE(n_components=3, k=5, a=20, t=150)
    >>> tree_phate = phate_operator.fit_transform(tree_data)
    >>> tree_phate.shape
    (2000, 2)
    >>> phate.plot.rotate_scatter3d(tree_phate, c=tree_clusters)
    """
    if in_ipynb():
        # credit to
        # http://tiao.io/posts/notebooks/save-matplotlib-animations-as-gifs/
        rc('animation', html=ipython_html)

    if filename is not None:
        if filename.endswith(".gif"):
            writer = 'imagemagick'
        elif filename.endswith(".mp4"):
            writer = "ffmpeg"
        else:
            raise ValueError(
                "filename must end in .gif or .mp4. Got {}".format(filename))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize,
                               subplot_kw={'projection': '3d'})
        show = True
    else:
        fig = ax.get_figure()
        show = False

    degrees_per_frame = rotation_speed / fps
    frames = int(round(360 / degrees_per_frame))
    # fix rounding errors
    degrees_per_frame = 360 / frames
    interval = 1000 * degrees_per_frame / rotation_speed

    scatter3d(data, ax=ax, **kwargs)

    def init():
        ax.view_init(azim=0, elev=elev)
        return ax

    def animate(i):
        ax.view_init(azim=i * degrees_per_frame, elev=elev)
        return ax

    ani = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=range(frames), interval=interval, blit=False)

    if filename is not None:
        ani.save(filename, writer=writer)

    if in_ipynb():
        # credit to https://stackoverflow.com/a/45573903/3996580
        plt.close()
    elif show:
        plt.tight_layout()
        fig.show()

    return ani
