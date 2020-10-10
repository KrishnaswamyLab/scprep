import numpy as np
import pandas as pd

from .. import utils
from .utils import _get_figure, show, temp_fontsize, parse_fontsize, _with_default
from .tools import label_axis, generate_colorbar, generate_legend

from .scatter import _ScatterParams


class _JitterParams(_ScatterParams):
    @property
    def x_labels(self):
        try:
            return self._x_labels
        except AttributeError:
            self._x_coords, self._x_labels = pd.factorize(self.x_array, sort=True)
            return self._x_labels

    @property
    def x_coords(self):
        # check this exists
        self.x_labels
        return self._x_coords[self.plot_idx]


@utils._with_pkg(pkg="matplotlib", min_version=3)
def jitter(
    labels,
    values,
    sigma=0.1,
    c=None,
    cmap=None,
    cmap_scale="linear",
    s=None,
    mask=None,
    plot_means=True,
    means_s=100,
    means_c="lightgrey",
    discrete=None,
    ax=None,
    legend=None,
    colorbar=None,
    shuffle=True,
    figsize=None,
    ticks=True,
    xticks=None,
    yticks=None,
    ticklabels=True,
    xticklabels=None,
    yticklabels=None,
    xlabel=None,
    ylabel=None,
    title=None,
    fontsize=None,
    legend_title=None,
    legend_loc="best",
    legend_anchor=None,
    vmin=None,
    vmax=None,
    filename=None,
    dpi=None,
    **plot_kwargs
):
    """Creates a 2D scatterplot showing the distribution of `values` for points
    that have associated `labels`.

    Parameters
    ----------
    labels : array-like, shape=[n_cells]
        Class labels associated with each point.
    values : array-like, shape=[n_cells]
        Values associated with each cell
    sigma : float, optinoal, default: 0.1
        Adjusts the amount of jitter.
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
    cmap_scale : {'linear', 'log', 'symlog', 'sqrt'} or `matplotlib.colors.Normalize`, optional (default: 'linear')
        Colormap normalization scale. For advanced use, see
        <https://matplotlib.org/users/colormapnorms.html>
    s : float, optional (default: None)
        Point size. If `None`, set to 200 / sqrt(n_samples)
    mask : list-like, optional (default: None)
        boolean mask to hide data points
    plot_means : bool, optional (default: True)
        If True, plot the mean value for each label.
    means_s : float, optional (default: 100)
        Point size for mean values.
    means_c : string, list-like or matplotlib color, optional (default: 'lightgrey')
        Point color(s) for mean values.
    discrete : bool or None, optional (default: None)
        If True, the legend is categorical. If False, the legend is a colorbar.
        If None, discreteness is detected automatically. Data containing
        non-numeric `c` is always discrete, and numeric data with 20 or less
        unique values is discrete.
    ax : `matplotlib.Axes` or None, optional (default: None)
        axis on which to plot. If None, an axis is created
    legend : bool, optional (default: None)
        States whether or not to create a legend. If data is continuous,
        the legend is a colorbar. If `None`, a legend is created where possible
    colorbar : bool, optional (default: None)
        Synonym for `legend`
    shuffle : bool, optional (default: True)
        If True. shuffles the order of points on the plot.
    figsize : tuple, optional (default: None)
        Tuple of floats for creation of new `matplotlib` figure. Only used if
        `ax` is None.
    ticks : True, False, or list-like (default: True)
        If True, keeps default axis ticks. If False, removes axis ticks.
        If a list, sets custom axis ticks
    {x,y}ticks : True, False, or list-like (default: None)
        If set, overrides `ticks`
    ticklabels : True, False, or list-like (default: True)
        If True, keeps default axis tick labels. If False, removes axis tick labels.
        If a list, sets custom axis tick labels
    {x,y}ticklabels : True, False, or list-like (default: None)
        If set, overrides `ticklabels`
    {x,y}label : str or None (default : None)
        Axis labels. If None, no label is set.
    title : str or None (default: None)
        axis title. If None, no title is set.
    fontsize : float or None (default: None)
        Base font size.
    legend_title : str (default: None)
        title for the colorbar of legend
    legend_loc : int or string or pair of floats, default: 'best'
        Matplotlib legend location. Only used for discrete data.
        See <https://matplotlib.org/api/_as_gen/matplotlib.pyplot.legend.html>
        for details.
    legend_anchor : `BboxBase`, 2-tuple, or 4-tuple
        Box that is used to position the legend in conjunction with loc.
        See <https://matplotlib.org/api/_as_gen/matplotlib.pyplot.legend.html>
        for details.
    vmin, vmax : float, optional (default: None)
        Range of values to use as the range for the colormap.
        Only used if data is continuous
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

    """
    with temp_fontsize(fontsize):
        params = _JitterParams(
            labels,
            values,
            c=c,
            discrete=discrete,
            cmap=cmap,
            cmap_scale=cmap_scale,
            vmin=vmin,
            vmax=vmax,
            s=s,
            mask=mask,
            legend=legend,
            colorbar=colorbar,
            xlabel=xlabel,
            ylabel=ylabel,
        )

        fig, ax, show_fig = _get_figure(ax, figsize, subplot_kw=params.subplot_kw)

        # Plotting cells
        sc = ax.scatter(
            params.x_coords + np.random.normal(0, sigma, params.size)[params.plot_idx],
            params.y,
            c=params.c,
            cmap=params.cmap,
            norm=params.norm,
            s=params.s,
            vmin=params.vmin,
            vmax=params.vmax,
            **plot_kwargs,
        )

        # Plotting means
        if plot_means:
            ax.scatter(
                np.arange(len(params.x_labels)),
                [
                    np.nanmean(params.y[params.x_coords == i])
                    for i in range(len(params.x_labels))
                ],
                c=means_c,
                edgecolors="black",
                lw=1.5,
                marker="o",
                zorder=3,
                s=means_s,
            )

        # Plotting vetical lines
        for i in range(len(params.x_labels)):
            ax.axvline(i, c="k", lw=0.1, zorder=0)

        # x axis labels
        xticks = _with_default(xticks, ticks)
        if xticks is True:
            xticks = np.arange(len(params.x_labels))
            xticklabels = _with_default(xticklabels, ticklabels)
            if xticklabels is True:
                xticklabels = params.x_labels

        # label axes
        label_axis(ax.xaxis, xticks, xticklabels, params.xlabel)
        label_axis(
            ax.yaxis,
            _with_default(yticks, ticks),
            _with_default(yticklabels, ticklabels),
            params.ylabel,
        )

        # manually set x limits
        xmin = np.min(params.x_coords)
        xmax = np.max(params.x_coords)
        ax.set_xlim(xmin - 0.5, xmax + 0.5)

        if title is not None:
            ax.set_title(title, fontsize=parse_fontsize(None, "xx-large"))

        # generate legend
        if params.legend:
            if params.discrete:
                generate_legend(
                    {
                        params.labels[i]: sc.cmap(sc.norm(i))
                        for i in range(len(params.labels))
                    },
                    ax=ax,
                    loc=legend_loc,
                    bbox_to_anchor=legend_anchor,
                    title=legend_title,
                )
            else:
                generate_colorbar(
                    params.cmap,
                    ax=ax,
                    vmin=params.vmin,
                    vmax=params.vmax,
                    title=legend_title,
                    extend=params.extend,
                    scale=sc.norm,
                )

        # save and show
        if show_fig:
            show(fig)
        if filename is not None:
            fig.savefig(filename, dpi=dpi)
    return ax
