import numpy as np
import numbers
import pandas as pd
import warnings
try:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib import animation
except ImportError:
    pass

from .. import utils, select
from .utils import (_get_figure, _with_matplotlib, _is_color_array,
                    show, _in_ipynb, parse_fontsize, temp_fontsize)
from .tools import (create_colormap, create_normalize,
                    label_axis, generate_colorbar, generate_legend)


def _with_default(param, default):
    return param if param is not None else default


class _ScatterParams(object):

    def __init__(self, x, y, z=None, c=None, discrete=None,
                 cmap=None, cmap_scale=None, vmin=None,
                 vmax=None, s=None, legend=None, colorbar=None,
                 shuffle=True):
        self._x = utils.toarray(x).squeeze()
        self._y = utils.toarray(y).squeeze()
        self._z = utils.toarray(z).squeeze() if z is not None else None
        self._c = c
        self._discrete = discrete
        self._cmap = cmap
        self._cmap_scale = cmap_scale
        self._vmin = vmin
        self._vmax = vmax
        self._s = s
        self._legend = legend
        self._colorbar = colorbar
        self._labels = None
        self._c_discrete = None
        self.shuffle = shuffle
        self.check_size()
        self.check_c()
        self.check_discrete()
        self.check_legend()
        self.check_cmap()
        self.check_cmap_scale()
        self.check_vmin_vmax()

    @property
    def size(self):
        return len(self._x)

    @property
    def plot_idx(self):
        try:
            return self._plot_idx
        except AttributeError:
            if self.shuffle:
                self._plot_idx = np.random.permutation(self.size)
            else:
                self._plot_idx = np.arange(self.size)
            return self._plot_idx

    @property
    def x(self):
        return self._x[self.plot_idx]

    @property
    def y(self):
        return self._y[self.plot_idx]

    @property
    def z(self):
        return self._z[self.plot_idx] if self._z is not None else None

    @property
    def data(self):
        if self.z is not None:
            return [self.x, self.y, self.z]
        else:
            return [self.x, self.y]

    @property
    def _data(self):
        if self._z is not None:
            return [self._x, self._y, self._z]
        else:
            return [self._x, self._y]

    @property
    def s(self):
        if self._s is not None:
            return self._s
        else:
            return 200 / np.sqrt(self.size)

    def constant_c(self):
        return self._c is None or mpl.colors.is_color_like(self._c)

    def array_c(self):
        return _is_color_array(self._c)

    @property
    def discrete(self):
        if self._discrete is not None:
            return self._discrete
        else:
            if self.constant_c() or self.array_c():
                return None
            else:
                if isinstance(self._cmap, dict) or not \
                    np.all([isinstance(x, numbers.Number)
                            for x in self._c]):
                    # cmap dictionary or non-numeric values force discrete
                    return True
                else:
                    # guess based on number of unique elements
                    return len(np.unique(self._c)) <= 20

    @property
    def c_discrete(self):
        if self._c_discrete is None:
            if isinstance(self._cmap, dict):
                self._labels = np.array(list(self._cmap.keys()))
                self._c_discrete = np.zeros_like(self._c, dtype=int)
                for i, label in enumerate(self._labels):
                    self._c_discrete[self._c == label] = i
            else:
                self._c_discrete, self._labels = pd.factorize(
                    self._c, sort=True)
        return self._c_discrete

    @property
    def c(self):
        if self.constant_c() or self.array_c():
            return self._c
        elif self.discrete:
            return self.c_discrete[self.plot_idx]
        else:
            return self._c[self.plot_idx]

    @property
    def labels(self):
        if self.constant_c() or self.array_c():
            return None
        elif self.discrete:
            # make sure this exists
            self.c_discrete
            return self._labels
        else:
            return None

    @property
    def legend(self):
        if self._legend is not None:
            return self._legend
        else:
            if self.constant_c() or self.array_c():
                return False
            else:
                return True

    @property
    def vmin(self):
        if self._vmin is not None:
            return self._vmin
        else:
            if self.constant_c() or self.array_c() or self.discrete:
                return None
            else:
                return np.min(self.c)

    @property
    def vmax(self):
        if self._vmax is not None:
            return self._vmax
        else:
            if self.constant_c() or self.array_c() or self.discrete:
                return None
            else:
                return np.max(self.c)

    def list_cmap(self):
        return hasattr(self._cmap, '__len__') and \
            not isinstance(self._cmap, (str, dict))

    @property
    def cmap(self):
        if self._cmap is not None:
            if isinstance(self._cmap, dict):
                return mpl.colors.ListedColormap(
                    [mpl.colors.to_rgba(self._cmap[l]) for l in self.labels])
            elif self.list_cmap():
                return create_colormap(self._cmap)
            else:
                return self._cmap
        else:
            if self.constant_c() or self.array_c():
                return None
            elif self.discrete:
                n_unique_colors = len(np.unique(self.c))
                if n_unique_colors <= 10:
                    return mpl.colors.ListedColormap(
                        mpl.cm.tab10.colors[:n_unique_colors])
                else:
                    return 'tab20'
            else:
                return 'inferno'

    @property
    def cmap_scale(self):
        if self._cmap_scale is not None:
            return self._cmap_scale
        else:
            if self.discrete or not self.legend:
                return None
            else:
                return 'linear'

    @property
    def norm(self):
        if self.cmap_scale is not None and self.cmap_scale != 'linear':
            return create_normalize(self.vmin, self.vmax, self.cmap_scale)
        else:
            return None

    @property
    def extend(self):
        if self.legend and not self.discrete:
            # migrate this to _ScatterParams
            extend_min = np.min(self.c) < self.vmin
            extend_max = np.max(self.c) > self.vmax
            if extend_min:
                return 'both' if extend_max else 'min'
            else:
                return 'max' if extend_max else 'neither'
        else:
            return None

    @property
    def subplot_kw(self):
        if self.z is not None:
            return {'projection': '3d'}
        else:
            return {}

    def check_vmin_vmax(self):
        if self.constant_c():
            if self._vmin is not None or self._vmax is not None:
                warnings.warn(
                    "Cannot set `vmin` or `vmax` with constant `c={}`. "
                    "Setting `vmin = vmax = None`.".format(self.c),
                    UserWarning)
            self._vmin = None
            self._vmax = None
        elif self.discrete:
            if self._vmin is not None or self._vmax is not None:
                warnings.warn(
                    "Cannot set `vmin` or `vmax` with discrete data. "
                    "Setting to `None`.",
                    UserWarning)
            self._vmin = None
            self._vmax = None

    def check_legend(self):
        # legend and colorbar are synonyms
        if self._colorbar is not None:
            if self._legend is not None and self._legend != self._colorbar:
                raise ValueError(
                    "Received conflicting values for synonyms "
                    "`legend={}` and `colorbar={}`".format(
                        self._legend, self._colorbar))
            else:
                self._legend = self._colorbar
        if self._legend:
            if self.array_c():
                warnings.warn(
                    "`c` is a color array and cannot be used to create a "
                    "legend. To interpret these values as labels instead, "
                    "provide a `cmap` dictionary with label-color pairs.",
                    UserWarning)
                self._legend = False
            elif self.constant_c():
                warnings.warn(
                    "Cannot create a legend with constant `c={}`".format(
                        self.c), UserWarning)
                self._legend = False

    def check_size(self):
        # check data shape
        for d in self._data:
            if len(d) != self.size:
                raise ValueError(
                    "Expected all axes of data to have the same length"
                    ". Got {}".format([len(d) for d in self._data]))

    def check_c(self):
        if not self.constant_c() or self.array_c():
            self._c = utils.toarray(self._c).squeeze()
            if not len(self._c) == self.size:
                raise ValueError("Expected c of length {} or 1. Got {}".format(
                    self.size, len(self._c)))

    def check_discrete(self):
        if self._discrete is False:
            if not np.all([isinstance(x, numbers.Number) for x in self._c]):
                raise ValueError(
                    "Cannot treat non-numeric data as continuous.")

    def check_cmap(self):
        if isinstance(self._cmap, dict):
            # dictionary cmap
            if self.constant_c() or self.array_c():
                raise ValueError("Expected list-like `c` with dictionary cmap."
                                 " Got {}".format(type(self._c)))
            elif not self.discrete:
                raise ValueError("Cannot use dictionary cmap with "
                                 "continuous data.")
            elif np.any([l not in self._cmap for l in np.unique(self._c)]):
                missing = set(np.unique(self._c).tolist()
                              ).difference(self._cmap.keys())
                raise ValueError(
                    "Dictionary cmap requires a color "
                    "for every unique entry in `c`. "
                    "Missing colors for [{}]".format(
                        ", ".join([str(l) for l in missing])))
        elif self.list_cmap():
            if self.constant_c() or self.array_c():
                raise ValueError("Expected list-like `c` with list cmap. "
                                 "Got {}".format(type(self._c)))

    def check_cmap_scale(self):
        if self._cmap_scale is not None and self._cmap_scale != 'linear':
            if self.array_c():
                warnings.warn(
                    "Cannot use non-linear `cmap_scale` with "
                    "`c` as a color array.",
                    UserWarning)
                self._cmap_scale = 'linear'
            elif self.constant_c():
                warnings.warn(
                    "Cannot use non-linear `cmap_scale` with constant "
                    "`c={}`.".format(self._c),
                    UserWarning)
                self._cmap_scale = 'linear'
            elif self.discrete:
                warnings.warn(
                    "Cannot use non-linear `cmap_scale` with discrete data.",
                    UserWarning)
                self._cmap_scale = 'linear'


@_with_matplotlib
def scatter(x, y, z=None,
            c=None, cmap=None, cmap_scale='linear', s=None, discrete=None,
            ax=None,
            legend=None, colorbar=None,
            shuffle=True,
            figsize=None,
            ticks=True,
            xticks=None,
            yticks=None,
            zticks=None,
            ticklabels=True,
            xticklabels=None,
            yticklabels=None,
            zticklabels=None,
            label_prefix=None,
            xlabel=None,
            ylabel=None,
            zlabel=None,
            title=None,
            fontsize=None,
            legend_title=None,
            legend_loc='best',
            legend_anchor=None,
            vmin=None, vmax=None,
            elev=None, azim=None,
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
    cmap_scale : {'linear', 'log', 'symlog', 'sqrt'} or `matplotlib.colors.Normalize`, optional (default: 'linear')
        Colormap normalization scale. For advanced use, see
        <https://matplotlib.org/users/colormapnorms.html>
    s : float, optional (default: None)
        Point size. If `None`, set to 200 / sqrt(n_samples)
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
    {x,y,z}ticks : True, False, or list-like (default: None)
        If set, overrides `ticks`
    ticklabels : True, False, or list-like (default: True)
        If True, keeps default axis tick labels. If False, removes axis tick labels.
        If a list, sets custom axis tick labels
    {x,y,z}ticklabels : True, False, or list-like (default: None)
        If set, overrides `ticklabels`
    label_prefix : str or None (default: None)
        Prefix for all axis labels. Axes will be labelled `label_prefix`1,
        `label_prefix`2, etc. Can be overriden by setting `xlabel`,
        `ylabel`, and `zlabel`.
    {x,y,z}label : str or None (default : None)
        Axis labels. Overrides the automatic label given by
        label_prefix. If None and label_prefix is None, no label is set.
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
    elev : int, optional (default: None)
        Elevation angle of viewpoint from horizontal for 3D plots, in degrees
    azim : int, optional (default: None)
        Azimuth angle in x-y plane of viewpoint for 3D plots, in degrees
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
    >>> import scprep
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> data = np.random.normal(0, 1, [200, 3])
    >>> # Continuous color vector
    >>> colors = data[:, 0]
    >>> scprep.plot.scatter(x=data[:, 0], y=data[:, 1], c=colors)
    >>> # Discrete color vector with custom colormap
    >>> colors = np.random.choice(['a','b'], data.shape[0], replace=True)
    >>> data[colors == 'a'] += 5
    >>> scprep.plot.scatter(x=data[:, 0], y=data[:, 1], z=data[:, 2],
    ...                     c=colors, cmap={'a' : [1,0,0,1], 'b' : 'xkcd:sky blue'})
    """
    with temp_fontsize(fontsize):
        params = _ScatterParams(
            x, y, z, c=c, discrete=discrete,
            cmap=cmap, cmap_scale=cmap_scale,
            vmin=vmin, vmax=vmax, s=s,
            legend=legend, colorbar=colorbar)

        fig, ax, show_fig = _get_figure(
            ax, figsize, subplot_kw=params.subplot_kw)

        # plot!
        sc = ax.scatter(
            *(params.data),
            c=params.c, cmap=params.cmap, norm=params.norm, s=params.s,
            vmin=params.vmin, vmax=params.vmax, **plot_kwargs)

        # automatic axis labels
        if label_prefix is not None:
            if xlabel is None:
                xlabel = label_prefix + "1"
            if ylabel is None:
                ylabel = label_prefix + "2"
            if zlabel is None:
                zlabel = label_prefix + "3"

        # label axes
        label_axis(ax.xaxis, _with_default(xticks, ticks),
                   _with_default(xticklabels, ticklabels), xlabel)
        label_axis(ax.yaxis, _with_default(yticks, ticks),
                   _with_default(yticklabels, ticklabels), ylabel)
        if z is not None:
            label_axis(ax.zaxis, _with_default(zticks, ticks),
                       _with_default(zticklabels, ticklabels), zlabel)

        if title is not None:
            ax.set_title(title, fontsize=parse_fontsize(None, 'xx-large'))

        # generate legend
        if params.legend:
            if params.discrete:
                generate_legend({params.labels[i]: sc.cmap(sc.norm(i))
                                 for i in range(len(params.labels))}, ax=ax,
                                loc=legend_loc, bbox_to_anchor=legend_anchor,
                                title=legend_title)
            else:
                generate_colorbar(params.cmap, ax=ax,
                                  vmin=params.vmin, vmax=params.vmax,
                                  title=legend_title, extend=params.extend,
                                  scale=sc.norm)

        # set viewpoint
        if z is not None:
            ax.view_init(elev=elev, azim=azim)

        # save and show
        if filename is not None:
            fig.savefig(filename, dpi=dpi)
        if show_fig:
            show(fig)
    return ax


@_with_matplotlib
def scatter2d(data,
              c=None, cmap=None, cmap_scale='linear', s=None, discrete=None,
              ax=None, legend=None, colorbar=None,
              shuffle=True, figsize=None,
              ticks=True,
              xticks=None,
              yticks=None,
              ticklabels=True,
              xticklabels=None,
              yticklabels=None,
              label_prefix=None,
              xlabel=None,
              ylabel=None,
              title=None,
              fontsize=None,
              legend_title=None,
              legend_loc='best',
              legend_anchor=None,
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
    cmap : `matplotlib` colormap, str, dict, list or None, optional (default: None)
        matplotlib colormap. If None, uses `tab20` for discrete data and
        `inferno` for continuous data. If a list, expects one color for every
        unique value in `c`, otherwise interpolates between given colors for
        continuous data. If a dictionary, expects one key
        for every unique value in `c`, where values are valid matplotlib colors
        (hsv, rbg, rgba, or named colors)
    cmap_scale : {'linear', 'log', 'symlog', 'sqrt'} or `matplotlib.colors.Normalize`, optional (default: 'linear')
        Colormap normalization scale. For advanced use, see
        <https://matplotlib.org/users/colormapnorms.html>
    s : float, optional (default: None)
        Point size. If `None`, set to 200 / sqrt(n_samples)
    discrete : bool or None, optional (default: None)
        If True, the legend is categorical. If False, the legend is a colorbar.
        If None, discreteness is detected automatically. Data containing
        non-numeric `c` is always discrete, and numeric data with 20 or less
        unique values is discrete.
    ax : `matplotlib.Axes` or None, optional (default: None)
        axis on which to plot. If None, an axis is created
    legend : bool, optional (default: None)
        States whether or not to create a legend. If data is continuous,
        the legend is a colorbar. If `None`, a legend is created where possible.
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
    label_prefix : str or None (default: None)
        Prefix for all axis labels. Axes will be labelled `label_prefix`1,
        `label_prefix`2, etc. Can be overriden by setting `xlabel`,
        `ylabel`, and `zlabel`.
    {x,y}label : str or None (default : None)
        Axis labels. Overrides the automatic label given by
        label_prefix. If None and label_prefix is None, no label is set.
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

    Examples
    --------
    >>> import scprep
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> data = np.random.normal(0, 1, [200, 2])
    >>> # Continuous color vector
    >>> colors = data[:, 0]
    >>> scprep.plot.scatter2d(data, c=colors)
    >>> # Discrete color vector with custom colormap
    >>> colors = np.random.choice(['a','b'], data.shape[0], replace=True)
    >>> data[colors == 'a'] += 10
    >>> scprep.plot.scatter2d(data, c=colors, cmap={'a' : [1,0,0,1], 'b' : 'xkcd:sky blue'})
    """
    return scatter(x=select.select_cols(data, idx=0),
                   y=select.select_cols(data, idx=1),
                   c=c, cmap=cmap, cmap_scale=cmap_scale, s=s, discrete=discrete,
                   ax=ax, legend=legend, colorbar=colorbar,
                   shuffle=shuffle, figsize=figsize,
                   ticks=ticks,
                   xticks=xticks,
                   yticks=yticks,
                   ticklabels=ticklabels,
                   xticklabels=xticklabels,
                   yticklabels=yticklabels,
                   label_prefix=label_prefix,
                   xlabel=xlabel,
                   ylabel=ylabel,
                   title=title,
                   fontsize=fontsize,
                   legend_title=legend_title,
                   legend_loc=legend_loc,
                   legend_anchor=legend_anchor,
                   filename=filename,
                   dpi=dpi,
                   **plot_kwargs)


@_with_matplotlib
def scatter3d(data,
              c=None, cmap=None, cmap_scale='linear', s=None, discrete=None,
              ax=None, legend=None, colorbar=None,
              shuffle=True,
              figsize=None,
              ticks=True,
              xticks=None,
              yticks=None,
              zticks=None,
              ticklabels=True,
              xticklabels=None,
              yticklabels=None,
              zticklabels=None,
              label_prefix=None,
              xlabel=None,
              ylabel=None,
              zlabel=None,
              title=None,
              fontsize=None,
              legend_title=None,
              legend_loc='best',
              legend_anchor=None,
              elev=None, azim=None,
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
    cmap : `matplotlib` colormap, str, dict, list or None, optional (default: None)
        matplotlib colormap. If None, uses `tab20` for discrete data and
        `inferno` for continuous data. If a list, expects one color for every
        unique value in `c`, otherwise interpolates between given colors for
        continuous data. If a dictionary, expects one key
        for every unique value in `c`, where values are valid matplotlib colors
        (hsv, rbg, rgba, or named colors)
    cmap_scale : {'linear', 'log', 'symlog', 'sqrt'} or `matplotlib.colors.Normalize`, optional (default: 'linear')
        Colormap normalization scale. For advanced use, see
        <https://matplotlib.org/users/colormapnorms.html>
    s : float, optional (default: None)
        Point size. If `None`, set to 200 / sqrt(n_samples)
    discrete : bool or None, optional (default: None)
        If True, the legend is categorical. If False, the legend is a colorbar.
        If None, discreteness is detected automatically. Data containing
        non-numeric `c` is always discrete, and numeric data with 20 or less
        unique values is discrete.
    ax : `matplotlib.Axes` or None, optional (default: None)
        axis on which to plot. If None, an axis is created
    legend : bool, optional (default: None)
        States whether or not to create a legend. If data is continuous,
        the legend is a colorbar. If `None`, a legend is created where possible.
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
    {x,y,z}ticks : True, False, or list-like (default: None)
        If set, overrides `ticks`
    ticklabels : True, False, or list-like (default: True)
        If True, keeps default axis tick labels. If False, removes axis tick labels.
        If a list, sets custom axis tick labels
    {x,y,z}ticklabels : True, False, or list-like (default: None)
        If set, overrides `ticklabels`
    label_prefix : str or None (default: None)
        Prefix for all axis labels. Axes will be labelled `label_prefix`1,
        `label_prefix`2, etc. Can be overriden by setting `xlabel`,
        `ylabel`, and `zlabel`.
    {x,y,z}label : str or None (default : None)
        Axis labels. Overrides the automatic label given by
        label_prefix. If None and label_prefix is None, no label is set.
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
    elev : int, optional (default: None)
        Elevation angle of viewpoint from horizontal, in degrees
    azim : int, optional (default: None)
        Azimuth angle in x-y plane of viewpoint
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
    >>> import scprep
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> data = np.random.normal(0, 1, [200, 3])
    >>> # Continuous color vector
    >>> colors = data[:, 0]
    >>> scprep.plot.scatter3d(data, c=colors)
    >>> # Discrete color vector with custom colormap
    >>> colors = np.random.choice(['a','b'], data.shape[0], replace=True)
    >>> data[colors == 'a'] += 5
    >>> scprep.plot.scatter3d(data, c=colors, cmap={'a' : [1,0,0,1], 'b' : 'xkcd:sky blue'})
    """
    return scatter(x=select.select_cols(data, idx=0),
                   y=select.select_cols(data, idx=1),
                   z=select.select_cols(data, idx=2),
                   c=c, cmap=cmap, cmap_scale=cmap_scale, s=s, discrete=discrete,
                   ax=ax, legend=legend, colorbar=colorbar,
                   shuffle=shuffle, figsize=figsize,
                   ticks=ticks,
                   xticks=xticks,
                   yticks=yticks,
                   zticks=zticks,
                   ticklabels=ticklabels,
                   xticklabels=xticklabels,
                   yticklabels=yticklabels,
                   zticklabels=zticklabels,
                   label_prefix=label_prefix,
                   xlabel=xlabel,
                   ylabel=ylabel,
                   zlabel=zlabel,
                   title=title,
                   fontsize=fontsize,
                   legend_title=legend_title,
                   legend_loc=legend_loc,
                   legend_anchor=legend_anchor,
                   elev=elev,
                   azim=azim,
                   filename=filename,
                   dpi=dpi,
                   **plot_kwargs)


@_with_matplotlib
def rotate_scatter3d(data,
                     filename=None,
                     rotation_speed=30,
                     fps=10,
                     ax=None,
                     figsize=None,
                     ipython_html="jshtml",
                     dpi=None,
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
    dpi : int or None, optional (default: None)
        The resolution in dots per inch. If None it will default to the value
        savefig.dpi in the matplotlibrc file. If 'figure' it will set the dpi
        to be the value of the figure. Only used if filename is not None.
    **kwargs : keyword arguments
        See :~func:`phate.plot.scatter3d`.

    Returns
    -------
    ani : `matplotlib.animation.FuncAnimation`
        animation object

    Examples
    --------
    >>> import scprep
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> data = np.random.normal(0, 1, [200, 3])
    >>> # Continuous color vector
    >>> colors = data[:, 0]
    >>> scprep.plot.rotate_scatter3d(data, c=colors, filename="animation.gif")
    >>> # Discrete color vector with custom colormap
    >>> colors = np.random.choice(['a','b'], data.shape[0], replace=True)
    >>> data[colors == 'a'] += 5
    >>> scprep.plot.rotate_scatter3d(data, c=colors, cmap={'a' : [1,0,0,1], 'b' : 'xkcd:sky blue'}, filename="animation.mp4")
    """
    if _in_ipynb():
        # credit to
        # http://tiao.io/posts/notebooks/save-matplotlib-animations-as-gifs/
        mpl.rc('animation', html=ipython_html)

    if filename is not None:
        if filename.endswith(".gif"):
            writer = 'imagemagick'
        elif filename.endswith(".mp4"):
            writer = "ffmpeg"
        else:
            raise ValueError(
                "filename must end in .gif or .mp4. Got {}".format(filename))

    fig, ax, show_fig = _get_figure(
        ax, figsize, subplot_kw={'projection': '3d'})

    degrees_per_frame = rotation_speed / fps
    frames = int(round(360 / degrees_per_frame))
    # fix rounding errors
    degrees_per_frame = 360 / frames
    interval = 1000 * degrees_per_frame / rotation_speed

    scatter3d(data, ax=ax, **kwargs)

    azim = ax.azim

    def init():
        return ax

    def animate(i):
        ax.view_init(azim=azim + i * degrees_per_frame)
        return ax

    ani = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=range(frames), interval=interval, blit=False)

    if filename is not None:
        ani.save(filename, writer=writer, dpi=dpi)

    if _in_ipynb():
        # credit to https://stackoverflow.com/a/45573903/3996580
        plt.close(fig)
    elif show_fig:
        show(fig)

    return ani
