import numpy as np
from . import tools

from .._lazyload import matplotlib as mpl

plt = mpl.pyplot


def tab10_continuous(n_colors=10, n_step=200, reverse=False):
    """Creates a discrete colormap with continuous gradation

    This colormap uses the colors in `matplotlib`'s `tab10` colormap
    as the base for a discrete colormap, but additionally creates
    a lightness gradient for each discrete hue.
    For an example, see the scprep scatterplot gallery.

    Parameters
    ----------
    n_colors : int, optional (default: 10)
        The number of discrete colors
    n_step : int, optional (default: 20)
        The number of unique gradations in each section of the colormap
    reverse : bool, optional (default: False)
        If True, gradations go from dark to light instead of light to dark

    Returns
    -------
    cmap : `matplotlib.colors.ListedColormap`
    """
    if n_colors < 1 or n_colors > 10:
        raise ValueError("Expected 0 < n_colors <= 10. Got {}".format(n_colors))
    if n_step < 2:
        raise ValueError("Expected n_step >= 2. Got {}".format(n_step))
    base_color_idx = np.repeat(np.arange(n_colors), 2) * 2
    if reverse:
        offset = np.tile([0, 1], n_colors)
    else:
        offset = np.tile([1, 0], n_colors)
    color_idx = base_color_idx + offset
    full_cmap = tools.create_colormap(np.array(plt.cm.tab20.colors)[color_idx])
    linspace = np.linspace(0, 1 / (n_colors * 2 - 1), n_step)
    restricted_cmap = mpl.colors.ListedColormap(
        full_cmap(
            np.concatenate(
                [linspace + 2 * i / (n_colors * 2 - 1) for i in range(n_colors)]
            )
        )
    )
    return restricted_cmap


def tab30():
    """A discrete colormap with 30 unique colors

    This colormap combines `matplotlib`'s `tab20b` and `tab20c` colormaps,
    removing the lightest color of each hue.

    Returns
    -------
    cmap : `matplotlib.colors.ListedColormap`
    """
    colors = np.vstack([mpl.cm.tab20c.colors, mpl.cm.tab20b.colors])
    select_idx = np.repeat(np.arange(10), 3) * 4 + np.tile(np.arange(3), 10)
    return mpl.colors.ListedColormap(colors[select_idx])


def tab40():
    """A discrete colormap with 40 unique colors

    This colormap combines `matplotlib`'s `tab20b` and `tab20c` colormaps

    Returns
    -------
    cmap : `matplotlib.colors.ListedColormap`
    """
    colors = np.vstack([mpl.cm.tab20c.colors, mpl.cm.tab20b.colors])
    return mpl.colors.ListedColormap(colors)


def tab(n=10):
    """A discrete colormap with an arbitrary number of colors

    This colormap chooses the best of the following, in order:
    - `plt.cm.tab10`
    - `plt.cm.tab20`
    - `scprep.plot.colors.tab30`
    - `scprep.plot.colors.tab40`
    - `scprep.plot.colors.tab10_continuous`

    If the number of colors required is less than the number of colors
    available, colors are selected specifically in order to reduce similarity
    between selected colors.

    Parameters
    ----------
    n : int, optional (default: 10)
        Number of required colors.

    Returns
    -------
    cmap : `matplotlib.colors.ListedColormap`
    """
    if n < 1:
        raise ValueError("Expected n >= 1. Got {}".format(n))
    n_shades = int(np.ceil(n / 10))
    if n_shades == 1:
        cmap = mpl.cm.tab10
    elif n_shades == 2:
        cmap = mpl.cm.tab20
    elif n_shades == 3:
        cmap = tab30()
    elif n_shades == 4:
        cmap = tab40()
    else:
        cmap = tab10_continuous(n_colors=10, n_step=n_shades)
    # restrict to n values
    if n > 1 and n < cmap.N:
        select_idx = np.tile(np.arange(10), n_shades) * n_shades + np.repeat(
            np.arange(n_shades), 10
        )
        cmap = mpl.colors.ListedColormap(np.array(cmap.colors)[select_idx[:n]])
    return cmap
