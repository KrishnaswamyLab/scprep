import numpy as np

from .. import utils
from .._lazyload import matplotlib as mpl

from .utils import _get_figure, show, temp_fontsize
from .tools import label_axis


@utils._with_pkg(pkg="matplotlib", min_version=3)
def scree_plot(
    singular_values,
    cumulative=False,
    ax=None,
    figsize=None,
    xlabel="Principal Component",
    ylabel="Explained Variance (%)",
    fontsize=None,
    filename=None,
    dpi=None,
    **kwargs
):
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
    {x,y}label : str, optional
        Labels to display on the x and y axis.
    fontsize : float or None (default: None)
        Base font size.
    filename : str or None (default: None)
        file to which the output is saved
    dpi : int or None, optional (default: None)
        The resolution in dots per inch. If None it will default to the value
        savefig.dpi in the matplotlibrc file. If 'figure' it will set the dpi
        to be the value of the figure. Only used if filename is not None.
    **kwargs : additional arguments for `matplotlib.pyplot.plot`

    Returns
    -------
    ax : `matplotlib.Axes`
        axis on which plot was drawn

    Examples
    --------
    >>> import scprep
    >>> import numpy as np
    >>> data = np.random.normal(0, 1, [200, 1000])
    >>> pca_data, singular_values = scprep.reduce.pca(data, n_components=100, return_singular_values=True)
    >>> scprep.plot.scree_plot(singular_values)
    >>> scprep.plot.scree_plot(singular_values, cumulative=True)
    """
    with temp_fontsize(fontsize):
        explained_variance = singular_values ** 2
        explained_variance = explained_variance / explained_variance.sum() * 100
        if cumulative:
            explained_variance = np.cumsum(explained_variance)
        fig, ax, show_fig = _get_figure(ax, figsize)
        ax.bar(np.arange(len(explained_variance)) + 1, explained_variance, **kwargs)
        label_axis(ax.xaxis, label=xlabel)
        label_axis(ax.yaxis, label=ylabel)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
        ax.set_xlim(0.3, len(explained_variance) + 0.7)
        if show_fig:
            show(fig)
        if filename is not None:
            fig.savefig(filename, dpi=dpi)
    return ax
