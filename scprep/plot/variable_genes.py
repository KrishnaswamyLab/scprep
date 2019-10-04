from .scatter import scatter
from .. import utils, measure
from ..filter import _get_filter_idx


@utils._with_pkg(pkg="matplotlib", min_version=3)
def plot_variable_genes(data, span=0.7, interpolate=0.2, kernel_size=0.05,
                        cutoff=None, percentile=90,
                        ax=None, figsize=None,
                        xlabel='Gene mean',
                        ylabel='Standardized variance',
                        title=None,
                        fontsize=None,
                        filename=None,
                        dpi=None, **kwargs):
    """Plot the histogram of gene variability

    Variability is computed as the deviation from a loess fit
    to the rolling median of the mean-variance curve

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data. Multiple datasets may be given as a list of array-likes.
    span : float, optional (default: 0.7)
        Fraction of genes to use when computing the loess estimate at each point
    interpolate : float, optional (default: 0.2)
        Multiple of the standard deviation of variances at which to interpolate
        linearly in order to reduce computation time.
    kernel_size : float or int, optional (default: 0.05)
        Width of rolling median window. If a float, the width is given by
        kernel_size * data.shape[1]
    cutoff : float or `None`, optional (default: `None`)
        Absolute cutoff at which to draw a vertical line.
        Only one of `cutoff` and `percentile` may be given.
    percentile : float or `None`, optional (default: 90)
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
    filename : str or None (default: None)
        file to which the output is saved
    dpi : int or None, optional (default: None)
        The resolution in dots per inch. If None it will default to the value
        savefig.dpi in the matplotlibrc file. If 'figure' it will set the dpi
        to be the value of the figure. Only used if filename is not None.
    **kwargs : additional arguments for `matplotlib.pyplot.hist`

    Returns
    -------
    ax : `matplotlib.Axes`
        axis on which plot was drawn
    """
    variability, means = measure.variable_genes(data, span=span, interpolate=interpolate,
                                                kernel_size=kernel_size, return_means=True)
    keep_cells_idx = _get_filter_idx(variability,
                                     cutoff, percentile,
                                     keep_cells='above')
    return scatter(means, variability, c=keep_cells_idx,
                   cmap={True : 'red', False : 'black'}, 
                   xlabel=xlabel, ylabel=ylabel, title=title,
                   fontsize=fontsize, filename=filename, dpi=dpi,
                   **kwargs)
