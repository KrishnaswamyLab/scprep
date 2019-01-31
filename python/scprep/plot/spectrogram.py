import numpy as np
import warnings
from scipy.linalg import block_diag

from .utils import _with_matplotlib, _get_figure, show
from .tools import generate_colorbar


@_with_matplotlib
def plot_spectrogram(spectrogram, n_freq_scales=20,
                     n_vertex_bins=0, cmap='inferno', colorbar=True,
                     ax=None, figsize=None,
                     xlabel="Frequency",
                     ylabel="Vertex",
                     cbarlabel="Magnitude", colorbar_args=None, **kwargs):
    """Plot a vertex-frequency spectrogram with scaling.

    Parameters
    ----------
    `spectrogram` : array-like, shape=[n_samples,n_samples]
        Input spectrogram
    n_freq_scales : int, optional (default: 20)
        Number of log10 frequency scales to bin the spectrogram
    n_vertex_bins : int, optional (default: 0)
        Number of vertex bins.  Not recommended unless the
        rows of `spectrogram` are ordered.
    cmap : `matplotlib` colormap or str (default: 'inferno')
        Colormap with which to draw colorbar
    colorbar : bool, optional (default: `True`)
        Draw colorbar for spectrogram
    ax : `matplotlib.Axes` or None, optional (default: None)
        Axis to plot on. If None, a new axis will be created.
    figsize : tuple or None, optional (default: None)
        If not None, sets the figure size (width, height)
    [x,y,cbar]label : str, optional
        Labels to display on the x, y, and colorbar axis.
    colorbar_args : dict or `None`, optional (default: None)
        Parameters to pass to scprep.plot.generate_colorbar.
    **kwargs : additional arguments for `matplotlib.pyplot.pcolor`

    Returns
    -------
    ax : `matplotlib.Axes`
        axis on which plot was drawn
    im : `matplotlib.collections.PolyCollection`
        image object for the spectrogram
    """

    N = spectrogram.shape[0]
    fig, ax, show_fig = _get_figure(ax, figsize)

    if n_vertex_bins == 0:
        yscaler = np.arange(N)
        vertex_downsampler = np.eye(N)
    else:
        warnings.warn(
            "Linear downsampling does not make sense for most graphs")
        yscaler = np.arange(n_vertex_bins)
        vertex_downsampler = block_diag(
            *[(np.ones((N // n_vertex_bins, 1))) for j in range(N // (N // n_vertex_bins))]).T
    if n_freq_scales == 0:
        xscaler = np.arange(N)
        mat = spectrogram
    else:
        xscaler = np.logspace(0, np.log10(N), n_freq_scales)

        binsizes = np.round(xscaler)
        binsizes = np.diff(binsizes)
        binsizes = np.where(binsizes == 0, 1, binsizes).astype(int)

        if binsizes.sum() != N:
            binsizes[-1] = binsizes[-1] - (binsizes.sum() - N)

        mat = np.ndarray((N, n_freq_scales))
        curidx = 0
        for i, step in enumerate(binsizes):
            nextidx = curidx + step
            slice = spectrogram[:, curidx:nextidx]
            mat[:, i] = np.sum(slice / np.linalg.norm(slice,
                                                      axis=0), axis=1)
            curidx = nextidx
        mat = mat / np.linalg.norm(mat, axis=0)

    im = ax.pcolor(xscaler, yscaler, vertex_downsampler@mat, cmap=cmap, **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if colorbar:
        if colorbar_args is None:
            colorbar_args = {}
        generate_colorbar(
            cmap, ax=ax, mappable=im, title=cbarlabel, **colorbar_args)
    if show_fig:
        show(fig)
    return ax, im
