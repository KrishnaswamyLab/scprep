import numpy as np
import pandas as pd

from .. import utils, stats, select
from .utils import (_with_matplotlib, _get_figure, show,
                    temp_fontsize, parse_fontsize, shift_ticklabels)
from .tools import label_axis


@_with_matplotlib
def jitter(labels, values, sigma=0.1, c=None, title=None, figsize=(8, 6),
           ax=None, fontsize=None):
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
    c : type
        Description of parameter `c`.
    title : type
        Description of parameter `title`.
    figsize : type
        Description of parameter `figsize`.
    ax : type
        Description of parameter `ax`.
    fontsize : type
        Description of parameter `fontsize`.

    Returns
    -------
    jitter(x, y, c=None,title=None,
        Description of returned object.

    """
    with temp_fontsize(fontsize):

        if not labels.shape == values.shape and len(labels.shape) == 1:
            raise ValueError(
                '`labels` and `values` must be 1D arrays of the same length.')

        labels_idx = np.arange(len(set(labels)))
        n_labels = len(set(labels_idx))

        # Calculate means
        means = np.zeros(n_labels)
        for i, cl in enumerate(np.unique(labels)):
            means[i] = np.mean(values[labels == cl])

        # Plotting cells
        x = labels + np.random.normal(0, sigma, len(labels))
        y = values
        r = shuffle_idx(y)  # How do you do the shuffling here?
        ax.scatter(x[r], y[r], c=c[r], s=2)

        # Plotting means
        ax.scatter(np.arange(n_labels), means, c='#cbc9ff',
                   edgecolors='black', lw=1.5, marker='o', zorder=3, s=100)

        # Plotting vetical lines
        for i in np.unique(labels):
            ax.axvline(i, c='k', lw=.1, zorder=0)

        # x axis
        ax.set_xticks(range(n_labels))
        ax.set_xticklabels(np.unique(labels), fontsize=14)
        #ax.set_xlabel('Clusters', fontsize=20)
