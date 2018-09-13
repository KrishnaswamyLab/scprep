# author: Daniel Burkhardt <daniel.burkhardt@yale.edu>
# (C) 2018 Krishnaswamy Lab GPLv2

from __future__ import print_function, division
import numbers
import numpy as np
from scipy import stats
from sklearn import neighbors, metrics
from . import plot, utils

try:
    import matplotlib.pyplot as plt
except ImportError:
    pass


def EMD(x, y, bins=100):
    """Earth Mover's Distance between samples

    Calculates an approximation of Earth Mover's Distance (also called
    Wasserstein distance) for 2 variables. This can be thought of as the
    distance between two probability distributions. This metric is useful for
    identifying differentially expressed genes between two groups of cells. For
    more information see https://en.wikipedia.org/wiki/Wasserstein_metric.


    Parameters
    ----------
    x : array-like, shape=[n_samples]
        Input data (feature 1)
    y : array-like, shape=[n_samples]
        Input data (feature 2)
    bins : int or array-like, (default: 8)
        Passed to np.histogram to calculate CDFs for each variable.

    Returns
    -------
    emd : float
        Earth Mover's Distance between x and y.
    """
    x, y = _vector_coerce_two_dense(x, y)

    countsx, _ = np.histogram(x, bins=bins)
    countsx = countsx / countsx.sum()
    countsx = countsx.cumsum()

    countsy, _ = np.histogram(y, bins=bins)
    countsy = countsy / countsy.sum()
    countsy = countsy.cumsum()

    emd = np.abs(countsx - countsy).sum()
    return emd


def mutual_information(x, y, bins=8):
    """Mutual information score with set number of bins

    Helper function for sklearn.metrics.mutual_info_score that builds your
    contingency table for you using a set number of bins


    Parameters
    ----------
    x : array-like, shape=[n_samples]
        Input data (feature 1)
    y : array-like, shape=[n_samples]
        Input data (feature 2)
    bins : int or array-like, (default: 8)
        Passed to np.histogram2d to calculate a contingency table.

    Returns
    -------
    mi : float
        Mutual information between x and y.
    """
    x, y = _vector_coerce_two_dense(x, y)

    c_xy = np.histogram2d(x, y, bins)[0]
    mi = metrics.mutual_info_score(None, None, contingency=c_xy)
    return mi


def knnDREMI(x, y, k=10, n_bins=20, n_mesh=3, n_jobs=1,
             plot_data=None, **kwargs):
    """kNN conditional Density Resampled Estimate of Mutual Information

    Calculates k-Nearest Neighbor conditional Density Resampled Estimate of
    Mutual Information as defined in Van Dijk et al. 2018
    (doi:10.1016/j.cell.2018.05.061)

    kNN-DREMI is an adaptation of DREMI (Krishnaswamy et al. 2014,
    doi:10.1126/science.1250689) for single cell RNA-sequencing data. DREMI
    captures the functional relationship between two genes across their entire
    dynamic range. The key change to kNN-DREMI is the replacement of the heat
    diffusion-based kernel-density estimator from (Botev et al., 2010) by a
    k-nearest neighbor-based density estimator (Sricharan et al., 2012), which
    has been shown to be an effective method for sparse and high dimensional
    datasets.

    Note that kNN-DREMI, like Mutual Information and DREMI, is not symmetric.
    Here we are estimating I(Y|X). There are many good articles about mutual
    information on the web.

    Parameters
    ----------
    x : array-like, shape=[n_samples]
        Input data (independent feature)
    y : array-like, shape=[n_samples]
        Input data (dependent feature)
    k : int, range=[0:n_samples), optional (default: 10)
        Number of neighbors
    n_bins : int, range=[0:inf), optional (default: 20)
        Number of bins for density resampling
    n_mesh : int, range=[0:inf), optional (default: 3)
        In each bin, density will be calculcated around (mesh ** 2) points
    n_jobs : int, optional (default: 1)
        Used for kNN calculation
    plot_data : bool, optional (default())
        If True, DREMI create plots of the data like those seen in
        Fig 5C/D of van Dijk et al. 2018. (doi:10.1016/j.cell.2018.05.061).
    **kwargs : additional arguments for `scprep.stats.plot_knnDREMI`

    Returns
    -------
    dremi : float
        kNN condtional Density resampled estimate of mutual information

    Examples
    --------
    >>> import scprep
    >>> data = scprep.io.load_csv("my_data.csv")
    >>> dremi = scprep.stats.knnDREMI(data['GENE1'], data['GENE2'],
    ...                               plot_data=True,
    ...                               plot_filename='dremi.png')
    """
    x, y = _vector_coerce_two_dense(x, y)

    if not isinstance(k, numbers.Integral):
        raise ValueError(
            "Expected k as an integer. Got {}".format(type(k)))
    if not isinstance(n_bins, numbers.Integral):
        raise ValueError(
            "Expected n_bins as an integer. Got {}".format(type(n_bins)))
    if not isinstance(n_mesh, numbers.Integral):
        raise ValueError(
            "Expected n_mesh as an integer. Got {}".format(type(n_mesh)))

    # 0. Z-score X and Y
    x = stats.zscore(x)
    y = stats.zscore(y)

    # 1. Create bin and mesh points
    x_bins = np.linspace(min(x), max(x), n_bins + 1)  # plus 1 for edges
    y_bins = np.linspace(min(y), max(y), n_bins + 1)
    x_mesh = np.linspace(min(x), max(x), ((n_mesh + 1) * n_bins) + 1)
    y_mesh = np.linspace(min(y), max(y), ((n_mesh + 1) * n_bins) + 1)

    # calculate the kNN density around the mesh points
    mesh_points = np.vstack([np.tile(x_mesh, len(y_mesh)),
                             np.repeat(y_mesh, len(x_mesh))]).T

    # Next, we find the nearest points in the data from the mesh
    knn = neighbors.NearestNeighbors(n_neighbors=k, n_jobs=n_jobs).fit(
        np.vstack([x, y]).T)  # this is the data
    # get dists of closests points in data to mesh
    dists, _ = knn.kneighbors(mesh_points)

    # Get area, density of each point
    area = np.pi * (dists[:, -1] ** 2)
    density = k / area

    # get list of all mesh points that are not bin intersections
    mesh_mask = np.logical_or(np.isin(mesh_points[:, 0], x_bins),
                              np.isin(mesh_points[:, 1], y_bins))
    # Sum the densities of each point over the bins
    bin_density, _, _ = np.histogram2d(mesh_points[~mesh_mask, 0],
                                       mesh_points[~mesh_mask, 1],
                                       bins=[x_bins, y_bins],
                                       weights=density[~mesh_mask])
    bin_density = bin_density.T
    # sum the whole grid should be 1
    bin_density = bin_density / np.sum(bin_density)

    # Calculate conditional entropy
    # NB: not using thresholding here; entr(M) calcs -x*log(x) elementwise
    bin_density_norm = bin_density_norm = bin_density / \
        np.sum(bin_density, axis=0)  # columns sum to 1
    # calc entropy of each column
    cond_entropies = stats.entropy(bin_density_norm, base=2)

    # Mutual information (not normalized)
    marginal_entropy = stats.entropy(
        np.sum(bin_density, axis=1), base=2)  # entropy of Y

    # Multiply the entropy of each column by the density of each column
    # Conditional entropy is the entropy in Y that isn't exmplained by X
    cond_sums = np.sum(bin_density, axis=0)  # distribution of X
    conditional_entropy = np.sum(cond_entropies * cond_sums)
    mutual_info = marginal_entropy - conditional_entropy

    # DREMI
    marginal_entropy_norm = stats.entropy(np.sum(bin_density_norm, axis=1),
                                          base=2)
    cond_sums_norm = np.mean(bin_density_norm)
    conditional_entropy_norm = np.sum(cond_entropies * cond_sums_norm)

    dremi = marginal_entropy_norm - conditional_entropy_norm

    if plot_data is True:
        plot_knnDREMI(dremi, mutual_info,
                      x, y, n_bins, n_mesh,
                      density, bin_density, bin_density_norm, **kwargs)
    return dremi


@plot._with_matplotlib
def plot_knnDREMI(dremi, mutual_info, x, y, n_bins, n_mesh,
                  density, bin_density, bin_density_norm,
                  figsize=(12, 3.5), filename=None,
                  xlabel="Feature 1", ylabel="Feature 2",
                  title_fontsize=18, label_fontsize=16,
                  dpi=150):
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    # Plot raw data
    axes[0].scatter(x, y, c="k", s=4)
    axes[0].set_title("Input\ndata", fontsize=title_fontsize)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    axes[0].set_xlabel(xlabel, fontsize=label_fontsize)
    axes[0].set_ylabel(ylabel, fontsize=label_fontsize)

    # Plot kNN density
    n = ((n_mesh + 1) * n_bins) + 1
    axes[1].imshow(np.log(density.reshape(n, n)),
                   cmap='inferno', origin="lower", aspect="auto")
    for b in np.linspace(0, n, n_bins + 1):
        axes[1].axhline(b - 0.5, c="grey", linewidth=1)

    for b in np.linspace(0, n, n_bins + 1):
        axes[1].axvline(b - 0.5, c="grey", linewidth=1)

    axes[1].set_xticks([])
    axes[1].set_yticks([])
    axes[1].set_title("kNN\nDensity", fontsize=title_fontsize)
    axes[1].set_xlabel(xlabel, fontsize=label_fontsize)

    # Plot joint probability
    raw_density_data = bin_density
    axes[2].imshow(raw_density_data,
                   cmap="inferno", origin="lower", aspect="auto")
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    axes[2].set_title("Joint Prob.\nMI={:.2f}".format(mutual_info),
                      fontsize=title_fontsize)
    axes[2].set_xlabel(xlabel, fontsize=label_fontsize)

    # Plot conditional probability
    raw_density_data = bin_density_norm
    axes[3].imshow(raw_density_data,
                   cmap="inferno", origin="lower", aspect="auto")
    axes[3].set_xticks([])
    axes[3].set_yticks([])
    axes[3].set_title("Conditional Prob.\nDREMI={:.2f}".format(dremi),
                      fontsize=title_fontsize)
    axes[3].set_xlabel(xlabel, fontsize=label_fontsize)

    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename, dpi=dpi)
    if plot._mpl_is_gui_backend():
        fig.show()


def _vector_coerce_dense(x):
    x = utils.toarray(x)
    x_1d = x.flatten()
    if not len(x_1d) == x.shape[0]:
        raise ValueError(
            "x must be a 1D array. Got shape {}".format(x.shape))
    return x_1d


def _vector_coerce_two_dense(x, y):
    try:
        x = _vector_coerce_dense(x)
        y = _vector_coerce_dense(y)
    except ValueError as e:
        if "x must be a 1D array. Got shape " in str(e):
            raise ValueError("Expected x and y to be 1D arrays. "
                             "Got shapes x {}, y {}".format(x.shape, y.shape))
        else:
            raise
    return x, y
