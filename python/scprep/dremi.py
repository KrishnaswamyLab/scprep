# author: Daniel Burkhardt <daniel.burkhardt@yale.edu>
# (C) 2018 Krishnaswamy Lab GPLv2

from __future__ import print_function, division
import numpy as np
import pandas as pd
from scipy.stats import entropy, zscore
from scipy.special import entr
from scipy import sparse

from sklearn.neighbors import NearestNeighbors
import matplotlib as mpl
import matplotlib.pyplot as plt

def knnDREMI(x, y, k=10, n_bins=20, n_mesh=3, n_jobs=1, plot_data=None):
    """Calculates k-Nearest Neighbor conditional Density Resampled Estimate of Mutual
    Information as defined in Van Dijk et al. 2018 (doi:10.1016/j.cell.2018.05.061)

    kNN-DREMI is an adaptation of DREMI (Krishnaswamy et al. 2014, doi:10.1126/science.1250689)
    for single cell RNA-sequencing data. DREMI captures the functional relationship between two
    genes across their entire dynamic range. The key change to kNN-DREMI is the replacement of
    the heat diffusion-based kernel-density estimator from (Botev et al., 2010) by a k-nearest
    neighbor-based density estimator (Sricharan et al., 2012), which has been shown
    to be an effective method for sparse and high dimensional datasets.

    Note that kNN-DREMI, like Mutual Information and DREMI, is not symmetric. Here we are
    estimating I(Y|X). There are many good articles about mutual information on the web.

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
    plot_data : bool, optional
        If True, DREMI will return three matrices: knn_density, joint_probability,
        and conditional probability. These are useful for creating figures like those
        seen in Fig 5C/D of van Dijk et al. 2018. (doi:10.1016/j.cell.2018.05.061)

    Returns
    -------
    dremi : float
        kNN condtional Density resampled estimate of mutual information"""

    if not (isinstance(k, int)) and (isinstance(bins, int)) and (isinstance(mesh, int)) \
           and (k > 0) and (bins > 0) and (mesh > 0):
        raise ValueError('k, bins, and mesh must all be positive ints.')
    if sparse.issparse(x) or sparse.issparse(y):
        x = x.toarray()
        y = y.toarray()

    # 0. Z-score X and Y
    x = zscore(x)
    y = zscore(y)

    # 1. Create bin and mesh points
    xb = np.linspace(min(x), max(x), n_bins + 1) # plus 1 for edges
    yb = np.linspace(min(y), max(y), n_bins + 1)
    xm = np.linspace(min(x), max(x), ((n_mesh + 1 )* n_bins ) + 1)
    ym = np.linspace(min(y), max(y), ((n_mesh + 1 )* n_bins ) + 1)

    #   get list of all mesh points that are not bin intersections
    #   we will calculate the kNN density around these points
    mesh_all = np.vstack([np.tile(xm, len(ym)), np.repeat(ym, len(xm))]).T
    intersects_x = np.hstack([np.tile(np.hstack([[True], np.repeat([False], n_mesh)]), n_bins), [True]])
    intersects_y = np.hstack([np.tile(np.hstack([[True], np.repeat([False], n_mesh)]), n_bins), [True]])
    xm = xm[~intersects_x]
    ym = ym[~intersects_y]
    mesh_points = np.vstack([np.tile(xm, len(ym)), np.repeat(ym, len(xm))]).T

    # Next, we find the nearest points in the data from the mesh
    knn = NearestNeighbors(n_neighbors=k, n_jobs=n_jobs).fit(np.vstack([x,y]).T) # this is the data
    dists, _ = knn.kneighbors(mesh_points) # get dists of closests points in data to mesh

    # Get area, density of each point
    area = np.pi * (dists[:,-1] ** 2)
    density = k / area

    # Sum the densities of each point over the bins
    bin_density ,_ ,_ = np.histogram2d(mesh_points[:,0], mesh_points[:,1], bins=[xb,yb], weights=density)
    bin_density = bin_density / np.sum(bin_density) # sum the whole grid should be 1

    # Calculate conditional entropy
    # NB: not using thresholding here; entr(M) calcs -x*log(x) elementwise
    bin_density_norm = bin_density / np.sum(bin_density, axis=0) # columns sum to 1
    cond_entropies = entropy(bin_density_norm) # calc entropy of each column

    # Mutual information (not normalized)
    marginal_entropy = entropy(np.sum(bin_density, axis=1)) # entropy of Y (?)
    cond_sums = np.sum(bin_density, axis=1) # distribution of X

    # Multiply the entropy of each column by the density of each column
    # Conditional entropy is the entropy in Y that isn't exmplained by X
    conditional_entropy = np.sum(cond_entropies * cond_sums)
    mi = marginal_entropy - conditional_entropy

    # DREMI
    marginal_entropy_norm = entropy(np.sum(bin_density_norm, axis=1))
    cond_sums_norm = np.mean(bin_density_norm)
    conditional_entropy_norm = np.sum(cond_entropies * cond_sums_norm)
    dremi = marginal_entropy_norm - conditional_entropy_norm
    if plot_data is True:
        return dremi, x, y, mi, bin_density, bin_density_norm
    else:
        return dremi

def generate_DREMI_plots(dremi, x, y, mi, bin_density, bin_density_norm, figsize=(12,3.5), filename=None):
    fig, axes = plt.subplots(1,4, figsize=(12,3.5))
    mpl.rcParams['font.sans-serif'] = "Arial"
    # Plot raw data
    ax = axes[0]
    ax.scatter(x,y, c='k')
    ax.set_title('Input\ndata')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')

    # Plot kNN density
    ax = axes[1]
    for b in yb:
        ax.axhline(b, c='grey')
    for b in xb:
        ax.axvline(b, c='grey')
        #plt.scatter(mesh_intersects[:,0], mesh_intersects[:,1])
    ax.scatter(mesh_points[:,0], mesh_points[:,1], c=np.log(density), cmap='inferno', s=4)
    #ax.scatter(x,y, c='k')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('kNN\nDensity')
    ax.set_xlabel('Feature 1')
    ax.set_xlim(xlim); ax.set_ylim(ylim)

    # Plot joint probability
    ax = axes[2]
    raw_density_data = np.log(bin_density)[::-1,:]
    cg = sns.heatmap(raw_density_data, cmap='inferno', ax=ax, cbar=False)
    cg.set_xticks([])
    cg.set_yticks([])
    cg.set_title('Joint Prob.\nMI=%.2f'%mi)
    cg.set_xlabel('Feature 1')

    # Plot conditional probability
    ax = axes[3]
    raw_density_data = np.log(bin_density_norm)[::-1,:]
    cg = sns.heatmap(raw_density_data, cmap='inferno', ax=ax, cbar=False)
    cg.set_xticks([])
    cg.set_yticks([])
    cg.set_title('Conditional Prob.\nDREMI=%.2f'%dremi)
    cg.set_xlabel('Feature 1')

    fig.subplots_adjust(wspace=-1)
    fig.tight_layout()
    fig.subplots_adjust(left=0.1)
    if filename is not None:
        fig.savefig(filename, dpi=150)
    else:
        plt.show()
