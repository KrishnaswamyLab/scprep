import numpy as np
import pandas as pd
import warnings

from . import r_function
from .. import utils


def install(site_repository=None, update=False, version=None, verbose=True):
    """Install the required R packages to run Slingshot

    Parameters
    ----------
    site_repository : string, optional (default: None)
        additional repository in which to look for packages to install.
        This repository will be prepended to the default repositories
    update : boolean, optional (default: False)
        When False, don't attempt to update old packages.
        When True, update old packages automatically.
    version : string, optional (default: None)
        Bioconductor version to install, e.g., version = "3.8".
        The special symbol version = "devel" installs the current 'development' version.
        If None, installs from the current version.
    verbose : boolean, optional (default: True)
        Install script verbosity.
    """
    r_function.install_bioconductor(
        "slingshot",
        site_repository=site_repository,
        update=update,
        version=version,
        verbose=verbose,
    )


_Slingshot = r_function.RFunction(
    setup="""
        library(slingshot)
    """,
    args="""
        data, cluster_labels,
        start_cluster = NULL, end_cluster = NULL,
        distance = NULL, omega = NULL, lineages = list(), shrink = TRUE,
        extend = "y", reweight = TRUE, reassign = TRUE, thresh = 0.001,
        max_iter = 15, stretch = 2,
        smoother = "smooth.spline", shrink_method = "cosine",
        allow_breaks = TRUE, seed = NULL
    """,
    body="""
        set.seed(seed)
        data <- as.matrix(data)
        cluster_labels <- as.factor(cluster_labels)

        # Run Slingshot
        sling <- slingshot(data, clusterLabels = cluster_labels,
                         start.clus = start_cluster, end.clus = end_cluster,
                         dist.fun = distance, omega = omega, lineages = lineages, shrink = shrink,
                         extend = extend, reweight = reweight, reassign = reassign, thresh = thresh,
                         maxit = max_iter, stretch = stretch,
                         smoother = smoother, shrink.method = shrink_method,
                         allow.breaks = allow_breaks)
        list(pseudotime = slingPseudotime(sling),
             curves = lapply(sling@curves, function(curve) curve$s[curve$ord,]))
    """,
)


def Slingshot(
    data,
    cluster_labels,
    start_cluster=None,
    end_cluster=None,
    distance=None,
    omega=None,
    shrink=True,
    extend="y",
    reweight=True,
    reassign=True,
    thresh=0.001,
    max_iter=15,
    stretch=2,
    smoother="smooth.spline",
    shrink_method="cosine",
    allow_breaks=True,
    seed=None,
    verbose=1,
):
    """Perform lineage inference with Slingshot

    Given a reduced-dimensional data matrix n by p and a vector of cluster labels
    (or matrix of soft cluster assignments, potentially including a -1 label for "unclustered"),
    this function performs lineage inference using a cluster-based minimum spanning tree and
    constructing simulatenous principal curves for branching paths through the tree.

    For more details, read about Slingshot on GitHub_ and Bioconductor_.

    .. _GitHub: https://github.com/kstreet13/slingshot
    .. _Bioconductor: https://bioconductor.org/packages/release/bioc/html/slingshot.html

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_dimensions]
        matrix of (reduced dimension) coordinates
        to be used for lineage inference.
    cluster_labels : list-like, shape=[n_samples]
        a vector of cluster labels, optionally including -1's for "unclustered."
    start_cluster : string, optional (default: None)
        indicates the cluster(s) of origin.
        Lineages will be represented by paths coming out of this cluster.
    end_cluster : string, optional (default: None)
        indicates the cluster(s) which will be forced leaf nodes.
        This introduces a constraint on the MST algorithm.
    distance : callable, optional (default: None)
        method for calculating distances between clusters.
        Must take two matrices as input, corresponding to subsets of reduced_dim.
        If the minimum cluster size is larger than the number dimensions,
        the default is to use the joint covariance matrix to find squared distance
        between cluster centers. If not, the default is to use the diagonal of the
        joint covariance matrix. Not currently implemented
    omega : float, optional (default: None)
        this granularity parameter determines the distance between every
        real cluster and the artificial cluster.
        It is parameterized such that this distance is omega / 2,
        making omega the maximum distance between two connected clusters.
        By default, omega = Inf.
    shrink : boolean or float, optional (default: True)
        boolean or numeric between 0 and 1, determines whether and how much to shrink
        branching lineages toward their average prior to the split.
    extend : {'y', 'n', 'pc1'}, optional (default: "y")
        how to handle root and leaf clusters of lineages when
        constructing the initial, piece-wise linear curve.
    reweight : boolean, optional (default: True)
        whether to allow cells shared between lineages to be reweighted during curve-fitting.
        If True, cells shared between lineages will be iteratively
        reweighted based on the quantiles of their projection distances to each curve.
    reassign : boolean, optional (default: True)
        whether to reassign cells to lineages at each iteration.
        If True, cells will be added to a lineage when their
        projection distance to the curve is less than the median
        distance for all cells currently assigned to the lineage.
        Additionally, shared cells will be removed from a lineage if
        their projection distance to the curve is above the 90th
        percentile and their weight along the curve is less than 0.1.
    thresh : float, optional (default: 0.001)
        determines the convergence criterion. Percent change in the
        total distance from cells to their projections along curves
        must be less than thresh.
    max_iter : int, optional (default: 15)
        maximum number of iterations
    stretch : int, optional (default: 2)
        factor between 0 and 2 by which curves can be extrapolated beyond endpoints
    smoother : {"smooth.spline", "lowess", "periodic_lowess"}, optional (default: "smooth.spline")
        choice of smoother. "periodic_lowess" allows one to fit closed
        curves. Beware, you may want to use iter = 0 with "lowess".
    shrink_method : string, optional (default: "cosine")
        how to determine the appropriate amount of shrinkage for a
        branching lineage. Accepted values: "gaussian", "rectangular",
        "triangular", "epanechnikov", "biweight", "triweight",
        "cosine", "optcosine", "density".
    allow_breaks : boolean, optional (default: True)
        determines whether curves that branch very close to the origin
        should be allowed to have different starting points.
    seed : int or None, optional (default: None)
        Seed to use for generating random numbers.
    verbose : int, optional (default: 1)
        Logging verbosity between 0 and 2.

    Returns
    -------
    slingshot : dict
        Contains the following keys:
    pseudotime : array-like, shape=[n_samples, n_curves]
        Pseudotime projection of each cell onto each principal curve.
        Value is `np.nan` if the cell does not lie on the curve
    branch : list-like, shape=[n_samples]
        Branch assignment for each cell
    curves : array_like, shape=[n_curves, n_samples, n_dimensions]
        Coordinates of each principle curve in the reduced dimension

    Examples
    --------
    >>> import scprep
    >>> import phate
    >>> data, clusters = phate.tree.gen_dla(n_branch=4, n_dim=200, branch_length=200)
    >>> phate_op = phate.PHATE()
    >>> data_phate = phate_op.fit_transform(data)
    >>> slingshot = scprep.run.Slingshot(data_phate, clusters)
    >>> ax = scprep.plot.scatter2d(data_phate, c=slingshot['pseudotime'][:,0], cmap='magma', legend_title='Branch 1')
    >>> scprep.plot.scatter2d(data_phate, c=slingshot['pseudotime'][:,1], cmap='viridis', ax=ax,
    ...                       ticks=False, label_prefix='PHATE', legend_title='Branch 2')
    >>> for curve in slingshot['curves']:
    ...     ax.plot(curve[:,0], curve[:,1], c='black')
    >>> ax = scprep.plot.scatter2d(data_phate, c=slingshot['branch'], legend_title='Branch',
    ...                            ticks=False, label_prefix='PHATE')
    >>> for curve in slingshot['curves']:
    ...     ax.plot(curve[:,0], curve[:,1], c='black')
    """
    if seed is None:
        seed = np.random.randint(2 ** 16 - 1)
    if distance is not None:
        raise NotImplementedError("distance argument not currently implemented")
    np.random.seed(seed)

    index = data.index if isinstance(data, pd.DataFrame) else None

    data = utils.toarray(data)
    if data.shape[1] > 3:
        warnings.warn(
            "Expected data to be low-dimensional. "
            "Got data.shape[1] = {}".format(data.shape[1]),
            UserWarning,
        )
    cluster_labels = utils.toarray(cluster_labels).flatten()
    if not cluster_labels.shape[0] == data.shape[0]:
        raise ValueError(
            "Expected len(cluster_labels) ({}) to equal "
            "data.shape[0] ({})".format(cluster_labels.shape[0], data.shape[0])
        )

    kwargs = {}
    if start_cluster is not None:
        kwargs["start_cluster"] = start_cluster
    if end_cluster is not None:
        kwargs["end_cluster"] = end_cluster
    if omega is not None:
        kwargs["omega"] = omega

    slingshot = _Slingshot(
        data=data,
        cluster_labels=cluster_labels,
        shrink=shrink,
        extend=extend,
        reweight=reweight,
        reassign=reassign,
        thresh=thresh,
        max_iter=max_iter,
        stretch=stretch,
        smoother=smoother,
        shrink_method=shrink_method,
        allow_breaks=allow_breaks,
        **kwargs,
        seed=seed,
        rpy_verbose=verbose,
    )
    slingshot["curves"] = np.array(list(slingshot["curves"].values()))

    membership = (~np.isnan(slingshot["pseudotime"])).astype(int)
    branch = np.sum(membership * (2 ** np.arange(membership.shape[1])), axis=1)
    # reorder based on pseudotime
    branch_ids = np.unique(branch)
    branch_means = [
        np.nanmean(slingshot["pseudotime"][branch == id])
        if not np.all(np.isnan(slingshot["pseudotime"][branch == id]))
        else np.nan
        for id in branch_ids
    ]
    branch_order = np.argsort(branch_means)
    branch_old = branch.copy()
    for i in range(len(branch_order)):
        j = branch_order[i]
        if np.isnan(branch_means[j]):
            branch[branch_old == branch_ids[j]] = -1
        else:
            branch[branch_old == branch_ids[j]] = i
    slingshot["branch"] = branch

    if index is not None:
        slingshot["pseudotime"] = pd.DataFrame(slingshot["pseudotime"], index=index)
        slingshot["branch"] = pd.Series(slingshot["branch"], name="branch", index=index)
    return slingshot
