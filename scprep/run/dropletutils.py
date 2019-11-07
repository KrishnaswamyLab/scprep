import numpy as np
import pandas as pd
import warnings
from scipy import sparse

from . import r_function
from .. import utils


def install(site_repository=None, update=False, version=None, verbose=True):
    """Install the required R packages to run DropletUtils

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
        "DropletUtils",
        site_repository=site_repository,
        update=update,
        version=version,
        verbose=verbose,
    )


_EmptyDrops = r_function.RFunction(
    setup="""
        library(DropletUtils)
    """,
    args="""
        data
    """,
    body="""
        data <- t(as.matrix(data))
        result <- emptyDrops(data, lower=100, retain=NULL, barcode.args=list(),
                             niters=10000, test.ambient=FALSE, ignore=NULL, alpha=NULL)
        as.list(result)
    """,
)


def EmptyDrops(
    data, verbose=1,
):
    """Perform lineage inference with Slingshot

    Given a reduced-dimensional data matrix n by p and a vector of cluster labels
    (or matrix of soft cluster assignments, potentially including a -1 label for "unclustered"),
    this function performs lineage inference using a cluster-based minimum spanning tree and
    constructing simulatenous principal curves for branching paths through the tree.

    For more details, read about Slingshot on [GitHub](https://github.com/kstreet13/slingshot)
    and [Bioconductor](https://bioconductor.org/packages/release/bioc/html/slingshot.html).

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_dimensions]
        matrix of (reduced dimension) coordinates
        to be used for lineage inference.
    verbose : int, optional (default: 1)
        Logging verbosity between 0 and 2.

    Returns
    -------
    result : pd.DataFrame with columns Total

    Examples
    --------
    >>> import scprep
    """
    index = data.index if isinstance(data, pd.DataFrame) else None
    data = utils.to_array_or_spmatrix(data)
    if sparse.issparse(data):
        warnings.warn(
            "EmptyDrops is not currently implemented for sparse data. "
            "Converting data to dense",
            UserWarning,
        )
        data = utils.toarray(data)
    emptydrops = pd.DataFrame(_EmptyDrops(data=data, rpy_verbose=verbose), index=index)
    return emptydrops
