import numpy as np
import warnings

from . import utils, select


def library_size(data):
    """Measure the library size of each cell.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data

    Returns
    -------
    library_size : list-like, shape=[n_samples]
        Sum over all genes for each cell
    """
    library_size = utils.matrix_sum(data, axis=1)
    return library_size


def gene_set_expression(data, genes=None, library_size_normalize=True,
                        starts_with=None, ends_with=None, regex=None):
    """Measure the expression of a set of genes in each cell.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    genes : list-like, shape<=[n_features], optional (default: None)
        Integer column indices or string gene names included in gene set
    library_size_normalize : bool, optional (default: True)
        Divide gene set expression by library size
    starts_with : str or None, optional (default: None)
        If not None, select genes that start with this prefix
    ends_with : str or None, optional (default: None)
        If not None, select genes that end with this suffix
    regex : str or None, optional (default: None)
        If not None, select genes that match this regular expression

    Returns
    -------
    gene_set_expression : list-like, shape=[n_samples]
        Sum over genes for each cell
    """
    gene_data = select.select_cols(data, idx=genes, starts_with=starts_with,
                                   ends_with=ends_with, regex=regex)
    gene_set_expression = library_size(gene_data)
    if library_size_normalize:
        libsize = library_size(data)
        libsize[libsize == 0] = 1
        gene_set_expression /= libsize * np.median(np.array(libsize))
    return gene_set_expression


def _get_percentile_cutoff(data, cutoff=None, percentile=None, required=False):
    """Get a cutoff for a dataset

    Parameters
    ----------
    data : array-like
    cutoff : float or None, optional (default: None)
        Absolute cutoff value. Only one of cutoff and percentile may be given
    percentile : float or None, optional (default: None)
        Percentile cutoff value between 0 and 100.
        Only one of cutoff and percentile may be given
    required : bool, optional (default: False)
        If True, one of cutoff and percentile must be given.

    Returns
    -------
    cutoff : float or None
        Absolute cutoff value. Can only be None if required is False and
        cutoff and percentile are both None.
    """
    if percentile is not None:
        if cutoff is not None:
            raise ValueError(
                "Only one of `cutoff` and `percentile` should be given."
                "Got cutoff={}, percentile={}".format(cutoff, percentile))
        if percentile < 1:
            warnings.warn(
                "`percentile` expects values between 0 and 100."
                "Got {}. Did you mean {}?".format(percentile,
                                                  percentile * 100),
                UserWarning)
        cutoff = np.percentile(np.array(data).reshape(-1), percentile)
    elif cutoff is None and required:
        raise ValueError(
            "One of either `cutoff` or `percentile` must be given.")
    return cutoff
