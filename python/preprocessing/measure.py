import numpy as np
import pandas as pd
import warnings

from .utils import select_cols


def library_size(data):
    if isinstance(data, pd.SparseDataFrame):
        # densifies matrix if you take the sum
        cell_sums = pd.Series(
            np.array(data.to_coo().sum(axis=1)).reshape(-1),
            index=data.index)
    else:
        cell_sums = data.sum(axis=1)
    if isinstance(cell_sums, np.matrix):
        cell_sums = np.array(cell_sums).reshape(-1)
    return cell_sums


def gene_set_expression(data, genes):
    gene_data = select_cols(data, genes)
    return library_size(gene_data)


def _get_percentile_cutoff(data, cutoff, percentile, required=False):
    if percentile is not None:
        if cutoff is not None:
            warnings.warn(
                "Only one of `cutoff` and `percentile` should be given.",
                UserWarning)
        if percentile < 1:
            warnings.warn(
                "`percentile` expects values between 0 and 100. "
                "Got {}. Did you mean {}?".format(percentile,
                                                  percentile * 100),
                UserWarning)
        cutoff = np.percentile(np.array(data).reshape(-1), percentile)
    elif cutoff is None and required:
        raise ValueError(
            "One of either `cutoff` or `percentile` must be given.")
    return cutoff
