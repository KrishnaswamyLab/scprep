# author: Scott Gigante <scott.gigante@yale.edu>
# (C) 2018 Krishnaswamy Lab GPLv2

import pandas as pd
import scipy.sparse as sp
import warnings
import numpy as np

from .. import utils, sanitize


def _parse_header(header, n_expected, header_type="gene_names"):
    """
    Parameters
    ----------
    header : `str` filename, array-like or `None`

    n_expected : `int`
        Expected header length

    header_type : argument name for error printing

    Returns
    -------
    columns : list-like or `None`
        Parsed column names.
    """
    if header is None or header is False:
        return None
    elif isinstance(header, str):
        # treat as a file
        if header.endswith("tsv"):
            delimiter = "\t"
        else:
            delimiter = ","
        columns = (
            pd.read_csv(header, delimiter=delimiter, header=None)
            .values.flatten()
            .astype(str)
        )
        if not len(columns) == n_expected:
            raise ValueError(
                "Expected {} entries in {}. Got {}".format(
                    n_expected, header, len(columns)
                )
            )
    else:
        # treat as list
        columns = header
        if not len(columns) == n_expected:
            raise ValueError(
                "Expected {} entries in {}. Got {}".format(
                    n_expected, header_type, len(columns)
                )
            )
    return columns


def _parse_gene_names(header, data):
    header = _parse_header(header, data.shape[1], header_type="gene_names")
    if header is None:
        try:
            return data.columns
        except AttributeError:
            pass
    return header


def _parse_cell_names(header, data):
    header = _parse_header(header, data.shape[0], header_type="cell_names")
    if header is None:
        try:
            return data.index
        except AttributeError:
            pass
    return header


def _matrix_to_data_frame(data, gene_names=None, cell_names=None, sparse=None):
    """Return the optimal data type given data, gene names and cell names.

    Parameters
    ----------

    data : array-like

    gene_names : `str`, array-like or `None` (default: None)
        Either a filename or an array containing a list of gene symbols or ids.

    cell_names : `str`, array-like or `None` (default: None)
        Either a filename or an array containing a list of cell barcodes.

    sparse : `bool` or `None` (default: None)
        If not `None`, overrides default sparsity of the data.
    """
    if gene_names is None and cell_names is None and not isinstance(data, pd.DataFrame):
        # just a matrix
        if sparse is not None:
            if sparse:
                if not sp.issparse(data):
                    # return scipy.sparse.csr_matrix
                    data = sp.csr_matrix(data)
            elif sp.issparse(data) and not sparse:
                # return numpy.ndarray
                data = data.toarray()
        else:
            # return data as is
            pass
    else:
        gene_names = _parse_gene_names(gene_names, data)
        cell_names = _parse_cell_names(cell_names, data)
        # dataframe with index and/or columns
        if sparse is None:
            # let the input data decide
            sparse = utils.is_sparse_dataframe(data) or sp.issparse(data)
        if (
            sparse
            and gene_names is not None
            and len(np.unique(gene_names)) < len(gene_names)
        ):
            warnings.warn(
                "Duplicate gene names detected! Forcing dense matrix.", RuntimeWarning
            )
            sparse = False
        if cell_names is not None and len(np.unique(cell_names)) < len(cell_names):
            warnings.warn(
                "Duplicate cell names detected! Some functions may not work as intended. You can fix this by running `scprep.sanitize.check_index(data)`.",
                RuntimeWarning,
            )
        if sparse:
            # return pandas.DataFrame[SparseArray]
            if isinstance(data, pd.DataFrame):
                if gene_names is not None:
                    data.columns = gene_names
                if cell_names is not None:
                    data.index = cell_names
                if not utils.is_sparse_dataframe(data):
                    data = utils.dataframe_to_sparse(data, fill_value=0.0)
            elif sp.issparse(data):
                data = pd.DataFrame.sparse.from_spmatrix(
                    data, index=cell_names, columns=gene_names
                )
            else:
                data = pd.DataFrame(data, index=cell_names, columns=gene_names)
                data = utils.dataframe_to_sparse(data, fill_value=0.0)
        else:
            # return pandas.DataFrame
            if isinstance(data, pd.DataFrame):
                if gene_names is not None:
                    data.columns = gene_names
                if cell_names is not None:
                    data.index = cell_names
                if utils.is_sparse_dataframe(data):
                    data = data.sparse.to_dense()
            else:
                if sp.issparse(data):
                    data = data.toarray()
                data = pd.DataFrame(data, index=cell_names, columns=gene_names)
    data = sanitize.check_numeric(data, suppress_errors=True)
    return data
