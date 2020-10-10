# author: Scott Gigante <scott.gigante@yale.edu>
# (C) 2018 Krishnaswamy Lab GPLv2

import pandas as pd

from .utils import _matrix_to_data_frame
from .. import utils


def _read_csv_sparse(filename, chunksize=10000, fill_value=0.0, **kwargs):
    """Read a csv file into a pd.DataFrame[pd.SparseArray]"""
    chunks = pd.read_csv(filename, chunksize=chunksize, **kwargs)
    data = pd.concat(
        utils.dataframe_to_sparse(chunk, fill_value=fill_value) for chunk in chunks
    )
    return data


def load_csv(
    filename,
    cell_axis="row",
    delimiter=",",
    gene_names=True,
    cell_names=True,
    sparse=False,
    chunksize=10000,
    **kwargs
):
    """Load a csv file

    Parameters
    ----------
    filename : str
        The name of the csv file to be loaded
    cell_axis : {'row', 'column'}, optional (default: 'row')
        If your data has genes on the rows and cells on the columns, use
        cell_axis='column'
    delimiter : str, optional (default: ',')
        Use '\\t' for tab separated values (tsv)
    gene_names : `bool`, `str`, array-like, or `None` (default: True)
        If `True`, we assume gene names are in the first row/column. Otherwise
        expects a filename or an array containing a list of gene symbols or ids
    cell_names : `bool`, `str`, array-like, or `None` (default: True)
        If `True`, we assume cell names are in the first row/column. Otherwise
        expects a filename or an array containing a list of cell barcodes.
    sparse : bool, optional (default: False)
        If True, loads the data as a pd.DataFrame[pd.SparseArray]. This uses less memory
        but more CPU.
    chunksize : int, optional (default: 10000)
        If `sparse=True`, read this many lines of dense data at a time
        before converting to sparse.
    **kwargs : optional arguments for `pd.read_csv`.

    Returns
    -------
    data : array-like, shape=[n_samples, n_features]
        If either gene or cell names are given, data will be a pd.DataFrame or
        pd.DataFrame[pd.SparseArray]. If no names are given, data will be a np.ndarray
        or scipy.sparse.spmatrix
    """
    if cell_axis not in ["row", "column", "col"]:
        raise ValueError(
            "cell_axis {} not recognized. Expected 'row' or 'column'".format(cell_axis)
        )

    if "index_col" in kwargs:
        # override
        index_col = kwargs["index_col"]
        cell_names = None
        del kwargs["index_col"]
    elif cell_names is True:
        index_col = 0
        cell_names = None
    else:
        index_col = None

    if "header" in kwargs:
        # override
        header = kwargs["header"]
        del kwargs["header"]
        gene_names = None
    elif gene_names is True:
        header = 0
        gene_names = None
    else:
        header = None

    # Read in csv file
    if sparse:
        read_fun = _read_csv_sparse
        kwargs["chunksize"] = chunksize
    else:
        read_fun = pd.read_csv
    data = read_fun(
        filename, delimiter=delimiter, header=header, index_col=index_col, **kwargs
    )

    if cell_axis in ["column", "col"]:
        data = data.T

    data = _matrix_to_data_frame(
        data, gene_names=gene_names, cell_names=cell_names, sparse=sparse
    )
    return data


def load_tsv(
    filename,
    cell_axis="row",
    delimiter="\t",
    gene_names=True,
    cell_names=True,
    sparse=False,
    **kwargs
):
    """Load a tsv file

    Parameters
    ----------
    filename : str
        The name of the csv file to be loaded
    cell_axis : {'row', 'column'}, optional (default: 'row')
        If your data has genes on the rows and cells on the columns, use
        cell_axis='column'
    delimiter : str, optional (default: '\\t')
        Use ',' for comma separated values (csv)
    gene_names : `bool`, `str`, array-like, or `None` (default: True)
        If `True`, we assume gene names are in the first row/column. Otherwise
        expects a filename or an array containing a list of gene symbols or ids
    cell_names : `bool`, `str`, array-like, or `None` (default: True)
        If `True`, we assume cell names are in the first row/column. Otherwise
        expects a filename or an array containing a list of cell barcodes.
    sparse : bool, optional (default: False)
        If True, loads the data as a pd.DataFrame[pd.SparseArray]. This uses less memory
        but more CPU.
    **kwargs : optional arguments for `pd.read_csv`.

    Returns
    -------
    data : array-like, shape=[n_samples, n_features]
        If either gene or cell names are given, data will be a pd.DataFrame or
        pd.DataFrame[pd.SparseArray]. If no names are given, data will be a np.ndarray
        or scipy.sparse.spmatrix
    """
    return load_csv(
        filename,
        cell_axis=cell_axis,
        delimiter=delimiter,
        gene_names=gene_names,
        cell_names=cell_names,
        sparse=sparse,
        **kwargs,
    )
