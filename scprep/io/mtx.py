# author: Scott Gigante <scott.gigante@yale.edu>
# (C) 2018 Krishnaswamy Lab GPLv2

import scipy.io as sio
from scipy import sparse
import pandas as pd
import os

from .utils import _matrix_to_data_frame
from .. import utils


def load_mtx(mtx_file, cell_axis="row", gene_names=None, cell_names=None, sparse=None):
    """Load a mtx file

    Parameters
    ----------
    filename : str
        The name of the mtx file to be loaded
    cell_axis : {'row', 'column'}, optional (default: 'row')
        If your data has genes on the rows and cells on the columns, use
        cell_axis='column'
    gene_names : `str`, array-like, or `None` (default: None)
        Expects a filename or an array containing a list of gene symbols or ids
    cell_names : `str`, array-like, or `None` (default: None)
        Expects a filename or an array containing a list of cell barcodes.
    sparse : bool, optional (default: None)
        If True, loads the data as a pd.DataFrame[pd.SparseArray]. This uses less memory
        but more CPU.

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
    # Read in mtx file
    data = sio.mmread(mtx_file)
    if cell_axis in ["column", "col"]:
        data = data.T
    data = _matrix_to_data_frame(
        data, gene_names=gene_names, cell_names=cell_names, sparse=sparse
    )
    return data


def save_mtx(data, destination, cell_names=None, gene_names=None):
    """Save a mtx file

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data, saved to destination/matrix.mtx
    destination : str
        Directory in which to save the data
    cell_names : list-like, shape=[n_samples], optional (default: None)
        Cell names associated with rows, saved to destination/cell_names.tsv.
        If `data` is a pandas DataFrame and `cell_names` is None,
        these are autopopulated from `data.index`.
    gene_names : list-like, shape=[n_features], optional (default: None)
        Cell names associated with rows, saved to destination/gene_names.tsv.
        If `data` is a pandas DataFrame and `gene_names` is None,
        these are autopopulated from `data.columns`.

    Examples
    --------
    >>> import scprep
    >>> scprep.io.save_mtx(data, destination="my_data")
    >>> reload = scprep.io.load_mtx("my_data/matrix.mtx",
    ...                             cell_names="my_data/cell_names.tsv",
    ...                             gene_names="my_data/gene_names.tsv")
    """
    if isinstance(data, pd.DataFrame):
        if cell_names is None:
            cell_names = data.index
        if gene_names is None:
            gene_names = data.columns
    data = utils.to_array_or_spmatrix(data)
    data = sparse.coo_matrix(data)
    # handle ~/ and relative paths
    destination = os.path.expanduser(destination)
    if not os.path.isdir(destination):
        os.mkdir(destination)
    if cell_names is not None:
        with open(os.path.join(destination, "cell_names.tsv"), "w") as handle:
            for name in cell_names:
                handle.write("{}\n".format(name))
    if gene_names is not None:
        with open(os.path.join(destination, "gene_names.tsv"), "w") as handle:
            for name in gene_names:
                handle.write("{}\n".format(name))
    sio.mmwrite(os.path.join(destination, "matrix.mtx"), data)
