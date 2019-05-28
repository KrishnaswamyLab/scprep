# author: Scott Gigante <scott.gigante@yale.edu>
# (C) 2018 Krishnaswamy Lab GPLv2

import scipy.io as sio
from .utils import _matrix_to_data_frame


def load_mtx(mtx_file, cell_axis='row',
             gene_names=None, cell_names=None, sparse=None):
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
        If True, loads the data as a pd.SparseDataFrame. This uses less memory
        but more CPU.

    Returns
    -------
    data : array-like, shape=[n_samples, n_features]
        If either gene or cell names are given, data will be a pd.DataFrame or
        pd.SparseDataFrame. If no names are given, data will be a np.ndarray
        or scipy.sparse.spmatrix
    """
    if cell_axis not in ['row', 'column', 'col']:
        raise ValueError(
            "cell_axis {} not recognized. Expected 'row' or 'column'".format(
                cell_axis))
    # Read in mtx file
    data = sio.mmread(mtx_file)
    if cell_axis in ['column', 'col']:
        data = data.T
    data = _matrix_to_data_frame(
        data, gene_names=gene_names,
        cell_names=cell_names, sparse=sparse)
    return data
