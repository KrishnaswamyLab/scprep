# author: Scott Gigante <scott.gigante@yale.edu>
# (C) 2018 Krishnaswamy Lab GPLv2

from __future__ import print_function, division
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp
import warnings
import numpy as np
import os
import fcsparser
import tables


def _parse_header(header, n_expected, header_type="gene_names"):
    """

    Parameters
    ----------
    header : `str` filename, array-like or `None`

    n_expected : `int`
        Expected header length

    header_type : argument name for error printing
    """
    if header is None or header is False:
        return None
    elif isinstance(header, str):
        # treat as a file
        if header.endswith("tsv"):
            delimiter = "\t"
        else:
            delimiter = ","
        columns = pd.read_csv(header, delimiter=delimiter,
                              header=None).toarray().reshape(-1)
        if not len(columns) == n_expected:
            raise ValueError("Expected {} entries in {}. Got {}".format(
                n_expected, header, len(columns)))
    else:
        # treat as list
        if not len(columns) == n_expected:
            raise ValueError("Expected {} entries in {}. Got {}".format(
                n_expected, header_type, len(columns)))
    return columns


def _parse_gene_names(header, data):
    return _parse_header(header, data.shape[1],
                         header_type="gene_names")


def _parse_cell_names(header, data):
    return _parse_header(header, data.shape[0],
                         header_type="cell_names")


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
    if gene_names is None and cell_names is None:
        # just a matrix
        if sparse is not None:
            if sparse and not sp.issparse(data):
                # return scipy.sparse.csr_matrix
                data = sp.csr_matrix(data)
            elif sp.issparse(data) and not sparse:
                # return numpy.ndarray
                data = data.toarray()
        else:
            # return data as is
            pass
        return data
    else:
        gene_names = _parse_gene_names(gene_names)
        cell_names = _parse_cell_names(cell_names)
        # dataframe with index and/or columns
        if sparse is None:
            # let the input data decide
            sparse = sp.issparse(data)
        if sparse and len(np.unique(gene_names)) < len(gene_names):
            warnings.warn(
                "Duplicate gene names detected! Forcing dense matrix",
                RuntimeWarning)
            sparse = False
        if sparse:
            # return pandas.SparseDataFrame
            data = pd.SparseDataFrame(data, default_fill_value=0.0,
                                      index=cell_names, columns=gene_names)
        else:
            # return pandas.DataFrame
            data = pd.DataFrame(data, index=cell_names, columns=gene_names)
        return data


def _read_csv_sparse(filename, chunksize=1000000, **kwargs):
    chunks = pd.read_csv(filename, chunksize=chunksize, **kwargs)
    data = pd.concat(chunk.to_sparse(fill_value=0.0)
                     for chunk in chunks)
    return data


def load_csv(filename, cell_axis=0, delimiter=',',
             gene_names=True, cell_names=True,
             sparse=False):
    """
    gene_names : `bool`, `str`, array-like, or `None` (default: True)
        If `True`, we assume gene names are in the first row/column. Otherwise
        expects a filename or an array containing a list of gene symbols or ids

    cell_names : `bool`, `str`, array-like, or `None` (default: True)
        If `True`, we assume cell names are in the first row/column. Otherwise
        expects a filename or an array containing a list of cell barcodes.
    """
    if cell_axis not in ['row', 'column', 'col']:
        raise ValueError(
            "cell_axis {} not recognized. Expected 'row' or 'column'".format(
                cell_axis))

    if cell_names is True:
        index_col = 0
        cell_names = None
    else:
        index_col = None
    if gene_names is True:
        header = True
        gene_names = None
    else:
        header = False

    # Read in csv file
    if sparse:
        read_fun = _read_csv_sparse
    else:
        read_fun = pd.read_csv
    data = read_fun(filename, delimiter=delimiter,
                    header=header, index_col=index_col)

    if cell_axis in ['column', 'col']:
        data = data.T

    data = _matrix_to_data_frame(
        data, gene_names=gene_names,
        cell_names=cell_names, sparse=sparse)
    return data


def load_fcs(fcs_file,
             metadata_channels=['Time', 'Event_length', 'DNA1', 'DNA2',
                                'Cisplatin', 'beadDist', 'bead1']):
    # Parse the fcs file
    text, data = fcsparser.parse(fcs_file)
    # Extract the S and N features (Indexing assumed to start from 1)
    # Assumes channel names are in S
    # TODO: is valid / unnecessary?
    no_channels = text['$PAR']
    channel_names = [''] * no_channels
    for i in range(1, no_channels + 1):
        # S name
        try:
            channel_names[i - 1] = text['$P%dS' % i]
        except KeyError:
            channel_names[i - 1] = text['$P%dN' % i]
    data.columns = channel_names

    # Metadata and data
    metadata_channels = data.columns.intersection(metadata_channels)
    data_channels = data.columns.difference(metadata_channels)
    metadata = data[metadata_channels]
    data = data[data_channels]

    return data, metadata


def load_mtx(mtx_file, cell_axis='row',
             gene_names=None, cell_names=None, sparse=None):
    """
    Parameters
    ----------

    cell_axis : {'row', 'column'}
        Axis on which cells are placed. cell_axis='row' implies that the csv
        file has `n_cells` row and `n_genes` columns. `cell_axis='column'`
        implies that the csv file has `n_cells` columns and `n_genes` rows.
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


def _combine_gene_id(symbols, ids):
    """Creates gene labels of the form SYMBOL (ID)

    Parameters
    ----------

    genes: pandas.DataFrame with columns['symbol', 'id']

    Returns
    -------

    pandas.Index with combined gene symbols and ids
    """
    columns = np.core.defchararray.add(
        np.array(symbols, dtype=str), ' (')
    columns = np.core.defchararray.add(
        columns, np.array(ids, dtype=str))
    columns = np.core.defchararray.add(columns, ')')
    return columns


def _parse_10x_genes(symbols, ids, gene_labels='symbol', force_unique=False):
    if gene_labels not in ['symbol', 'id', 'both']:
        raise ValueError("gene_labels='{}' not recognized. Choose from "
                         "['symbol', 'id', 'both']")
    if gene_labels == 'both':
        columns = _combine_gene_id(symbols, ids)
    if gene_labels == 'symbol':
        columns = symbols
        if force_unique and len(np.unique(columns)) > len(columns):
            warnings.warn(
                "Duplicate gene names detected! Forcing `gene_labels='id'`."
                "Alternatively, try `gene_labels='both'` or loading the matrix"
                " with `sparse=False`", RuntimeWarning)
            gene_labels = 'id'
    if gene_labels == 'id':
        columns = ids
    return columns


def load_10X(data_dir, sparse=True, gene_labels='symbol'):
    """Basic IO for 10X data produced from the 10X Cellranger pipeline.

    A default run of the `cellranger count` command will generate gene-barcode
    matrices for secondary analysis. For both "raw" and "filtered" output,
    directories are created containing three files:
    'matrix.mtx', 'barcodes.tsv', 'genes.tsv'.
    Running `phate.io.load_10X(data_dir)` will return a Pandas DataFrame with
    genes as columns and cells as rows. The returned DataFrame will be ready to
    use with PHATE.

    Parameters
    ----------
    data_dir: string
        path to input data directory
        expects 'matrix.mtx', 'genes.tsv', 'barcodes.tsv' to be present and
        will raise and error otherwise
    sparse: boolean
        If True, a sparse Pandas DataFrame is returned.
    gene_labels: string, {'id', 'symbol', 'both'} optional, default: 'symbol'
        Whether the columns of the dataframe should contain gene ids or gene
        symbols. If 'both', returns symbols followed by ids in parentheses.

    Returns
    -------
    data: pandas.DataFrame shape = (n_cell, n_genes)
        imported data matrix
    """

    if gene_labels not in ['id', 'symbol', 'both']:
        raise ValueError("gene_labels not in ['id', 'symbol', 'both']")

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            "{} is not a directory".format(data_dir))

    try:
        m = sio.mmread(os.path.join(data_dir, "matrix.mtx"))
        genes = pd.read_csv(os.path.join(data_dir, "genes.tsv"),
                            delimiter='\t', header=None)
        genes.columns = ['id', 'symbol']
        barcodes = pd.read_csv(os.path.join(data_dir, "barcodes.tsv"),
                               delimiter='\t', header=None)

    except (FileNotFoundError, OSError):
        raise FileNotFoundError(
            "'matrix.mtx', 'genes.tsv', and 'barcodes.tsv' must be present "
            "in {}".format(data_dir))

    index = barcodes[0]
    columns = _parse_10x_genes(genes['symbol'], genes['id'],
                               gene_labels=gene_labels, force_unique=sparse)

    data = _matrix_to_data_frame(m.T, index=index,
                                 columns=columns,
                                 sparse=sparse)
    return data


def load_10x_HDF5(filename, genome, sparse=True, gene_labels='symbol'):
    with tables.open_file(filename, 'r') as f:
        try:
            group = f.get_node(f.root, genome)
        except tables.NoSuchNodeError:
            raise ValueError(
                "Genome {} not found in {}.".format(genome, filename))
            # TODO: print available genomes.
        columns = _parse_10x_genes(
            symbols=[g.decode() for g in getattr(group, 'gene_names').read()],
            ids=[g.decode() for g in getattr(group, 'gene').read()],
            gene_labels=gene_labels, force_unique=sparse)
        index = [b.decode() for b in getattr(group, 'barcodes').read()]
        data = getattr(group, 'data').read()
        indices = getattr(group, 'indices').read()
        indptr = getattr(group, 'indptr').read()
        shape = getattr(group, 'shape').read()
        data = sp.csr_matrix((data, indices, indptr), shape=shape)
        data = _matrix_to_data_frame(data.T,
                                     columns=columns,
                                     index=index,
                                     sparse=sparse)
        return data
