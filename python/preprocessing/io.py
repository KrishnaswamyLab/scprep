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


def _matrix_to_data_frame(data, columns=None, index=None, sparse=None):
    if columns is None and index is None:
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
        # dataframe with index and/or columns
        if sparse is None:
            # let the input data decide
            sparse = sp.issparse(data)
        if sparse and len(np.unique(columns)) < len(columns):
            warnings.warn(
                "Duplicate gene names detected! Forcing dense matrix",
                RuntimeWarning)
            sparse = False
        if sparse:
            # return pandas.SparseDataFrame
            data = pd.SparseDataFrame(data, default_fill_value=0.0,
                                      index=index, columns=columns)
        else:
            # return pandas.DataFrame
            data = pd.DataFrame(data, index=index, columns=columns)
        return data


def _combine_gene_id(symbols, ids):
    """Creates gene labels of the form SYMBOL (ID)

    Parameters
    ----------

    genes : pandas.DataFrame with columns ['symbol', 'id']

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
    data_dir : string
        path to input data directory
        expects 'matrix.mtx', 'genes.tsv', 'barcodes.csv' to be present and
        will raise and error otherwise
    sparse : boolean
        If True, a sparse Pandas DataFrame is returned.
    gene_labels : string, {'id', 'symbol', 'both'} optional, default: 'symbol'
        Whether the columns of the dataframe should contain gene ids or gene
        symbols. If 'both', returns symbols followed by ids in parentheses.

    Returns
    -------
    data : pandas.DataFrame shape=(n_cell, n_genes)
        imported data matrix
    """

    if gene_labels not in ['id', 'symbol', 'both']:
        raise ValueError("gene_labels not in ['id', 'symbol', 'both']")

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
            "in data_dir")

    index = barcodes[0]
    columns = _parse_10x_genes(genes['symbol'], genes['id'],
                               gene_labels=gene_labels, force_unique=sparse)

    data = _matrix_to_data_frame(m.T, index=index,
                                 columns=columns,
                                 sparse=sparse)

    print("Imported data matrix with %s cells and %s genes." %
          (data.shape[0], data.shape[1]))
    return data


def load_csv(counts_csv_file, cell_axis=0, delimiter=',',
             header=True, index=True,
             rows_after_header_to_skip=0, cols_after_header_to_skip=0,
             sparse=False, chunksize=1000000):
    # TODO: allow index and header to be a file path or a list
    if index:
        index_col = 0
    else:
        index_col = None

    # Read in csv file
    if sparse:
        chunks = pd.read_csv(counts_csv_file, chunksize=chunksize,
                             sep=delimiter, index_col=index_col)
        data = pd.concat(chunk.to_sparse(fill_value=0.0)
                         for chunk in chunks)
        del chunks
    else:
        data = pd.read_csv(counts_csv_file, sep=delimiter, index_col=index_col)

    data.drop(data.index[1:rows_after_header_to_skip + 1],
              axis=0, inplace=True)
    data.drop(data.columns[1:cols_after_header_to_skip + 1],
              axis=1, inplace=True)

    if cell_axis != 0:
        data = data.transpose()

    return data


def load_fcs(fcs_file,
             metadata_channels=['Time', 'Event_length', 'DNA1', 'DNA2',
                                'Cisplatin', 'beadDist', 'bead1']):

    # Parse the fcs file
    text, data = fcsparser.parse(fcs_file)
    # Extract the S and N features (Indexing assumed to start from 1)
    # Assumes channel names are in S
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


def load_mtx(mtx_file, gene_name_file, sparse=None):

    # Read in mtx file
    count_matrix = sio.mmread(mtx_file)

    gene_names = np.loadtxt(gene_name_file, dtype=np.dtype('S'))
    gene_names = np.array([gene.decode('utf-8') for gene in gene_names])

    # remove todense
    data = _matrix_to_data_frame(
        count_matrix, index=None, columns=gene_names, sparse=sparse)

    return data


def load_10x_HDF5(cls, filename, genome, gene_labels='symbol', sparse=None):
    if sparse is None:
        # hdf5 format comes in sparse form
        sparse = True
    with tables.open_file(filename, 'r') as f:
        try:
            group = f.get_node(f.root, genome)
        except tables.NoSuchNodeError:
            print("That genome does not exist in this file.")
            return None
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
