# author: Scott Gigante <scott.gigante@yale.edu>
# (C) 2018 Krishnaswamy Lab GPLv2

import pandas as pd
import scipy.io as sio
import scipy.sparse as sp
import warnings
import numpy as np
import os
import zipfile
import tempfile
import urllib
import shutil

from .utils import _matrix_to_data_frame
from . import hdf5


def _combine_gene_id(symbols, ids):
    """Creates gene labels of the form SYMBOL (ID)

    Parameters
    ----------

    genes: pandas.DataFrame with columns['symbol', 'id']

    Returns
    -------

    pandas.Index with combined gene symbols and ids
    """
    columns = np.core.defchararray.add(np.array(symbols, dtype=str), " (")
    columns = np.core.defchararray.add(columns, np.array(ids, dtype=str))
    columns = np.core.defchararray.add(columns, ")")
    return columns


def _parse_10x_genes(symbols, ids, gene_labels="symbol", allow_duplicates=True):
    assert gene_labels in ["symbol", "id", "both"]
    if gene_labels == "symbol":
        columns = symbols
        if not allow_duplicates and len(np.unique(columns)) < len(columns):
            warnings.warn(
                "Duplicate gene names detected! Forcing `gene_labels='both'`. "
                "Alternatively, try `gene_labels='id'`, "
                "`allow_duplicates=True`, or load the matrix"
                " with `sparse=False`",
                RuntimeWarning,
            )
            gene_labels = "both"
    if gene_labels == "both":
        columns = _combine_gene_id(symbols, ids)
    elif gene_labels == "id":
        columns = ids
    return columns


def _find_gz_file(*path):
    """Find a file that could be gzipped."""
    path = os.path.join(*path)
    if os.path.isfile(path):
        return path
    else:
        return path + ".gz"


def load_10X(data_dir, sparse=True, gene_labels="symbol", allow_duplicates=None):
    """Basic IO for 10X data produced from the 10X Cellranger pipeline.

    A default run of the `cellranger count` command will generate gene-barcode
    matrices for secondary analysis. For both "raw" and "filtered" output,
    directories are created containing three files:
    'matrix.mtx', 'barcodes.tsv', 'genes.tsv'.
    Running `scprep.io.load_10X(data_dir)` will return a Pandas DataFrame with
    genes as columns and cells as rows.

    Parameters
    ----------
    data_dir: string
        path to input data directory
        expects 'matrix.mtx(.gz)', '[genes/features].tsv(.gz)', 'barcodes.tsv(.gz)'
        to be present and will raise an error otherwise
    sparse: boolean
        If True, a sparse Pandas DataFrame is returned.
    gene_labels: string, {'id', 'symbol', 'both'} optional, default: 'symbol'
        Whether the columns of the dataframe should contain gene ids or gene
        symbols. If 'both', returns symbols followed by ids in parentheses.
    allow_duplicates : bool, optional (default: None)
        Whether or not to allow duplicate gene names. If None, duplicates are
        allowed for dense input but not for sparse input.

    Returns
    -------
    data: array-like, shape=[n_samples, n_features]
        If sparse, data will be a pd.DataFrame[pd.SparseArray]. Otherwise, data will
        be a pd.DataFrame.
    """

    if gene_labels not in ["id", "symbol", "both"]:
        raise ValueError(
            "gene_labels='{}' not recognized. "
            "Choose from ['symbol', 'id', 'both']".format(gene_labels)
        )

    if not os.path.isdir(data_dir):
        raise FileNotFoundError("{} is not a directory".format(data_dir))

    try:
        m = sio.mmread(_find_gz_file(data_dir, "matrix.mtx"))
        try:
            genes = pd.read_csv(
                _find_gz_file(data_dir, "genes.tsv"), delimiter="\t", header=None
            )
        except FileNotFoundError:
            genes = pd.read_csv(
                _find_gz_file(data_dir, "features.tsv"), delimiter="\t", header=None
            )
        if genes.shape[1] == 2:
            # Cellranger < 3.0
            genes.columns = ["id", "symbol"]
        else:
            # Cellranger >= 3.0
            genes.columns = ["id", "symbol", "measurement"]
        barcodes = pd.read_csv(
            _find_gz_file(data_dir, "barcodes.tsv"), delimiter="\t", header=None
        )

    except (FileNotFoundError, IOError):
        raise FileNotFoundError(
            "'matrix.mtx(.gz)', '[genes/features].tsv(.gz)', and 'barcodes.tsv(.gz)' must be present "
            "in {}".format(data_dir)
        )

    cell_names = barcodes[0]
    if allow_duplicates is None:
        allow_duplicates = not sparse
    gene_names = _parse_10x_genes(
        genes["symbol"].values.astype(str),
        genes["id"].values.astype(str),
        gene_labels=gene_labels,
        allow_duplicates=allow_duplicates,
    )

    data = _matrix_to_data_frame(
        m.T, cell_names=cell_names, gene_names=gene_names, sparse=sparse
    )
    return data


def load_10X_zip(filename, sparse=True, gene_labels="symbol", allow_duplicates=None):
    """Basic IO for zipped 10X data produced from the 10X Cellranger pipeline.

    Runs `load_10X` after unzipping the data contained in `filename`

    Parameters
    ----------
    filename: string
        path to zipped input data directory
        expects 'matrix.mtx', 'genes.tsv', 'barcodes.tsv' to be present and
        will raise an error otherwise
    sparse: boolean
        If True, a sparse Pandas DataFrame is returned.
    gene_labels: string, {'id', 'symbol', 'both'} optional, default: 'symbol'
        Whether the columns of the dataframe should contain gene ids or gene
        symbols. If 'both', returns symbols followed by ids in parentheses.
    allow_duplicates : bool, optional (default: None)
        Whether or not to allow duplicate gene names. If None, duplicates are
        allowed for dense input but not for sparse input.

    Returns
    -------
    data: array-like, shape=[n_samples, n_features]
        If sparse, data will be a pd.DataFrame[pd.SparseArray]. Otherwise, data will
        be a pd.DataFrame.
    """

    if gene_labels not in ["id", "symbol", "both"]:
        raise ValueError(
            "gene_labels='{}' not recognized. "
            "Choose from ['symbol', 'id', 'both']".format(gene_labels)
        )

    if not os.path.isfile(filename):
        with tempfile.TemporaryDirectory() as download_dir:
            zip_filename = os.path.join(download_dir, "download.zip")
            try:
                with urllib.request.urlopen(filename) as url:
                    with open(zip_filename, "wb") as handle:
                        handle.write(url.read())
            except ValueError as e:
                if str(e).startswith("unknown url type:"):
                    # not actually a url
                    raise FileNotFoundError("No such file: '{}'".format(filename))
                else:
                    raise
            else:
                return load_10X_zip(
                    zip_filename,
                    sparse=sparse,
                    gene_labels=gene_labels,
                    allow_duplicates=allow_duplicates,
                )

    tmpdir = tempfile.mkdtemp()
    with zipfile.ZipFile(filename) as handle:
        files = handle.namelist()
        if len(files) < 3:
            valid_dirnames = []
        else:
            valid_dirnames = []
            for dirname in set([""] + ["/".join(f.split("/")[:-1]) for f in files]):
                subdir_files = [f for f in files if f.startswith(dirname)]
                path = lambda x: "{}/{}".format(dirname, x) if dirname != "" else x
                if (
                    (
                        path("barcodes.tsv") in subdir_files
                        or path("barcodes.tsv.gz") in subdir_files
                    )
                    and (
                        (
                            path("genes.tsv") in subdir_files
                            or path("genes.tsv.gz") in subdir_files
                        )
                        or (
                            path("features.tsv") in subdir_files
                            or path("features.tsv.gz") in subdir_files
                        )
                    )
                    and (
                        path("matrix.mtx") in subdir_files
                        or path("matrix.mtx.gz") in subdir_files
                    )
                ):
                    valid_dirnames.append(dirname)
        if len(valid_dirnames) != 1:
            raise ValueError(
                "Expected a single zipped folder containing 'matrix.mtx(.gz)', "
                "'[genes/features].tsv(.gz)', and 'barcodes.tsv(.gz)'. Got {}".format(
                    files
                )
            )
        dirname = valid_dirnames[0]
        handle.extractall(path=tmpdir)
    data = load_10X(os.path.join(tmpdir, dirname))
    shutil.rmtree(tmpdir)
    return data


@hdf5.with_HDF5
def load_10X_HDF5(
    filename,
    genome=None,
    sparse=True,
    gene_labels="symbol",
    allow_duplicates=None,
    backend=None,
):
    """Basic IO for HDF5 10X data produced from the 10X Cellranger pipeline.

    Equivalent to `load_10X` but for HDF5 format.

    Parameters
    ----------
    filename: string
        path to HDF5 input data
    genome : str or None, optional (default: None)
        Name of the genome to which CellRanger ran analysis. If None, selects
        the first available genome, and prints all available genomes if more
        than one is available. Invalid for Cellranger 3.0 HDF5 files.
    sparse: boolean
        If True, a sparse Pandas DataFrame is returned.
    gene_labels: string, {'id', 'symbol', 'both'} optional, default: 'symbol'
        Whether the columns of the dataframe should contain gene ids or gene
        symbols. If 'both', returns symbols followed by ids in parentheses.
    allow_duplicates : bool, optional (default: None)
        Whether or not to allow duplicate gene names. If None, duplicates are
        allowed for dense input but not for sparse input.
    backend : string, {'tables', 'h5py' or None} optional, default: None
        Selects the HDF5 backend. By default, selects whichever is available,
        using tables if both are available.

    Returns
    -------
    data: array-like, shape=[n_samples, n_features]
        If sparse, data will be a pd.DataFrame[pd.SparseArray]. Otherwise, data will
        be a pd.DataFrame.
    """

    if gene_labels not in ["id", "symbol", "both"]:
        raise ValueError(
            "gene_labels='{}' not recognized. "
            "Choose from ['symbol', 'id', 'both']".format(gene_labels)
        )

    # default allow_duplicates
    if allow_duplicates is None:
        allow_duplicates = not sparse

    with hdf5.open_file(filename, "r", backend=backend) as f:

        # handle genome
        groups = hdf5.list_nodes(f)
        try:
            # Cellranger 3.0
            group = hdf5.get_node(f, "matrix")
            if genome is not None:
                raise NotImplementedError(
                    "Selecting genomes for Cellranger 3.0 files is not "
                    "currently supported. Please file an issue at "
                    "https://github.com/KrishnaswamyLab/scprep/issues"
                )
        except (AttributeError, KeyError):
            # Cellranger 2.0
            if genome is None:
                print_genomes = ", ".join(groups)
                genome = groups[0]
                if len(groups) > 1:
                    print(
                        "Available genomes: {}. Selecting {} by default".format(
                            print_genomes, genome
                        )
                    )
            try:
                group = hdf5.get_node(f, genome)
            except (AttributeError, KeyError):
                print_genomes = ", ".join(groups)
                raise ValueError(
                    "Genome {} not found in {}. "
                    "Available genomes: {}".format(genome, filename, print_genomes)
                )

        try:
            # Cellranger 3.0
            features = hdf5.get_node(group, "features")
            gene_symbols = hdf5.get_node(features, "name")
            gene_ids = hdf5.get_node(features, "id")
        except (KeyError, IndexError):
            # Cellranger 2.0
            gene_symbols = hdf5.get_node(group, "gene_names")
            gene_ids = hdf5.get_node(group, "genes")

        # convert to string column names
        gene_names = _parse_10x_genes(
            symbols=[g.decode() for g in hdf5.get_values(gene_symbols)],
            ids=[g.decode() for g in hdf5.get_values(gene_ids)],
            gene_labels=gene_labels,
            allow_duplicates=allow_duplicates,
        )

        cell_names = [
            b.decode() for b in hdf5.get_values(hdf5.get_node(group, "barcodes"))
        ]
        data = hdf5.get_values(hdf5.get_node(group, "data"))
        indices = hdf5.get_values(hdf5.get_node(group, "indices"))
        indptr = hdf5.get_values(hdf5.get_node(group, "indptr"))
        shape = hdf5.get_values(hdf5.get_node(group, "shape"))
        data = sp.csc_matrix((data, indices, indptr), shape=shape)
        data = _matrix_to_data_frame(
            data.T, gene_names=gene_names, cell_names=cell_names, sparse=sparse
        )
        return data
