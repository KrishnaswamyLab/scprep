import pandas as pd
import numpy as np
import fcsparser

import os
import copy
import shutil
import zipfile
import urllib
import unittest

import scprep
import scprep.io.utils

from tools import data, utils

from scipy import sparse
from parameterized import parameterized
from nose.tools import assert_raises


class TestMatrixToDataFrame(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.X_dense = data.load_10X(sparse=False)
        self.X_sparse = data.load_10X(sparse=True)
        self.X_numpy = self.X_dense.to_numpy()
        self.X_coo = self.X_sparse.sparse.to_coo()
        self.cell_names = self.X_dense.index
        self.gene_names = self.X_dense.columns

    def test_matrix_to_dataframe_no_names_sparse(self):
        Y = scprep.io.utils._matrix_to_data_frame(self.X_numpy, sparse=True)
        assert isinstance(Y, sparse.csr_matrix)
        assert np.all(scprep.utils.toarray(Y) == self.X_numpy)
        Y = scprep.io.utils._matrix_to_data_frame(self.X_coo, sparse=True)
        assert isinstance(Y, sparse.spmatrix)
        assert np.all(scprep.utils.toarray(Y) == self.X_numpy)

    def test_matrix_to_dataframe_no_names_dataframe_sparse(self):
        Y = scprep.io.utils._matrix_to_data_frame(self.X_dense, sparse=True)
        assert scprep.utils.is_sparse_dataframe(Y)
        assert not scprep.utils.is_SparseDataFrame(Y)
        assert np.all(scprep.utils.toarray(Y) == self.X_numpy)
        utils.assert_matrix_class_equivalent(Y, self.X_sparse)
        Y = scprep.io.utils._matrix_to_data_frame(self.X_sparse, sparse=True)
        assert scprep.utils.is_sparse_dataframe(Y)
        assert not scprep.utils.is_SparseDataFrame(Y)
        assert np.all(scprep.utils.toarray(Y) == self.X_numpy)
        utils.assert_matrix_class_equivalent(Y, self.X_sparse)

    def test_matrix_to_dataframe_no_names_dense(self):
        Y = scprep.io.utils._matrix_to_data_frame(self.X_numpy, sparse=False)
        assert isinstance(Y, np.ndarray)
        assert np.all(Y == self.X_numpy)
        Y = scprep.io.utils._matrix_to_data_frame(self.X_coo, sparse=False)
        assert isinstance(Y, np.ndarray)
        assert np.all(Y == self.X_numpy)

    def test_matrix_to_dataframe_no_names_dataframe_dense(self):
        Y = scprep.io.utils._matrix_to_data_frame(self.X_dense, sparse=False)
        assert isinstance(Y, pd.DataFrame)
        assert not scprep.utils.is_sparse_dataframe(Y)
        assert not scprep.utils.is_SparseDataFrame(Y)
        assert np.all(scprep.utils.toarray(Y) == self.X_numpy)
        utils.assert_matrix_class_equivalent(Y, self.X_dense)
        Y = scprep.io.utils._matrix_to_data_frame(self.X_sparse, sparse=False)
        assert isinstance(Y, pd.DataFrame)
        assert not scprep.utils.is_sparse_dataframe(Y)
        assert not scprep.utils.is_SparseDataFrame(Y)
        assert np.all(scprep.utils.toarray(Y) == self.X_numpy)
        utils.assert_matrix_class_equivalent(Y, self.X_dense)

    def test_matrix_to_dataframe_names_sparse(self):
        Y = scprep.io.utils._matrix_to_data_frame(
            self.X_dense,
            cell_names=self.cell_names,
            gene_names=self.gene_names,
            sparse=True,
        )
        assert scprep.utils.is_sparse_dataframe(Y)
        assert not scprep.utils.is_SparseDataFrame(Y)
        assert np.all(scprep.utils.toarray(Y) == self.X_numpy)
        utils.assert_matrix_class_equivalent(Y, self.X_sparse)
        Y = scprep.io.utils._matrix_to_data_frame(
            self.X_sparse,
            cell_names=self.cell_names,
            gene_names=self.gene_names,
            sparse=True,
        )
        assert scprep.utils.is_sparse_dataframe(Y)
        assert not scprep.utils.is_SparseDataFrame(Y)
        assert np.all(scprep.utils.toarray(Y) == self.X_numpy)
        utils.assert_matrix_class_equivalent(Y, self.X_sparse)
        Y = scprep.io.utils._matrix_to_data_frame(
            self.X_numpy,
            cell_names=self.cell_names,
            gene_names=self.gene_names,
            sparse=True,
        )
        assert scprep.utils.is_sparse_dataframe(Y)
        assert not scprep.utils.is_SparseDataFrame(Y)
        assert np.all(scprep.utils.toarray(Y) == self.X_numpy)
        utils.assert_matrix_class_equivalent(Y, self.X_sparse)

    def test_matrix_to_dataframe_names_dense(self):
        Y = scprep.io.utils._matrix_to_data_frame(
            self.X_dense,
            cell_names=self.cell_names,
            gene_names=self.gene_names,
            sparse=False,
        )
        assert isinstance(Y, pd.DataFrame)
        assert not scprep.utils.is_sparse_dataframe(Y)
        assert not scprep.utils.is_SparseDataFrame(Y)
        assert np.all(scprep.utils.toarray(Y) == self.X_numpy)
        utils.assert_matrix_class_equivalent(Y, self.X_dense)
        Y = scprep.io.utils._matrix_to_data_frame(
            self.X_sparse,
            cell_names=self.cell_names,
            gene_names=self.gene_names,
            sparse=False,
        )
        assert isinstance(Y, pd.DataFrame)
        assert not scprep.utils.is_sparse_dataframe(Y)
        assert not scprep.utils.is_SparseDataFrame(Y)
        assert np.all(scprep.utils.toarray(Y) == self.X_numpy)
        utils.assert_matrix_class_equivalent(Y, self.X_dense)
        Y = scprep.io.utils._matrix_to_data_frame(
            self.X_numpy,
            cell_names=self.cell_names,
            gene_names=self.gene_names,
            sparse=False,
        )
        assert isinstance(Y, pd.DataFrame)
        assert not scprep.utils.is_sparse_dataframe(Y)
        assert not scprep.utils.is_SparseDataFrame(Y)
        assert np.all(scprep.utils.toarray(Y) == self.X_numpy)
        utils.assert_matrix_class_equivalent(Y, self.X_dense)

    def test_parse_names_none(self):
        assert scprep.io.utils._parse_gene_names(None, self.X_numpy) is None
        assert scprep.io.utils._parse_cell_names(None, self.X_numpy) is None


def test_10X_duplicate_gene_names():
    utils.assert_warns_message(
        RuntimeWarning,
        "Duplicate gene names detected! Forcing `gene_labels='both'`. "
        "Alternatively, try `gene_labels='id'`, `allow_duplicates=True`, or "
        "load the matrix with `sparse=False`",
        scprep.io.load_10X,
        os.path.join(data.data_dir, "test_10X_duplicate_gene_names"),
        gene_labels="symbol",
        sparse=True,
    )
    utils.assert_warns_message(
        RuntimeWarning,
        "Duplicate gene names detected! Forcing dense matrix",
        scprep.io.load_10X,
        os.path.join(data.data_dir, "test_10X_duplicate_gene_names"),
        allow_duplicates=True,
        sparse=True,
    )


def test_10X():
    X = data.load_10X()
    assert X.shape == (100, 100)
    assert scprep.utils.is_sparse_dataframe(X)
    assert X.columns[0] == "Arl8b"
    X = data.load_10X(gene_labels="id", sparse=False)
    assert X.shape == (100, 100)
    assert isinstance(X, pd.DataFrame)
    assert not scprep.utils.is_sparse_dataframe(X)
    assert X.columns[0] == "ENSMUSG00000030105"
    X = data.load_10X(gene_labels="both")
    assert X.shape == (100, 100)
    assert scprep.utils.is_sparse_dataframe(X)
    assert X.columns[0] == "Arl8b (ENSMUSG00000030105)"
    X_cellranger3 = scprep.io.load_10X(
        os.path.join(data.data_dir, "test_10X_cellranger3"), gene_labels="both"
    )
    np.testing.assert_array_equal(X.index, X_cellranger3.index)
    np.testing.assert_array_equal(X.columns, X_cellranger3.columns)
    np.testing.assert_array_equal(X.index, X_cellranger3.index)
    utils.assert_raises_message(
        ValueError,
        "gene_labels='invalid' not recognized. " "Choose from ['symbol', 'id', 'both']",
        data.load_10X,
        gene_labels="invalid",
    )
    utils.assert_raises_message(
        FileNotFoundError,
        "{} is not a directory".format(os.path.join(data.data_dir, "test_10X.zip")),
        scprep.io.load_10X,
        os.path.join(data.data_dir, "test_10X.zip"),
    )
    utils.assert_raises_message(
        FileNotFoundError,
        "'matrix.mtx(.gz)', '[genes/features].tsv(.gz)', and 'barcodes.tsv(.gz)' must be present "
        "in {}".format(data.data_dir),
        scprep.io.load_10X,
        data.data_dir,
    )


@parameterized([("test_10X.zip",), ("test_10X_no_subdir.zip",)])
def test_10X_zip(filename):
    X = data.load_10X()
    filename = os.path.join(data.data_dir, filename)
    X_zip = scprep.io.load_10X_zip(filename)
    assert scprep.utils.is_sparse_dataframe(X_zip)
    assert np.sum(np.sum(X != X_zip)) == 0
    np.testing.assert_array_equal(X.columns, X_zip.columns)
    np.testing.assert_array_equal(X.index, X_zip.index)


def test_10X_zip_error():
    filename = os.path.join(data.data_dir, "test_10X.zip")
    utils.assert_raises_message(
        ValueError,
        "gene_labels='invalid' not recognized. " "Choose from ['symbol', 'id', 'both']",
        scprep.io.load_10X_zip,
        filename,
        gene_labels="invalid",
    )
    utils.assert_raises_message(
        ValueError,
        "Expected a single zipped folder containing 'matrix.mtx(.gz)', "
        "'[genes/features].tsv(.gz)', and 'barcodes.tsv(.gz)'. Got ",
        scprep.io.load_10X_zip,
        os.path.join(data.data_dir, "test_10X_invalid.zip"),
    )


def test_10X_zip_url():
    X = data.load_10X()
    filename = "https://github.com/KrishnaswamyLab/scprep/raw/master/data/test_data/test_10X.zip"
    X_zip = scprep.io.load_10X_zip(filename)
    assert scprep.utils.is_sparse_dataframe(X_zip)
    assert np.sum(np.sum(X != X_zip)) == 0
    np.testing.assert_array_equal(X.columns, X_zip.columns)
    np.testing.assert_array_equal(X.index, X_zip.index)


def test_10X_zip_url_not_a_zip():
    utils.assert_raises_message(
        zipfile.BadZipFile,
        "File is not a zip file",
        scprep.io.load_10X_zip,
        "https://github.com/KrishnaswamyLab/scprep/raw/master/data/test_data/test_10X",
    )


def test_10X_zip_url_not_a_real_website():
    assert_raises(
        urllib.error.URLError, scprep.io.load_10X_zip, "http://invalid.not.a.url/scprep"
    )


def test_10X_zip_url_404():
    utils.assert_raises_message(
        urllib.error.HTTPError,
        "HTTP Error 404: Not Found",
        scprep.io.load_10X_zip,
        "https://github.com/KrishnaswamyLab/scprep/invalid_url",
    )


def test_10X_zip_not_a_file():
    utils.assert_raises_message(
        FileNotFoundError,
        "No such file: 'not_a_file.zip'",
        scprep.io.load_10X_zip,
        "not_a_file.zip",
    )


def test_10X_HDF5():
    X = data.load_10X()
    h5_file = os.path.join(data.data_dir, "test_10X.h5")
    # automatic tables backend
    X_hdf5 = scprep.io.load_10X_HDF5(h5_file)
    assert scprep.utils.is_sparse_dataframe(X_hdf5)
    assert np.sum(np.sum(X != X_hdf5)) == 0
    np.testing.assert_array_equal(X.columns, X_hdf5.columns)
    np.testing.assert_array_equal(X.index, X_hdf5.index)
    # explicit tables backend
    X_hdf5 = scprep.io.load_10X_HDF5(h5_file, backend="tables")
    assert scprep.utils.is_sparse_dataframe(X_hdf5)
    assert np.sum(np.sum(X != X_hdf5)) == 0
    np.testing.assert_array_equal(X.columns, X_hdf5.columns)
    np.testing.assert_array_equal(X.index, X_hdf5.index)
    # explicit h5py backend
    X_hdf5 = scprep.io.load_10X_HDF5(h5_file, backend="h5py")
    assert scprep.utils.is_sparse_dataframe(X_hdf5)
    assert np.sum(np.sum(X != X_hdf5)) == 0
    np.testing.assert_array_equal(X.columns, X_hdf5.columns)
    np.testing.assert_array_equal(X.index, X_hdf5.index)
    # automatic h5py backend
    tables = scprep.io.hdf5.tables
    del scprep.io.hdf5.tables
    X_hdf5 = scprep.io.load_10X_HDF5(h5_file)
    assert scprep.utils.is_sparse_dataframe(X_hdf5)
    assert np.sum(np.sum(X != X_hdf5)) == 0
    np.testing.assert_array_equal(X.columns, X_hdf5.columns)
    np.testing.assert_array_equal(X.index, X_hdf5.index)
    scprep.io.hdf5.tables = tables


def test_10X_HDF5_cellranger3():
    X = data.load_10X()
    h5_file = os.path.join(data.data_dir, "test_10X_cellranger3.h5")
    # automatic tables backend
    X_hdf5 = scprep.io.load_10X_HDF5(h5_file)
    assert scprep.utils.is_sparse_dataframe(X_hdf5)
    assert np.sum(np.sum(X != X_hdf5)) == 0
    np.testing.assert_array_equal(X.columns, X_hdf5.columns)
    np.testing.assert_array_equal(X.index, X_hdf5.index)
    # explicit tables backend
    X_hdf5 = scprep.io.load_10X_HDF5(h5_file, backend="tables")
    assert scprep.utils.is_sparse_dataframe(X_hdf5)
    assert np.sum(np.sum(X != X_hdf5)) == 0
    np.testing.assert_array_equal(X.columns, X_hdf5.columns)
    np.testing.assert_array_equal(X.index, X_hdf5.index)
    # explicit h5py backend
    X_hdf5 = scprep.io.load_10X_HDF5(h5_file, backend="h5py")
    assert scprep.utils.is_sparse_dataframe(X_hdf5)
    assert np.sum(np.sum(X != X_hdf5)) == 0
    np.testing.assert_array_equal(X.columns, X_hdf5.columns)
    np.testing.assert_array_equal(X.index, X_hdf5.index)
    # automatic h5py backend
    tables = scprep.io.hdf5.tables
    del scprep.io.hdf5.tables
    X_hdf5 = scprep.io.load_10X_HDF5(h5_file)
    assert scprep.utils.is_sparse_dataframe(X_hdf5)
    assert np.sum(np.sum(X != X_hdf5)) == 0
    np.testing.assert_array_equal(X.columns, X_hdf5.columns)
    np.testing.assert_array_equal(X.index, X_hdf5.index)
    scprep.io.hdf5.tables = tables


def test_10X_HDF5_invalid_genome():
    h5_file = os.path.join(data.data_dir, "test_10X.h5")
    utils.assert_raises_message(
        ValueError,
        "Genome invalid not found in {}. " "Available genomes: GRCh38".format(h5_file),
        scprep.io.load_10X_HDF5,
        filename=h5_file,
        genome="invalid",
    )


def test_10X_HDF5_genome_cellranger3():
    h5_file = os.path.join(data.data_dir, "test_10X_cellranger3.h5")
    utils.assert_raises_message(
        NotImplementedError,
        "Selecting genomes for Cellranger 3.0 files is not "
        "currently supported. Please file an issue at "
        "https://github.com/KrishnaswamyLab/scprep/issues",
        scprep.io.load_10X_HDF5,
        filename=h5_file,
        genome="GRCh38",
    )


def test_10X_HDF5_invalid_backend():
    h5_file = os.path.join(data.data_dir, "test_10X.h5")
    utils.assert_raises_message(
        ValueError,
        "Expected backend in ['tables', 'h5py']. Got invalid",
        scprep.io.load_10X_HDF5,
        filename=h5_file,
        backend="invalid",
    )


def test_10X_HDF5_invalid_gene_labels():
    h5_file = os.path.join(data.data_dir, "test_10X.h5")
    utils.assert_raises_message(
        ValueError,
        "gene_labels='invalid' not recognized. " "Choose from ['symbol', 'id', 'both']",
        scprep.io.load_10X_HDF5,
        filename=h5_file,
        gene_labels="invalid",
    )


def test_csv_and_tsv():
    X = data.load_10X()
    filename = os.path.join(data.data_dir, "test_small.csv")
    X_csv = scprep.io.load_csv(
        os.path.join(data.data_dir, "test_small.csv"), gene_names=True, cell_names=True
    )
    with utils.assert_warns_message(
        RuntimeWarning,
        "Duplicate cell names detected! Some functions may not work as intended. "
        "You can fix this by running `scprep.sanitize.check_index(data)`.",
    ):
        scprep.io.load_csv(
            os.path.join(data.data_dir, "test_small.csv"),
            gene_names=True,
            cell_names=[0] + list(range(X_csv.shape[1] - 1)),
        )
    X_csv2 = scprep.io.load_csv(
        os.path.join(data.data_dir, "test_small.csv"),
        gene_names=True,
        cell_names=None,
        index_col=0,
    )
    X_csv3 = scprep.io.load_csv(
        os.path.join(data.data_dir, "test_small.csv"),
        gene_names=None,
        cell_names=True,
        header=0,
    )
    X_csv4 = scprep.io.load_csv(
        os.path.join(data.data_dir, "test_small.csv"),
        gene_names=True,
        cell_names=True,
        cell_axis="col",
    )
    X_tsv = scprep.io.load_tsv(os.path.join(data.data_dir, "test_small.tsv"))
    assert np.sum(np.sum(X != X_csv)) == 0
    assert np.sum(np.sum(X_csv != X_csv2)) == 0
    assert np.sum(np.sum(X_csv != X_csv3)) == 0
    assert np.sum(np.sum(X_csv != X_csv4.T)) == 0
    assert np.sum(np.sum(X_csv != X_tsv)) == 0
    np.testing.assert_array_equal(X.columns, X_csv.columns)
    np.testing.assert_array_equal(X.index, X_csv.index)
    np.testing.assert_array_equal(X_csv.columns, X_csv2.columns)
    np.testing.assert_array_equal(X_csv.index, X_csv2.index)
    np.testing.assert_array_equal(X_csv.columns, X_csv3.columns)
    np.testing.assert_array_equal(X_csv.index, X_csv3.index)
    np.testing.assert_array_equal(X_csv.columns, X_csv4.index)
    np.testing.assert_array_equal(X_csv.index, X_csv4.columns)
    assert isinstance(X_csv, pd.DataFrame)
    assert not scprep.utils.is_sparse_dataframe(X_csv)
    X_csv = scprep.io.load_csv(
        os.path.join(data.data_dir, "test_small.csv"),
        gene_names=os.path.join(data.data_dir, "gene_symbols.csv"),
        cell_names=os.path.join(data.data_dir, "barcodes.tsv"),
        skiprows=1,
        usecols=range(1, 101),
    )
    assert np.sum(np.sum(X != X_csv)) == 0
    np.testing.assert_array_equal(X.columns, X_csv.columns)
    np.testing.assert_array_equal(X.index, X_csv.index)
    assert isinstance(X_csv, pd.DataFrame)
    assert not scprep.utils.is_sparse_dataframe(X_csv)
    X_csv = scprep.io.load_csv(
        os.path.join(data.data_dir, "test_small.csv"),
        gene_names=X.columns,
        cell_names=X.index,
        skiprows=1,
        usecols=range(1, 101),
    )
    assert np.sum(np.sum(X != X_csv)) == 0
    np.testing.assert_array_equal(X.columns, X_csv.columns)
    np.testing.assert_array_equal(X.index, X_csv.index)
    assert isinstance(X_csv, pd.DataFrame)
    assert not scprep.utils.is_sparse_dataframe(X_csv)
    X_csv = scprep.io.load_csv(
        os.path.join(data.data_dir, "test_small.csv"),
        gene_names=None,
        cell_names=None,
        sparse=True,
        skiprows=1,
        usecols=range(1, 101),
    )
    assert np.sum(np.sum(X.to_numpy() != X_csv.to_numpy())) == 0
    assert scprep.utils.is_sparse_dataframe(X_csv)
    X_csv = scprep.io.load_csv(
        os.path.join(data.data_dir, "test_small_duplicate_gene_names.csv")
    )
    assert "DUPLICATE" in X_csv.columns
    assert "DUPLICATE.1" in X_csv.columns
    utils.assert_raises_message(
        ValueError,
        "cell_axis neither not recognized. " "Expected 'row' or 'column'",
        scprep.io.load_csv,
        filename,
        cell_axis="neither",
    )


def test_mtx():
    X = data.load_10X()
    filename = os.path.join(data.data_dir, "test_10X", "matrix.mtx.gz")
    X_mtx = scprep.io.load_mtx(
        filename,
        gene_names=os.path.join(data.data_dir, "gene_symbols.csv"),
        cell_names=os.path.join(data.data_dir, "barcodes.tsv"),
        cell_axis="column",
    )
    assert np.sum(np.sum(X.to_numpy() != X_mtx.to_numpy())) == 0
    np.testing.assert_array_equal(X.columns, X_mtx.columns)
    np.testing.assert_array_equal(X.index, X_mtx.index)
    assert scprep.utils.is_sparse_dataframe(X_mtx)
    X_mtx = scprep.io.load_mtx(
        filename, gene_names=X.columns, cell_names=X.index, cell_axis="column"
    )
    assert np.sum(np.sum(X.to_numpy() != X_mtx.to_numpy())) == 0
    np.testing.assert_array_equal(X.columns, X_mtx.columns)
    np.testing.assert_array_equal(X.index, X_mtx.index)
    assert scprep.utils.is_sparse_dataframe(X_mtx)
    X_mtx = scprep.io.load_mtx(
        filename, gene_names=None, cell_names=None, sparse=False, cell_axis="column"
    )
    assert np.sum(np.sum(X.to_numpy() != X_mtx)) == 0
    assert isinstance(X_mtx, np.ndarray)
    utils.assert_raises_message(
        ValueError,
        "cell_axis neither not recognized. " "Expected 'row' or 'column'",
        scprep.io.load_mtx,
        filename,
        cell_axis="neither",
    )
    X_mtx = scprep.io.load_mtx(
        filename,
        gene_names=np.arange(X.shape[1]).astype("str"),
        cell_names=np.arange(X.shape[0]),
    )
    assert X_mtx.shape == (100, 100)
    assert scprep.utils.is_sparse_dataframe(X_mtx)
    assert X_mtx.columns[0] == "0"
    assert X_mtx.index[0] == 0


def test_save_mtx():
    filename = os.path.join(data.data_dir, "test_10X", "matrix.mtx.gz")
    X = scprep.io.load_mtx(
        filename,
        gene_names=os.path.join(data.data_dir, "gene_symbols.csv"),
        cell_names=os.path.join(data.data_dir, "barcodes.tsv"),
        cell_axis="column",
    )
    scprep.io.save_mtx(X, "test_mtx")
    Y = scprep.io.load_mtx(
        "test_mtx/matrix.mtx",
        gene_names="test_mtx/gene_names.tsv",
        cell_names="test_mtx/cell_names.tsv",
    )
    np.testing.assert_array_equal(X, Y)
    assert np.all(X.index == Y.index)
    assert np.all(X.columns == Y.columns)
    shutil.rmtree("test_mtx")


def _assert_fcs_meta_equal(fcsparser_meta, scprep_meta, reformat_meta=True):
    assert set(scprep_meta.keys()).difference(set(fcsparser_meta.keys())) == {
        "$DATAEND",
        "$DATASTART",
        "$ENDIAN",
    }
    for key in fcsparser_meta.keys():
        try:
            np.testing.assert_array_equal(fcsparser_meta[key], scprep_meta[key], key)
        except AssertionError:
            if key == "$NEXTDATA" or (key.startswith("$P") and key.endswith("B")):
                np.testing.assert_array_equal(
                    fcsparser_meta[key], int(scprep_meta[key]), key
                )
            elif key == "_channels_":
                for column in fcsparser_meta[key].columns:
                    scprep_column = scprep_meta[key][column].astype(
                        fcsparser_meta[key][column].dtype
                    )
                    np.testing.assert_array_equal(
                        fcsparser_meta[key][column], scprep_column, key + column
                    )
            elif key == "$DATATYPE":
                assert fcsparser_meta[key].lower() == scprep_meta[key].lower()
            else:
                raise


def test_fcs():
    path = fcsparser.test_sample_path
    meta, data = fcsparser.parse(path)
    _, _, X = scprep.io.load_fcs(path)
    assert "Time" not in X.columns
    assert len(set(X.columns).difference(data.columns)) == 0
    np.testing.assert_array_equal(X.index, data.index)
    np.testing.assert_array_equal(X.to_numpy(), data[X.columns].to_numpy())
    _, _, X = scprep.io.load_fcs(path, sparse=True)
    assert "Time" not in X.columns
    assert len(set(X.columns).difference(data.columns)) == 0
    np.testing.assert_array_equal(X.index, data.index)
    np.testing.assert_array_equal(
        X.sparse.to_dense().to_numpy(), data[X.columns].to_numpy()
    )

    X_meta, _, X = scprep.io.load_fcs(path, reformat_meta=False, override=True)
    _assert_fcs_meta_equal(meta, X_meta, reformat_meta=False)


def test_fcs_reformat_meta():
    path = fcsparser.test_sample_path
    meta, data = fcsparser.parse(path, reformat_meta=True)
    X_meta, _, X = scprep.io.load_fcs(path, reformat_meta=True, override=True)
    _assert_fcs_meta_equal(meta, X_meta)
    assert "Time" not in X.columns
    assert len(set(X.columns).difference(data.columns)) == 0
    np.testing.assert_array_equal(X.index, data.index)
    np.testing.assert_array_equal(X.values, data[X.columns].values)


def test_fcs_PnN():
    path = fcsparser.test_sample_path
    meta, data = fcsparser.parse(path, reformat_meta=True, channel_naming="$PnN")
    X_meta, _, X = scprep.io.load_fcs(
        path, reformat_meta=True, channel_naming="$PnN", override=True
    )
    _assert_fcs_meta_equal(meta, X_meta)
    assert "Time" not in X.columns
    assert len(set(X.columns).difference(data.columns)) == 0
    np.testing.assert_array_equal(X.index, data.index)
    np.testing.assert_array_equal(X.values, data[X.columns].values)


def test_fcs_file_error():
    utils.assert_raises_message(
        RuntimeError,
        "fcsparser failed to load {}, likely due to"
        " a malformed header. You can try using "
        "`override=True` to use scprep's built-in "
        "experimental FCS parser.".format(
            os.path.join(data.data_dir, "test_small.csv")
        ),
        scprep.io.load_fcs,
        os.path.join(data.data_dir, "test_small.csv"),
    )


def test_fcs_naming_error():
    path = fcsparser.test_sample_path
    utils.assert_raises_message(
        ValueError,
        "Expected channel_naming in ['$PnS', '$PnN']. " "Got 'invalid'",
        scprep.io.load_fcs,
        path,
        override=True,
        channel_naming="invalid",
    )


def test_fcs_header_error():
    path = fcsparser.test_sample_path
    meta, data = fcsparser.parse(path, reformat_meta=True, channel_naming="$PnN")
    meta_bad = copy.deepcopy(meta)
    meta_bad["$DATASTART"] = meta_bad["__header__"]["data start"]
    meta_bad["$DATAEND"] = meta_bad["__header__"]["data end"]
    meta_bad["__header__"]["data start"] = 0
    meta_bad["__header__"]["data end"] = 0
    assert (
        scprep.io.fcs._parse_fcs_header(meta_bad)["$DATASTART"]
        == scprep.io.fcs._parse_fcs_header(meta)["$DATASTART"]
    )
    assert (
        scprep.io.fcs._parse_fcs_header(meta_bad)["$DATAEND"]
        == scprep.io.fcs._parse_fcs_header(meta)["$DATAEND"]
    )

    meta_bad = copy.deepcopy(meta)
    meta_bad["$DATATYPE"] = "invalid"
    utils.assert_raises_message(
        ValueError,
        "Expected $DATATYPE in ['F', 'D']. " "Got 'invalid'",
        scprep.io.fcs._parse_fcs_header,
        meta_bad,
    )

    meta_bad = copy.deepcopy(meta)
    for byteord, endian in zip(["4,3,2,1", "1,2,3,4"], [">", "<"]):
        meta_bad["$BYTEORD"] = byteord
        assert scprep.io.fcs._parse_fcs_header(meta_bad)["$ENDIAN"] == endian
    meta_bad["$BYTEORD"] = "invalid"
    utils.assert_raises_message(
        ValueError,
        "Expected $BYTEORD in ['1,2,3,4', '4,3,2,1']. " "Got 'invalid'",
        scprep.io.fcs._parse_fcs_header,
        meta_bad,
    )


def test_parse_header():
    header1 = np.arange(10)
    header2 = os.path.join(data.data_dir, "gene_symbols.csv")
    utils.assert_raises_message(
        ValueError,
        "Expected 5 entries in gene_names. Got 10",
        scprep.io.utils._parse_header,
        header1,
        5,
    )
    utils.assert_raises_message(
        ValueError,
        "Expected 50 entries in {}. Got 100".format(os.path.abspath(header2)),
        scprep.io.utils._parse_header,
        header2,
        50,
    )


def test_download_google_drive():
    id = "1_T5bRqbid5mtuDYnyusoGvujc6fW1UKv"
    dest = "test.txt"
    scprep.io.download.download_google_drive(id, dest)
    assert os.path.isfile(dest)
    with open(dest, "r") as f:
        data = f.read()
        assert data == "test\n", data
    os.remove(dest)


def test_download_google_drive_large():
    id = "1FDDSWtSZcdQUVKpk-mPCZ8Ji1Fx8KSz9"
    response = scprep.io.download._GET_google_drive(id)
    assert response.status_code == 200
    response.close()


def test_download_url():
    X = data.load_10X()
    scprep.io.download.download_url(
        "https://github.com/KrishnaswamyLab/scprep/raw/master/data/test_data/test_10X/matrix.mtx.gz",
        "url_test.mtx.gz",
    )
    Y = scprep.io.load_mtx("url_test.mtx.gz").T
    assert (X.sparse.to_coo() - Y).nnz == 0
    os.remove("url_test.mtx.gz")


def test_download_zip():
    X = data.load_10X()
    scprep.io.download.download_and_extract_zip(
        "https://github.com/KrishnaswamyLab/scprep/raw/master/data/test_data/test_10X.zip",
        "zip_test",
    )
    Y = scprep.io.load_10X("zip_test/test_10X")
    assert np.all(X == Y)
    assert np.all(X.index == Y.index)
    assert np.all(X.columns == Y.columns)
    shutil.rmtree("zip_test")


def test_unzip_no_destination():
    X = data.load_10X()
    filename = os.path.join(data.data_dir, "test_10X.zip")
    tmp_filename = os.path.join("zip_test", "zip_extract_test.zip")
    os.mkdir("zip_test")
    shutil.copyfile(filename, tmp_filename)
    scprep.io.download.unzip(tmp_filename, delete=False)
    assert os.path.isfile(tmp_filename)
    Y = scprep.io.load_10X("zip_test/test_10X")
    assert np.all(X == Y)
    assert np.all(X.index == Y.index)
    assert np.all(X.columns == Y.columns)
    shutil.rmtree("zip_test")


def test_unzip_destination():
    X = data.load_10X()
    filename = os.path.join(data.data_dir, "test_10X.zip")
    tmp_filename = "zip_extract_test.zip"
    shutil.copyfile(filename, tmp_filename)
    scprep.io.download.unzip(tmp_filename, destination="zip_test")
    assert not os.path.isfile(tmp_filename)
    Y = scprep.io.load_10X("zip_test/test_10X")
    assert np.all(X == Y)
    assert np.all(X.index == Y.index)
    assert np.all(X.columns == Y.columns)
    shutil.rmtree("zip_test")
