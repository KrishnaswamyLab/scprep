import scprep
from sklearn.utils.testing import assert_warns_message, assert_raise_message
import pandas as pd
import numpy as np
import os
import fcsparser
from load_tests.data import data_dir, load_10X


def test_10X_duplicate_gene_names():
    assert_warns_message(
        RuntimeWarning,
        "Duplicate gene names detected! Forcing `gene_labels='id'`. "
        "Alternatively, try `gene_labels='both'`, `allow_duplicates=True`, or "
        "load the matrix with `sparse=False`",
        scprep.io.load_10X,
        os.path.join(data_dir, "test_10X_duplicate_gene_names"))


def test_10X():
    df = load_10X()
    assert df.shape == (100, 100)
    assert isinstance(df, pd.SparseDataFrame)
    assert df.columns[0] == "Arl8b"
    df = load_10X(gene_labels='id', sparse=False)
    assert df.shape == (100, 100)
    assert isinstance(df, pd.DataFrame)
    assert not isinstance(df, pd.SparseDataFrame)
    assert df.columns[0] == "ENSMUSG00000030105"
    df = load_10X(gene_labels='both')
    assert df.shape == (100, 100)
    assert isinstance(df, pd.SparseDataFrame)
    assert df.columns[0] == "Arl8b (ENSMUSG00000030105)"


def test_10X_zip():
    df = load_10X()
    zip_df = scprep.io.load_10X_zip(
        os.path.join(data_dir, "test_10X.zip"))
    assert isinstance(zip_df, pd.SparseDataFrame)
    assert np.sum(np.sum(df != zip_df)) == 0
    np.testing.assert_array_equal(df.columns, zip_df.columns)
    np.testing.assert_array_equal(df.index, zip_df.index)


def test_csv():
    df = load_10X()
    filename = os.path.join(data_dir, "test_small.csv")
    csv_df = scprep.io.load_csv(
        os.path.join(data_dir, "test_small.csv"),
        gene_names=True, cell_names=True)
    csv_df2 = scprep.io.load_csv(
        os.path.join(data_dir, "test_small.csv"),
        gene_names=True, cell_names=None, index_col=0)
    csv_df3 = scprep.io.load_csv(
        os.path.join(data_dir, "test_small.csv"),
        gene_names=None, cell_names=True, header=0)
    assert np.sum(np.sum(df != csv_df)) == 0
    assert np.sum(np.sum(csv_df != csv_df2)) == 0
    assert np.sum(np.sum(csv_df != csv_df3)) == 0
    np.testing.assert_array_equal(df.columns, csv_df.columns)
    np.testing.assert_array_equal(df.index, csv_df.index)
    assert isinstance(csv_df, pd.DataFrame)
    assert not isinstance(csv_df, pd.SparseDataFrame)
    csv_df = scprep.io.load_csv(
        os.path.join(data_dir, "test_small.csv"),
        gene_names=os.path.join(
            data_dir, "gene_symbols.csv"),
        cell_names=os.path.join(
            data_dir, "barcodes.tsv"),
        skiprows=1,
        usecols=range(1, 101))
    assert np.sum(np.sum(df != csv_df)) == 0
    np.testing.assert_array_equal(df.columns, csv_df.columns)
    np.testing.assert_array_equal(df.index, csv_df.index)
    assert isinstance(csv_df, pd.DataFrame)
    assert not isinstance(csv_df, pd.SparseDataFrame)
    csv_df = scprep.io.load_csv(
        os.path.join(data_dir, "test_small.csv"),
        gene_names=df.columns,
        cell_names=df.index,
        skiprows=1,
        usecols=range(1, 101))
    assert np.sum(np.sum(df != csv_df)) == 0
    np.testing.assert_array_equal(df.columns, csv_df.columns)
    np.testing.assert_array_equal(df.index, csv_df.index)
    assert isinstance(csv_df, pd.DataFrame)
    assert not isinstance(csv_df, pd.SparseDataFrame)
    csv_df = scprep.io.load_csv(
        os.path.join(data_dir, "test_small.csv"),
        gene_names=None,
        cell_names=None,
        sparse=True,
        skiprows=1,
        usecols=range(1, 101))
    assert np.sum(np.sum(df.values != csv_df.values)) == 0
    assert isinstance(csv_df, pd.SparseDataFrame)
    csv_df = scprep.io.load_csv(
        os.path.join(data_dir,
                     "test_small_duplicate_gene_names.csv"))
    assert 'DUPLICATE' in csv_df.columns
    assert 'DUPLICATE.1' in csv_df.columns
    assert_raise_message(
        ValueError,
        "cell_axis neither not recognized. "
        "Expected 'row' or 'column'",
        scprep.io.load_csv, filename,
        cell_axis='neither')


def test_mtx():
    df = load_10X()
    mtx_df = scprep.io.load_mtx(
        os.path.join(data_dir, "test_10X", "matrix.mtx"),
        gene_names=os.path.join(
            data_dir, "gene_symbols.csv"),
        cell_names=os.path.join(
            data_dir, "barcodes.tsv"),
        cell_axis="column")
    assert np.sum(np.sum(df.values != mtx_df.values)) == 0
    np.testing.assert_array_equal(df.columns, mtx_df.columns)
    np.testing.assert_array_equal(df.index, mtx_df.index)
    assert isinstance(mtx_df, pd.SparseDataFrame)
    mtx_df = scprep.io.load_mtx(
        os.path.join(data_dir, "test_10X", "matrix.mtx"),
        gene_names=df.columns,
        cell_names=df.index,
        cell_axis="column")
    assert np.sum(np.sum(df.values != mtx_df.values)) == 0
    np.testing.assert_array_equal(df.columns, mtx_df.columns)
    np.testing.assert_array_equal(df.index, mtx_df.index)
    assert isinstance(mtx_df, pd.SparseDataFrame)
    mtx_df = scprep.io.load_mtx(
        os.path.join(data_dir, "test_10X", "matrix.mtx"),
        gene_names=None,
        cell_names=None,
        sparse=False,
        cell_axis="column")
    assert np.sum(np.sum(df.values != mtx_df)) == 0
    assert isinstance(mtx_df, np.ndarray)


def test_fcs():
    path = fcsparser.test_sample_path
    meta, data = fcsparser.parse(path)
    _, X = scprep.io.load_fcs(path)
    assert 'Time' not in X.columns
    assert len(set(X.columns).difference(data.columns)) == 0
    np.testing.assert_array_equal(X.index, data.index)
    np.testing.assert_array_equal(X.values, data[X.columns].values)
    _, X = scprep.io.load_fcs(path, sparse=True)
    assert 'Time' not in X.columns
    assert len(set(X.columns).difference(data.columns)) == 0
    np.testing.assert_array_equal(X.index, data.index)
    np.testing.assert_array_equal(
        X.to_dense().values, data[X.columns].values)




#def test_load_tsv():



def test_parse_header():
    header = np.arange(10)
    n_expected = 5
    assert_raise_message(
        ValueError,
        "Expected 5 entries in gene_names. Got 10",
        scprep.io._parse_header, header, n_expected)


