import scutils
from sklearn.utils.testing import assert_warns_message
import pandas as pd
import numpy as np
import os
import fcsparser
import shutil

# TODO: write tests for hdf5

if os.getcwd().strip("/\\").endswith("test"):
    data_dir = os.path.join("..", "..", "data", "test_data")
else:
    data_dir = os.path.join("..", "data", "test_data")


def load_10X(**kwargs):
    return scutils.io.load_10X(os.path.join(data_dir, "test_10X"),
                               **kwargs)


def test_10X_duplicate_gene_names():
    assert_warns_message(
        RuntimeWarning,
        "Duplicate gene names detected! Forcing `gene_labels='id'`. "
        "Alternatively, try `gene_labels='both'`, `allow_duplicates=True`, or "
        "load the matrix with `sparse=False`",
        scutils.io.load_10X,
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
    zip_df = scutils.io.load_10X_zip(os.path.join(data_dir, "test_10X.zip"))
    assert isinstance(zip_df, pd.SparseDataFrame)
    assert np.sum(np.sum(df != zip_df)) == 0
    assert np.all(df.columns == zip_df.columns)
    assert np.all(df.index == zip_df.index)


def test_csv():
    df = load_10X()
    csv_df = scutils.io.load_csv(
        os.path.join(data_dir, "test_small.csv"),
        gene_names=True,
        cell_names=True)
    assert np.sum(np.sum(df != csv_df)) == 0
    assert np.all(df.columns == csv_df.columns)
    assert np.all(df.index == csv_df.index)
    assert isinstance(csv_df, pd.DataFrame)
    assert not isinstance(csv_df, pd.SparseDataFrame)
    csv_df = scutils.io.load_csv(
        os.path.join(data_dir, "test_small.csv"),
        gene_names=os.path.join(
            data_dir, "gene_symbols.tsv"),
        cell_names=os.path.join(
            data_dir, "barcodes.tsv"),
        skiprows=1,
        usecols=range(1, 101))
    assert np.sum(np.sum(df != csv_df)) == 0
    assert np.all(df.columns == csv_df.columns)
    assert np.all(df.index == csv_df.index)
    assert isinstance(csv_df, pd.DataFrame)
    assert not isinstance(csv_df, pd.SparseDataFrame)
    csv_df = scutils.io.load_csv(
        os.path.join(data_dir, "test_small.csv"),
        gene_names=df.columns,
        cell_names=df.index,
        skiprows=1,
        usecols=range(1, 101))
    assert np.sum(np.sum(df != csv_df)) == 0
    assert np.all(df.columns == csv_df.columns)
    assert np.all(df.index == csv_df.index)
    assert isinstance(csv_df, pd.DataFrame)
    assert not isinstance(csv_df, pd.SparseDataFrame)
    csv_df = scutils.io.load_csv(
        os.path.join(data_dir, "test_small.csv"),
        gene_names=None,
        cell_names=None,
        sparse=True,
        skiprows=1,
        usecols=range(1, 101))
    assert np.sum(np.sum(df.values != csv_df.values)) == 0
    assert isinstance(csv_df, pd.SparseDataFrame)
    csv_df = scutils.io.load_csv(
        os.path.join(data_dir,
                     "test_small_duplicate_gene_names.csv"))
    assert 'DUPLICATE' in csv_df.columns
    assert 'DUPLICATE.1' in csv_df.columns


def test_mtx():
    df = load_10X()
    mtx_df = scutils.io.load_mtx(
        os.path.join(data_dir, "test_10X", "matrix.mtx"),
        gene_names=os.path.join(
            data_dir, "gene_symbols.tsv"),
        cell_names=os.path.join(
            data_dir, "barcodes.tsv"),
        cell_axis="column")
    assert np.sum(np.sum(df.values != mtx_df.values)) == 0
    assert np.all(df.columns == mtx_df.columns)
    assert np.all(df.index == mtx_df.index)
    assert isinstance(mtx_df, pd.SparseDataFrame)
    mtx_df = scutils.io.load_mtx(
        os.path.join(data_dir, "test_10X", "matrix.mtx"),
        gene_names=df.columns,
        cell_names=df.index,
        cell_axis="column")
    assert np.sum(np.sum(df.values != mtx_df.values)) == 0
    assert np.all(df.columns == mtx_df.columns)
    assert np.all(df.index == mtx_df.index)
    assert isinstance(mtx_df, pd.SparseDataFrame)
    mtx_df = scutils.io.load_mtx(
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
    _, X = scutils.io.load_fcs(path)
    assert 'Time' not in X.columns
    assert len(set(X.columns).difference(data.columns)) == 0
    assert np.all(X.index == data.index)
    assert np.all(X.values == data[X.columns].values)
    _, X = scutils.io.load_fcs(path, sparse=True)
    assert 'Time' not in X.columns
    assert len(set(X.columns).difference(data.columns)) == 0
    assert np.all(X.index == data.index)
    assert np.all(X.to_dense().values == data[X.columns].values)
