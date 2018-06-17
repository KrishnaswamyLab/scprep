from preprocessing import io
from sklearn.utils.testing import assert_warns_message
import pandas as pd
import numpy as np
import os

# TODO: write tests for fcs, hdf5
# compare same matrix in csv, fcs, mtx, hdf5

if os.getcwd().strip("/").endswith("test"):
    data_dir = os.path.join("..", "..", "data", "test_data")
else:
    data_dir = os.path.join("..", "data", "test_data")


def load_10X(**kwargs):
    return io.load_10X(os.path.join(data_dir, "test_10X"), **kwargs)


def test_10X_duplicate_gene_names():
    assert_warns_message(
        RuntimeWarning,
        "Duplicate gene names detected! Forcing `gene_labels='id'`. "
        "Alternatively, try `gene_labels='both'` or loading the matrix with "
        "`sparse=False`",
        io.load_10X, os.path.join(data_dir, "test_10X_duplicate_gene_names"))


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


def test_csv():
    df = load_10X()
    csv_df = io.load_csv(os.path.join(data_dir, "test_small.csv"),
                         gene_names=True,
                         cell_names=True)
    assert np.sum(np.sum(df != csv_df)) == 0
    assert np.all(df.columns == csv_df.columns)
    assert np.all(df.index == csv_df.index)
    assert isinstance(csv_df, pd.DataFrame)
    assert not isinstance(csv_df, pd.SparseDataFrame)
    csv_df = io.load_csv(os.path.join(data_dir, "test_small.csv"),
                         gene_names="../../data/test_data/gene_symbols.tsv",
                         cell_names="../../data/test_data/barcodes.tsv")
    assert np.sum(np.sum(df != csv_df)) == 0
    assert np.all(df.columns == csv_df.columns)
    assert np.all(df.index == csv_df.index)
    assert isinstance(csv_df, pd.DataFrame)
    assert not isinstance(csv_df, pd.SparseDataFrame)
    csv_df = io.load_csv(os.path.join(data_dir, "test_small.csv"),
                         gene_names=df.columns,
                         cell_names=df.index)
    assert np.sum(np.sum(df != csv_df)) == 0
    assert np.all(df.columns == csv_df.columns)
    assert np.all(df.index == csv_df.index)
    assert isinstance(csv_df, pd.DataFrame)
    assert not isinstance(csv_df, pd.SparseDataFrame)
    csv_df = io.load_csv(os.path.join(data_dir, "test_small.csv"),
                         gene_names=False,
                         cell_names=False,
                         sparse=True)
    assert np.sum(np.sum(df != csv_df)) == 0
    assert isinstance(csv_df, pd.SparseDataFrame)
    assert_warns_message(
        RuntimeWarning,
        "Duplicate gene names detected! Forcing dense matrix",
        io.load_csv,
        os.path.join(data_dir, "test_small_duplicate_gene_names.csv"))
