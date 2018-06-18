from preprocessing import io
from sklearn.utils.testing import assert_warns_message
import pandas as pd
import numpy as np
import os

# TODO: write tests for fcs, hdf5
# compare same matrix in csv, fcs, mtx, hdf5

if os.getcwd().strip("/\\").endswith("test"):
    data_dir = os.path.join("..", "..", "data", "test_data")
else:
    data_dir = os.path.join("..", "data", "test_data")


def load_10X(**kwargs):
    return io.load_10X(os.path.join(data_dir, "test_10X"), **kwargs)


def test_10X_duplicate_gene_names():
    assert_warns_message(
        RuntimeWarning,
        "Duplicate gene names detected! Forcing `gene_labels='id'`. "
        "Alternatively, try `gene_labels='both'`, `allow_duplicates=True`, or "
        "load the matrix with `sparse=False`",
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
                         gene_names=os.path.join(data_dir, "gene_symbols.tsv"),
                         cell_names=os.path.join(data_dir, "barcodes.tsv"),
                         skiprows=1,
                         usecols=range(1, 101))
    assert np.sum(np.sum(df != csv_df)) == 0
    assert np.all(df.columns == csv_df.columns)
    assert np.all(df.index == csv_df.index)
    assert isinstance(csv_df, pd.DataFrame)
    assert not isinstance(csv_df, pd.SparseDataFrame)
    csv_df = io.load_csv(os.path.join(data_dir, "test_small.csv"),
                         gene_names=df.columns,
                         cell_names=df.index,
                         skiprows=1,
                         usecols=range(1, 101))
    assert np.sum(np.sum(df != csv_df)) == 0
    assert np.all(df.columns == csv_df.columns)
    assert np.all(df.index == csv_df.index)
    assert isinstance(csv_df, pd.DataFrame)
    assert not isinstance(csv_df, pd.SparseDataFrame)
    csv_df = io.load_csv(os.path.join(data_dir, "test_small.csv"),
                         gene_names=None,
                         cell_names=None,
                         sparse=True,
                         skiprows=1,
                         usecols=range(1, 101))
    assert np.sum(np.sum(df.values != csv_df.values)) == 0
    assert isinstance(csv_df, pd.SparseDataFrame)
    csv_df = io.load_csv(os.path.join(data_dir,
                                      "test_small_duplicate_gene_names.csv"))
    assert 'DUPLICATE' in csv_df.columns
    assert 'DUPLICATE.1' in csv_df.columns
