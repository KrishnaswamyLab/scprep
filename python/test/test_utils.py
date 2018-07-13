import scprep
from sklearn.utils.testing import assert_raise_message
import numpy as np
from load_tests import data


def test_get_gene_set():
    X = data.load_10X()
    gene_idx = np.argwhere([g.startswith("D") for g in X.columns]).flatten()
    gene_names = X.columns[gene_idx]
    assert np.all(scprep.utils.get_gene_set(X, starts_with="D") == gene_names)
    assert np.all(scprep.utils.get_gene_set(X, regex="^D") == gene_names)
    assert np.all(scprep.utils.get_gene_set(
        X.columns, regex="^D") == gene_names)
    gene_idx = np.argwhere([g.endswith("8") for g in X.columns]).flatten()
    gene_names = X.columns[gene_idx]
    assert np.all(scprep.utils.get_gene_set(X, ends_with="8") == gene_names)
    assert np.all(scprep.utils.get_gene_set(X, regex="8$") == gene_names)
    assert_raise_message(
        TypeError,
        "data must be a list of gene names or a pandas "
        "DataFrame. Got ndarray",
        scprep.utils.get_gene_set,
        data=X.values, regex="8$")


def test_get_cell_set():
    X = data.load_10X()
    cell_idx = np.argwhere([g.startswith("A") for g in X.index]).flatten()
    cell_names = X.index[cell_idx]
    assert np.all(scprep.utils.get_cell_set(X, starts_with="A") == cell_names)
    assert np.all(scprep.utils.get_cell_set(X, regex="^A") == cell_names)
    assert np.all(scprep.utils.get_cell_set(
        X.index, regex="^A") == cell_names)
    cell_idx = np.argwhere([g.endswith("G-1") for g in X.index]).flatten()
    cell_names = X.index[cell_idx]
    assert np.all(scprep.utils.get_cell_set(X, ends_with="G-1") == cell_names)
    assert np.all(scprep.utils.get_cell_set(X, regex="G\\-1$") == cell_names)
    assert_raise_message(
        TypeError,
        "data must be a list of cell names or a pandas "
        "DataFrame. Got ndarray",
        scprep.utils.get_cell_set,
        data=X.values, regex="G\\-1$")
