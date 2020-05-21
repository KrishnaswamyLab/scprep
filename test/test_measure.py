from tools import utils, matrix, data
import scprep
import pandas as pd
import numpy as np

from scipy import sparse
from functools import partial
import unittest


class TestGeneSetExpression(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.X_dense = data.load_10X(sparse=False)
        self.X_sparse = data.load_10X(sparse=True)
        self.Y = scprep.measure.gene_set_expression(self.X_dense, genes="Arl8b")

    def test_setup(self):
        assert self.Y.shape[0] == self.X_dense.shape[0]
        utils.assert_all_equal(
            self.Y, scprep.select.select_cols(self.X_dense, idx="Arl8b")
        )

    def test_single_pandas(self):
        matrix.test_pandas_matrix_types(
            self.X_dense,
            utils.assert_transform_equals,
            Y=self.Y,
            transform=scprep.measure.gene_set_expression,
            genes="Arl8b",
        )

    def test_array_pandas(self):
        matrix.test_pandas_matrix_types(
            self.X_dense,
            utils.assert_transform_equals,
            Y=self.Y,
            transform=scprep.measure.gene_set_expression,
            genes=["Arl8b"],
        )

    def test_starts_with_pandas(self):
        matrix.test_pandas_matrix_types(
            self.X_dense,
            utils.assert_transform_equals,
            Y=self.Y,
            transform=scprep.measure.gene_set_expression,
            starts_with="Arl8b",
        )

    def test_single_all(self):
        matrix.test_all_matrix_types(
            self.X_dense,
            utils.assert_transform_equals,
            Y=self.Y,
            transform=scprep.measure.gene_set_expression,
            genes=0,
        )

    def test_array_all(self):
        matrix.test_all_matrix_types(
            self.X_dense,
            utils.assert_transform_equals,
            Y=self.Y,
            transform=scprep.measure.gene_set_expression,
            genes=[0],
        )

    def test_library_size(self):
        def test_fun(X):
            x = scprep.measure.library_size(X)
            assert x.name == "library_size"
            assert np.all(x.index == self.X_dense.index)

        matrix.test_pandas_matrix_types(self.X_dense, test_fun)

    def test_gene_set_expression(self):
        def test_fun(X):
            x = scprep.measure.gene_set_expression(X, genes=[0, 1])
            assert x.name == "expression"
            assert np.all(x.index == self.X_dense.index)

        matrix.test_pandas_matrix_types(self.X_dense, test_fun)

    def test_variable_genes(self):
        X_dense = scprep.filter.filter_rare_genes(self.X_dense, cutoff=5)
        Y = scprep.measure.gene_variability(X_dense)
        matrix.test_all_matrix_types(
            X_dense,
            utils.assert_transform_equals,
            Y,
            scprep.measure.gene_variability,
            check=utils.assert_all_close,
        )

        def test_fun(X):
            x = scprep.measure.gene_variability(X)
            assert x.name == "variability"
            assert np.all(x.index == X.columns)

        matrix.test_pandas_matrix_types(X_dense, test_fun)
