from tools import utils, matrix, data
import scprep
from scipy import sparse
import numpy as np
import pandas as pd
from sklearn import decomposition

from functools import partial
import unittest


class TestPCA(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.X = data.generate_positive_sparse_matrix(shape=[100, 3000])
        self.X_sparse = sparse.csr_matrix(self.X)
        random_pca_op = decomposition.PCA(100, random_state=42)
        self.Y_random = random_pca_op.fit_transform(self.X)
        self.S_random = random_pca_op.singular_values_
        full_pca_op = decomposition.PCA(50, svd_solver="full")
        self.Y_full = full_pca_op.fit_transform(self.X)
        self.S_full = full_pca_op.singular_values_

    def test_dense(self):
        matrix.test_dense_matrix_types(
            self.X,
            utils.assert_transform_equals,
            Y=self.Y_random,
            transform=scprep.reduce.pca,
            n_components=100,
            seed=42,
        )
        matrix.test_all_matrix_types(
            self.X,
            utils.assert_transform_equals,
            Y=self.Y_random,
            transform=scprep.reduce.pca,
            n_components=100,
            seed=42,
            method="dense",
            check=partial(utils.assert_all_close, atol=1e-10),
        )

    def test_sparse_svd(self):
        matrix.test_sparse_matrix_types(
            self.X,
            utils.assert_transform_equals,
            Y=self.Y_full,
            transform=scprep.reduce.pca,
            check=partial(utils.assert_all_close, rtol=1e-3, atol=1e-5),
            n_components=50,
            eps=0.3,
            seed=42,
            method="svd",
        )

    def test_pandas(self):
        X = pd.DataFrame(
            self.X,
            index=np.arange(self.X.shape[0]).astype(str),
            columns=np.arange(self.X.shape[1]).astype(float),
        )

        def test_fun(X_pd):
            Y = scprep.reduce.pca(X_pd, n_components=100, seed=42)
            assert isinstance(Y, pd.DataFrame)
            assert np.all(Y.index == X.index)
            assert np.all(
                Y.columns == np.array(["PC{}".format(i + 1) for i in range(Y.shape[1])])
            )

        matrix.test_pandas_matrix_types(X, test_fun)

    def test_sparse_orth_rproj(self):
        def test_fn(*args, **kwargs):
            return scprep.utils.toarray(scprep.reduce.pca(*args, **kwargs))

        matrix.test_sparse_matrix_types(
            self.X,
            utils.assert_transform_equals,
            check=utils.assert_matrix_class_equivalent,
            Y=self.Y_full,
            transform=test_fn,
            n_components=50,
            eps=0.3,
            seed=42,
            method="orth_rproj",
        )

    def test_singular_values_dense(self):
        utils.assert_all_equal(
            self.S_random,
            scprep.reduce.pca(
                self.X, n_components=100, seed=42, return_singular_values=True
            )[1],
        )

    def test_singular_values_sparse(self):
        utils.assert_all_close(
            self.S_full,
            scprep.reduce.pca(
                self.X_sparse,
                n_components=50,
                eps=0.3,
                seed=42,
                return_singular_values=True,
            )[1],
            atol=1e-14,
        )

    def test_sparse_rproj(self):
        def test_fn(*args, **kwargs):
            return scprep.utils.toarray(scprep.reduce.pca(*args, **kwargs))

        matrix.test_sparse_matrix_types(
            self.X,
            utils.assert_transform_equals,
            check=utils.assert_matrix_class_equivalent,
            Y=self.Y_full,
            transform=test_fn,
            n_components=50,
            eps=0.3,
            seed=42,
            method="rproj",
        )

    def test_eps_too_low(self):
        utils.assert_all_close(
            self.Y_random,
            scprep.reduce.pca(self.X_sparse, n_components=100, eps=0.0001, seed=42),
        )

    def test_invalid_method(self):
        utils.assert_raises_message(
            ValueError,
            "Expected `method` in ['svd', 'orth_rproj', 'rproj']. " "Got 'invalid'",
            scprep.reduce.pca,
            self.X_sparse,
            method="invalid",
        )

    def test_bad_n_components(self):
        utils.assert_raises_message(
            ValueError,
            "n_components=0 must be between 0 and " "min(n_samples, n_features)=100",
            scprep.reduce.pca,
            self.X,
            n_components=0,
        )
        utils.assert_raises_message(
            ValueError,
            "n_components=101 must be between 0 and " "min(n_samples, n_features)=100",
            scprep.reduce.pca,
            self.X,
            n_components=101,
        )

    def test_deprecated(self):
        utils.assert_warns_message(
            FutureWarning,
            "n_pca is deprecated. Setting n_components=2",
            scprep.reduce.pca,
            self.X,
            n_pca=2,
        )
        utils.assert_warns_message(
            FutureWarning,
            "svd_offset is deprecated. Please use `eps` instead.",
            scprep.reduce.pca,
            self.X,
            n_components=2,
            svd_offset=100,
        )
        utils.assert_warns_message(
            FutureWarning,
            "svd_multiples is deprecated. Please use `eps` instead.",
            scprep.reduce.pca,
            self.X,
            n_components=2,
            svd_multiples=100,
        )

    def test_rproj_operator(self):
        pca_op = scprep.reduce.SparseInputPCA(
            n_components=50, eps=0.3, seed=42, method="rproj"
        )
        assert pca_op.fit(self.X_sparse) == pca_op
        Y = pca_op.transform(self.X_sparse)
        assert Y.shape == (self.X_sparse.shape[0], 50)
        assert len(pca_op.singular_values_) == 50
        assert len(pca_op.explained_variance_) == 50
        assert len(pca_op.explained_variance_ratio_) == 50
        assert pca_op.components_.shape == (50, self.X_sparse.shape[1])
        assert pca_op.inverse_transform(pca_op.components_[:, [0]].T).shape == (
            1,
            self.X_sparse.shape[1],
        )

    def test_orth_operator(self):
        pca_op = scprep.reduce.SparseInputPCA(
            n_components=50, eps=0.3, seed=42, method="orth_rproj"
        )
        assert pca_op.fit(self.X_sparse) == pca_op
        Y = pca_op.transform(self.X_sparse)
        assert Y.shape == (self.X_sparse.shape[0], 50)
        assert len(pca_op.singular_values_) == 50
        assert len(pca_op.explained_variance_) == 50
        assert len(pca_op.explained_variance_ratio_) == 50
        assert pca_op.components_.shape == (50, self.X_sparse.shape[1])
        assert pca_op.inverse_transform(pca_op.components_[:, [0]].T).shape == (
            1,
            self.X_sparse.shape[1],
        )
