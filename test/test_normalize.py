from tools import utils, matrix, data
import numpy as np
from sklearn.preprocessing import normalize

import scprep
from functools import partial
import unittest


class TestNormalize(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.X = data.generate_positive_sparse_matrix()
        self.libsize = self.X.sum(axis=1)
        self.median = np.median(self.libsize)
        self.mean = np.mean(self.X.sum(axis=1))
        self.X_norm = normalize(self.X, "l1")
        self.sample_idx = np.random.choice([0, 1], self.X.shape[0], replace=True)

    def test_libsize_norm_rescale_median(self):
        Y = self.X_norm * self.median
        utils.assert_all_close(Y.sum(1), np.median(np.sum(self.X, 1)))
        Y2, libsize2 = scprep.normalize.library_size_normalize(
            self.X, rescale="median", return_library_size=True
        )
        np.testing.assert_allclose(Y, Y2)
        np.testing.assert_allclose(self.libsize, libsize2)
        matrix.test_all_matrix_types(
            self.X,
            utils.assert_transform_equivalent,
            Y=Y,
            transform=scprep.normalize.library_size_normalize,
            rescale="median",
            check=utils.assert_all_close,
        )

    def test_libsize_norm_return_libsize(self):
        def test_fun(*args, **kwargs):
            return scprep.normalize.library_size_normalize(
                *args, return_library_size=True, **kwargs
            )[1]

        matrix.test_all_matrix_types(
            self.X,
            utils.assert_transform_equals,
            Y=self.libsize,
            transform=test_fun,
            check=utils.assert_all_close,
        )

    def test_libsize_norm_return_libsize_rescale_constant(self):
        def test_fun(*args, **kwargs):
            return scprep.normalize.library_size_normalize(
                *args, return_library_size=True, rescale=1, **kwargs
            )[1]

        matrix.test_all_matrix_types(
            self.X,
            utils.assert_transform_equals,
            Y=self.libsize,
            transform=test_fun,
            check=utils.assert_all_close,
        )

    def test_libsize_norm_rescale_mean(self):
        Y = self.X_norm * self.mean
        utils.assert_all_close(Y.sum(1), np.mean(np.sum(self.X, 1)))
        matrix.test_all_matrix_types(
            self.X,
            utils.assert_transform_equivalent,
            Y=Y,
            transform=scprep.normalize.library_size_normalize,
            check=utils.assert_all_close,
            rescale="mean",
        )

    def test_libsize_norm_rescale_none(self):
        Y = self.X_norm
        matrix.test_all_matrix_types(
            self.X,
            utils.assert_transform_equivalent,
            Y=Y,
            transform=scprep.normalize.library_size_normalize,
            check=utils.assert_all_close,
            rescale=None,
        )

    def test_libsize_norm_rescale_integer(self):
        Y = self.X_norm
        matrix.test_all_matrix_types(
            self.X,
            utils.assert_transform_equivalent,
            Y=Y,
            transform=scprep.normalize.library_size_normalize,
            check=utils.assert_all_close,
            rescale=1,
        )

    def test_libsize_norm_rescale_invalid(self):
        utils.assert_raises_message(
            ValueError,
            "Expected rescale in ['median', 'mean'], a number or `None`. "
            "Got invalid",
            scprep.normalize.library_size_normalize,
            self.X,
            rescale="invalid",
        )

    def test_libsize_norm_median_zero(self):
        X = self.X.copy()
        X[: X.shape[0] // 2 + 1] = 0
        utils.assert_warns_message(
            UserWarning,
            "Median library size is zero. " "Rescaling to mean instead.",
            scprep.normalize.library_size_normalize,
            X,
            rescale="median",
        )

    def test_batch_mean_center(self):
        X = self.X.copy()
        X[self.sample_idx == 1] += 1
        Y = X.copy()
        Y[self.sample_idx == 0] -= np.mean(Y[self.sample_idx == 0], axis=0)[None, :]
        Y[self.sample_idx == 1] -= np.mean(Y[self.sample_idx == 1], axis=0)[None, :]
        utils.assert_all_close(np.mean(Y[self.sample_idx == 0], axis=0), 0)
        utils.assert_all_close(np.mean(Y[self.sample_idx == 1], axis=0), 0)
        matrix.test_dense_matrix_types(
            X,
            utils.assert_transform_equivalent,
            Y=Y,
            transform=scprep.normalize.batch_mean_center,
            sample_idx=self.sample_idx,
        )

    def test_batch_mean_center_sparse(self):
        matrix.test_sparse_matrix_types(
            self.X,
            utils.assert_transform_raises,
            transform=scprep.normalize.batch_mean_center,
            sample_idx=self.sample_idx,
            exception=ValueError,
        )

    def test_batch_mean_center_one_sample(self):
        Y = self.X.copy()
        Y -= np.mean(Y, axis=0)[None, :]
        matrix.test_dense_matrix_types(
            self.X,
            utils.assert_transform_equivalent,
            Y=Y,
            transform=scprep.normalize.batch_mean_center,
        )

    def test_batch_mean_center_sparse_one_sample(self):
        matrix.test_sparse_matrix_types(
            self.X,
            utils.assert_transform_raises,
            transform=scprep.normalize.batch_mean_center,
            exception=ValueError,
        )
