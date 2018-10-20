from tools import utils, matrix, data
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.utils.testing import assert_warns_message, assert_raise_message
import scprep
from functools import partial


def test_libsize_norm():
    X = data.generate_positive_sparse_matrix()
    libsize = X.sum(axis=1)
    median = np.median(libsize)
    Y = normalize(X, 'l1') * median
    utils.assert_all_close(Y.sum(1), np.median(np.sum(X, 1)))
    Y2, libsize2 = scprep.normalize.library_size_normalize(
        X, return_library_size=True)
    assert np.all(Y == Y2)
    assert np.all(libsize == libsize2)
    matrix.test_all_matrix_types(
        X, utils.assert_transform_equivalent, Y=Y,
        transform=scprep.normalize.library_size_normalize,
        check=utils.assert_all_close)
    mean = np.mean(X.sum(axis=1))
    X = data.generate_positive_sparse_matrix()
    Y = normalize(X, 'l1') * mean
    utils.assert_all_close(Y.sum(1), np.mean(np.sum(X, 1)))
    matrix.test_all_matrix_types(
        X, utils.assert_transform_equivalent, Y=Y,
        transform=scprep.normalize.library_size_normalize,
        check=utils.assert_all_close, rescale='mean')
    X = data.generate_positive_sparse_matrix()
    Y = normalize(X, 'l1')
    matrix.test_all_matrix_types(
        X, utils.assert_transform_equivalent, Y=Y,
        transform=scprep.normalize.library_size_normalize,
        check=utils.assert_all_close, rescale=None)
    matrix.test_all_matrix_types(
        X, utils.assert_transform_equivalent, Y=Y,
        transform=scprep.normalize.library_size_normalize,
        check=utils.assert_all_close, rescale=1)
    assert_raise_message(
        ValueError,
        "Expected rescale in ['median', 'mean'], a number or `None`. "
        "Got invalid",
        scprep.normalize.library_size_normalize,
        X, rescale='invalid')
    X[:X.shape[0] // 2 + 1] = 0
    assert_warns_message(
        UserWarning,
        "Median library size is zero. "
        "Rescaling to mean instead.",
        scprep.normalize.library_size_normalize,
        X, rescale='median')


def test_batch_mean_center():
    X = data.generate_positive_sparse_matrix()
    sample_idx = np.random.choice([0, 1], X.shape[0], replace=True)
    X[sample_idx == 1] += 1
    Y = X.copy()
    Y[sample_idx == 0] -= np.mean(Y[sample_idx == 0], axis=0)[None, :]
    Y[sample_idx == 1] -= np.mean(Y[sample_idx == 1], axis=0)[None, :]
    utils.assert_all_close(np.mean(Y[sample_idx == 0], axis=0), 0)
    utils.assert_all_close(np.mean(Y[sample_idx == 1], axis=0), 0)
    matrix.test_dense_matrix_types(
        X, utils.assert_transform_equivalent, Y=Y,
        transform=partial(
            scprep.normalize.batch_mean_center,
            sample_idx=sample_idx))
    matrix.test_sparse_matrix_types(
        X, utils.assert_transform_raises,
        transform=partial(
            scprep.normalize.batch_mean_center,
            sample_idx=sample_idx),
        exception=ValueError)
    X = data.generate_positive_sparse_matrix()
    Y = X.copy()
    Y -= np.mean(Y, axis=0)[None, :]
    matrix.test_dense_matrix_types(
        X, utils.assert_transform_equivalent, Y=Y,
        transform=partial(
            scprep.normalize.batch_mean_center))
    matrix.test_sparse_matrix_types(
        X, utils.assert_transform_raises,
        transform=partial(
            scprep.normalize.batch_mean_center),
        exception=ValueError)
