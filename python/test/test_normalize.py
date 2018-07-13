import numpy as np
from sklearn.preprocessing import normalize
import scprep
from functools import partial
from load_tests import utils, matrix, data


def test_libsize_norm():
    X = data.generate_positive_sparse_matrix()
    median = np.median(X.sum(axis=1))
    Y = normalize(X, 'l1') * median
    utils.assert_all_close(Y.sum(1), np.median(np.sum(X, 1)))
    matrix.check_all_matrix_types(
        X, utils.check_transform_equivalent, Y=Y,
        transform=scprep.normalize.library_size_normalize,
        check=utils.assert_all_close)


def test_batch_mean_center():
    X = data.generate_positive_sparse_matrix()
    sample_idx = np.random.choice([0, 1], X.shape[0], replace=True)
    X[sample_idx == 1] += 1
    Y = X.copy()
    Y[sample_idx == 0] -= np.mean(Y[sample_idx == 0], axis=0)[None, :]
    Y[sample_idx == 1] -= np.mean(Y[sample_idx == 1], axis=0)[None, :]
    utils.assert_all_close(np.mean(Y[sample_idx == 0], axis=0), 0)
    utils.assert_all_close(np.mean(Y[sample_idx == 1], axis=0), 0)
    matrix.check_dense_matrix_types(
        X, utils.check_transform_equivalent, Y=Y,
        transform=partial(
            scprep.normalize.batch_mean_center,
            sample_idx=sample_idx))
    matrix.check_sparse_matrix_types(
        X, utils.check_transform_raises,
        transform=partial(
            scprep.normalize.batch_mean_center,
            sample_idx=sample_idx),
        exception=ValueError)
