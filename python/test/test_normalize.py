import numpy as np
from sklearn.preprocessing import normalize
import scpreprocess
from functools import partial
from load_tests.utils import (
    all_close,
    check_all_matrix_types,
    check_dense_matrix_types,
    check_sparse_matrix_types,
    generate_positive_sparse_matrix,
    check_transform_equivalent,
    check_transform_raises
)


def test_libsize_norm():
    X = generate_positive_sparse_matrix()
    median = np.median(X.sum(axis=1))
    Y = normalize(X, 'l1') * median
    assert np.allclose(Y.sum(1), np.median(np.sum(X, 1)))
    check_all_matrix_types(
        X, check_transform_equivalent, Y=Y,
        transform=scpreprocess.normalize.library_size_normalize,
        check=all_close)


def test_batch_mean_center():
    X = generate_positive_sparse_matrix()
    sample_idx = np.random.choice([0, 1], X.shape[0], replace=True)
    X[sample_idx == 1] += 1
    Y = X.copy()
    Y[sample_idx == 0] -= np.mean(Y[sample_idx == 0], axis=0)[None, :]
    Y[sample_idx == 1] -= np.mean(Y[sample_idx == 1], axis=0)[None, :]
    assert np.allclose(np.mean(Y[sample_idx == 0], axis=0), 0)
    assert np.allclose(np.mean(Y[sample_idx == 1], axis=0), 0)
    check_dense_matrix_types(X, check_transform_equivalent, Y=Y,
                             transform=partial(
                                 scpreprocess.normalize.batch_mean_center,
                                 sample_idx=sample_idx))
    check_sparse_matrix_types(X, check_transform_raises,
                              transform=partial(
                                  scpreprocess.normalize.batch_mean_center,
                                  sample_idx=sample_idx),
                              exception=ValueError)
