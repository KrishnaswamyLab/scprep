from load_tests import utils, matrix, data
import numpy as np
from scipy import stats
from sklearn.utils.testing import assert_warns_message, assert_raise_message
from sklearn.metrics import mutual_info_score
import scprep
from functools import partial


def test_EMD():
    X = data.generate_positive_sparse_matrix(
        shape=(500, 2), seed=42, poisson_mean=5)
    Y = scprep.stats.EMD(X[:, 0], X[:, 1], bins=40)
    assert isinstance(Y, float)
    np.testing.assert_allclose(Y, 0.5537161)
    matrix.check_all_matrix_types(
        X, utils.check_output_equivalent, Y=Y,
        transform=partial(_test_fun_2d, fun=scprep.stats.EMD),
        check=utils.assert_all_close)
    assert_raise_message(
        ValueError, "Expected x and y to be 1D arrays. "
        "Got shapes x {}, y {}".format(X.shape, X[:, 1].shape),
        scprep.stats.EMD, X, X[:, 1])


def shan_entropy(c):
    c_normalized = c / float(np.sum(c))
    return stats.entropy(c_normalized[c_normalized != 0])


def calc_MI(X, Y, bins):
    c_XY = np.histogram2d(X, Y, bins)[0]
    c_X = np.histogram(X, bins)[0]
    c_Y = np.histogram(Y, bins)[0]
    H_X = shan_entropy(c_X)
    H_Y = shan_entropy(c_Y)
    H_XY = shan_entropy(c_XY)
    MI = H_X + H_Y - H_XY
    return MI


def test_mutual_information():
    X = data.generate_positive_sparse_matrix(
        shape=(500, 2), seed=42, poisson_mean=5)
    Y = scprep.stats.mutual_information(X[:, 0], X[:, 1], bins=20)
    assert isinstance(Y, float)
    np.testing.assert_allclose(Y, calc_MI(X[:, 0], X[:, 1], bins=20))
    matrix.check_all_matrix_types(
        X, utils.check_output_equivalent, Y=Y,
        transform=partial(_test_fun_2d, fun=scprep.stats.mutual_information),
        check=utils.assert_all_close, bins=20)


def test_knnDREMI():
    X = data.generate_positive_sparse_matrix(
        shape=(500, 2), seed=42, poisson_mean=5)
    Y = scprep.stats.knnDREMI(X[:, 0], X[:, 1])
    assert isinstance(Y, float)
    np.testing.assert_allclose(Y, 0.16238906)
    scprep.stats.knnDREMI(X[:, 0], X[:, 1], plot=True)
    matrix.check_all_matrix_types(
        X, utils.check_output_equivalent, Y=Y,
        transform=partial(_test_fun_2d, fun=scprep.stats.knnDREMI),
        check=utils.assert_all_close)
    assert_raise_message(
        ValueError, "Expected k as an integer. Got ",
        scprep.stats.knnDREMI, X[:, 0], X[:, 1], k="invalid")
    assert_raise_message(
        ValueError, "Expected n_bins as an integer. Got ",
        scprep.stats.knnDREMI, X[:, 0], X[:, 1], n_bins="invalid")
    assert_raise_message(
        ValueError, "Expected n_mesh as an integer. Got ",
        scprep.stats.knnDREMI, X[:, 0], X[:, 1], n_mesh="invalid")


def _test_fun_2d(X, fun, **kwargs):
    return fun(scprep.utils.select_cols(X, 0), scprep.utils.select_cols(X, 1), **kwargs)
