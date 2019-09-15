from tools import utils, matrix, data
import numpy as np
from scipy import stats
from sklearn.utils.testing import assert_warns_message, assert_raise_message
from sklearn.metrics import mutual_info_score
import scprep
from functools import partial
import warnings
from parameterized import parameterized


def _test_fun_2d(X, fun, **kwargs):
    return fun(scprep.select.select_cols(X, idx=0), scprep.select.select_cols(X, idx=1), **kwargs)


def test_EMD():
    X = data.generate_positive_sparse_matrix(
        shape=(500, 2), seed=42, poisson_mean=5)
    Y = scprep.stats.EMD(X[:, 0], X[:, 1])
    assert isinstance(Y, float)
    np.testing.assert_allclose(Y, 0.5537161)
    matrix.test_all_matrix_types(
        X, utils.assert_transform_equals, Y=Y,
        transform=partial(_test_fun_2d, fun=scprep.stats.EMD),
        check=utils.assert_all_close)
    assert_raise_message(
        ValueError, "Expected x and y to be 1D arrays. "
        "Got shapes x {}, y {}".format(X.shape, X[:, 1].shape),
        scprep.stats.EMD, X, X[:, 1])


def test_pairwise_correlation():
    def test_fun(X, *args, **kwargs):
        return scprep.stats.pairwise_correlation(
            X,
            scprep.select.select_cols(X, idx=np.arange(10)),
            *args, **kwargs)
    D = data.generate_positive_sparse_matrix(
        shape=(500, 100), seed=42, poisson_mean=5)
    Y = test_fun(D)
    assert Y.shape == (D.shape[1], 10)
    assert np.allclose(Y[(np.arange(10), np.arange(10))], 1, atol=0)
    matrix.test_all_matrix_types(
        D, utils.assert_transform_equals, Y=Y,
        transform=test_fun,
        check=utils.assert_all_close)
    matrix.test_all_matrix_types(
        D, utils.assert_transform_equals, Y=Y,
        transform=partial(scprep.stats.pairwise_correlation,
                          Y=scprep.select.select_cols(D, idx=np.arange(10))),
        check=utils.assert_all_close)

    def test_fun(X, *args, **kwargs):
        return scprep.stats.pairwise_correlation(
            X=D,
            Y=X,
            *args, **kwargs)
    matrix.test_all_matrix_types(
        scprep.select.select_cols(D, idx=np.arange(10)),
        utils.assert_transform_equals, Y=Y,
        transform=test_fun, check=utils.assert_all_close)


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
    matrix.test_all_matrix_types(
        X, utils.assert_transform_equals, Y=Y,
        transform=partial(_test_fun_2d, fun=scprep.stats.mutual_information),
        check=utils.assert_all_close, bins=20)


def test_knnDREMI():
    X = data.generate_positive_sparse_matrix(
        shape=(500, 2), seed=42, poisson_mean=5)
    Y = scprep.stats.knnDREMI(X[:, 0], X[:, 1])
    assert isinstance(Y, float)
    np.testing.assert_allclose(Y, 0.16238906)
    Y2, drevi = scprep.stats.knnDREMI(X[:, 0], X[:, 1],
                                      plot=True, filename="test.png",
                                      return_drevi=True)
    assert Y2 == Y
    assert drevi.shape == (20, 20)
    matrix.test_all_matrix_types(
        X, utils.assert_transform_equals, Y=Y,
        transform=partial(_test_fun_2d, fun=scprep.stats.knnDREMI),
        check=utils.assert_all_close)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        assert scprep.stats.knnDREMI(X[:, 0], np.repeat(X[0, 1], X.shape[0]),
                                     return_drevi=True) == (0, None)
    assert_raise_message(
        ValueError, "Expected k as an integer. Got ",
        scprep.stats.knnDREMI, X[:, 0], X[:, 1], k="invalid")
    assert_raise_message(
        ValueError, "Expected n_bins as an integer. Got ",
        scprep.stats.knnDREMI, X[:, 0], X[:, 1], n_bins="invalid")
    assert_raise_message(
        ValueError, "Expected n_mesh as an integer. Got ",
        scprep.stats.knnDREMI, X[:, 0], X[:, 1], n_mesh="invalid")
    assert_warns_message(
        UserWarning,
        "Attempting to calculate kNN-DREMI on a constant array. "
        "Returning `0`", scprep.stats.knnDREMI, X[:, 0],
        np.zeros_like(X[:, 1]))


def test_mean_difference():
    X = data.load_10X()
    X = scprep.filter.filter_empty_genes(X)
    Y = scprep.stats.mean_difference(X.iloc[:20], X.iloc[20:100])
    assert np.allclose(np.max(Y), 16.8125)
    assert np.allclose(np.min(Y), -0.5625)
    def test_fun(X, **kwargs):
        return scprep.stats.mean_difference(
            scprep.select.select_rows(X, idx=np.arange(20)),
            scprep.select.select_rows(X, idx=np.arange(20, 100)),
        **kwargs)
    matrix.test_all_matrix_types(
        X, utils.assert_transform_equals, Y=Y,
        transform=test_fun,
        check=utils.assert_all_close)
    assert_raise_message(
        ValueError,
        "Expected X and Y to have the same number of columns. "
        "Got shapes {}, {}".format(X.shape, X.iloc[:,:10].shape),
        scprep.stats.mean_difference,
        X, X.iloc[:,:10])


@parameterized([('difference', 'up'), ('difference', 'down'), ('difference', 'both'),
               ('emd', 'up'), ('emd', 'down'), ('emd', 'both')])
def test_differential_expression(measure, direction):
    X = data.load_10X()
    X = scprep.filter.filter_empty_genes(X)
    result = scprep.stats.differential_expression(X.iloc[:20], X.iloc[20:100],
                                                 measure=measure, direction=direction)
    expected_results = {('difference', 'up') : ('Gstm5', 16.8125),
                        ('difference', 'down') : ('Slc2a3', -0.5625),
                        ('difference', 'both') : ('Gstm5', 16.8125),
                        ('emd', 'up') : ('Gstm5', 17.5625),
                        ('emd', 'down') : ('Slc2a3', -0.6875),
                        ('emd', 'both') : ('Gstm5', 17.5625)}
    assert result['gene'][0] == expected_results[(measure, direction)][0], result['gene'][0]
    assert np.allclose(result[measure][0],
                       expected_results[(measure, direction)][1])
    result_unnamed = scprep.stats.differential_expression(X.iloc[:20].to_coo(), X.iloc[20:100].to_coo(),
                                                         measure=measure, direction=direction)
    unique_results = ~np.isin(result[measure], result[measure][result[measure].duplicated()])
    assert np.all(X.columns[result_unnamed['gene']][unique_results] == result['gene'][unique_results])
    def test_fun(X, **kwargs):
        return scprep.stats.differential_expression(
            scprep.select.select_rows(X, idx=np.arange(20)),
            scprep.select.select_rows(X, idx=np.arange(20, 100)),
        **kwargs)
    def check_fun(Y1, Y2):
        if direction == 'both':
            Y1[measure] = np.abs(Y1[measure])
            Y2[measure] = np.abs(Y2[measure])
        np.testing.assert_allclose(Y1[measure], Y2[measure], atol=5e-4)
        Y1 = Y1.sort_values('gene')
        Y2 = Y2.sort_values('gene')
        np.testing.assert_allclose(Y1[measure], Y2[measure], atol=5e-4)
    matrix.test_all_matrix_types(
        X, utils.assert_transform_equals, Y=result,
        transform=test_fun,
        check=check_fun,
        gene_names=X.columns,
        measure=measure, direction=direction)


def test_differential_expression_error():
    X = data.load_10X()
    assert_raise_message(
        ValueError, "Expected `direction` in ['up', 'down', 'both']. "
        "Got invalid", scprep.stats.differential_expression,
        X, X, direction='invalid')
    assert_raise_message(
        ValueError, "Expected `measure` in ['difference', 'emd']. "
        "Got invalid", scprep.stats.differential_expression,
        X, X, measure='invalid')
    assert_raise_message(
        ValueError, "Expected `X` and `Y` to be matrices. "
        "Got shapes {}, {}".format(X.shape, X.iloc[0].shape),
        scprep.stats.differential_expression,
        X, X.iloc[0])
    assert_raise_message(
        ValueError, "Expected gene_names to have length {}. "
        "Got {}".format(X.shape[0], X.shape[0]//2),
        scprep.stats.differential_expression,
        X.to_coo(), X.to_coo(), gene_names=np.arange(X.shape[0]//2))
    assert_raise_message(
        ValueError, "Expected gene_names to have length {}. "
        "Got {}".format(X.shape[0], X.shape[0]//2),
        scprep.stats.differential_expression_by_cluster,
        X.to_coo(), np.random.choice(2, X.shape[0], replace=True),
        gene_names=np.arange(X.shape[0]//2))
    assert_warns_message(
        UserWarning, "Input data has inconsistent column names. "
        "Subsetting to 20 common columns.",
        scprep.stats.differential_expression,
        X, X.iloc[:,:20])


@parameterized([('difference', 'up'), ('difference', 'down'), ('difference', 'both'),
               ('emd', 'up'), ('emd', 'down'), ('emd', 'both')])
def test_differential_expression_by_cluster(measure, direction):
    X = data.load_10X()
    np.random.seed(42)
    clusters = np.random.choice(4, X.shape[0], replace=True)
    result = scprep.stats.differential_expression_by_cluster(
        X, clusters,
        measure=measure, direction=direction)
    for cluster in range(4):
        r = scprep.stats.differential_expression(
            scprep.select.select_rows(X, idx=clusters==cluster),
            scprep.select.select_rows(X, idx=clusters!=cluster),
        measure=measure, direction=direction)
        assert np.all(result[cluster] == r)
