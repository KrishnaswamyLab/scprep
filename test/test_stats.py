from tools import utils, matrix, data
import numpy as np
from scipy import stats

from sklearn.metrics import mutual_info_score
import scprep
from functools import partial
import warnings
import os
from parameterized import parameterized


def _test_fun_2d(X, fun, **kwargs):
    return fun(
        scprep.select.select_cols(X, idx=0),
        scprep.select.select_cols(X, idx=1),
        **kwargs,
    )


def test_EMD():
    X = data.generate_positive_sparse_matrix(shape=(500, 2), seed=42, poisson_mean=5)
    Y = scprep.stats.EMD(X[:, 0], X[:, 1])
    assert isinstance(Y, float)
    np.testing.assert_allclose(Y, 0.5537161)
    matrix.test_all_matrix_types(
        X,
        utils.assert_transform_equals,
        Y=Y,
        transform=partial(_test_fun_2d, fun=scprep.stats.EMD),
        check=utils.assert_all_close,
    )
    utils.assert_raises_message(
        ValueError,
        "Expected x and y to be 1D arrays. "
        "Got shapes x {}, y {}".format(X.shape, X[:, 1].shape),
        scprep.stats.EMD,
        X,
        X[:, 1],
    )


def test_u_statistic():
    X = data.generate_positive_sparse_matrix(shape=(500, 3), seed=42, poisson_mean=0.2)
    Y = data.generate_positive_sparse_matrix(shape=(500, 3), seed=42, poisson_mean=0.3)
    u_stat = [
        stats.mannwhitneyu(X[:, i], Y[:, i], alternative="two-sided")[0]
        for i in range(X.shape[1])
    ]

    def test_fun(X):
        return scprep.stats.rank_sum_statistic(
            scprep.select.select_rows(X, idx=np.arange(500)),
            scprep.select.select_rows(X, idx=np.arange(500, 1000)),
        )

    matrix.test_all_matrix_types(
        np.vstack([X, Y]),
        utils.assert_transform_equals,
        Y=u_stat,
        transform=test_fun,
        check=utils.assert_all_close,
    )


def test_t_statistic():
    X = data.generate_positive_sparse_matrix(shape=(500, 3), seed=42, poisson_mean=0.2)
    Y = data.generate_positive_sparse_matrix(shape=(500, 3), seed=42, poisson_mean=0.3)
    u_stat = [
        stats.ttest_ind(X[:, i], Y[:, i], equal_var=False)[0] for i in range(X.shape[1])
    ]

    def test_fun(X):
        return scprep.stats.t_statistic(
            scprep.select.select_rows(X, idx=np.arange(500)),
            scprep.select.select_rows(X, idx=np.arange(500, 1000)),
        )

    matrix.test_all_matrix_types(
        np.vstack([X, Y]),
        utils.assert_transform_equals,
        Y=u_stat,
        transform=test_fun,
        check=partial(utils.assert_all_close, rtol=2e-3),
    )


def test_pairwise_correlation():
    def test_fun(X, *args, **kwargs):
        return scprep.stats.pairwise_correlation(
            X, scprep.select.select_cols(X, idx=np.arange(10)), *args, **kwargs
        )

    D = data.generate_positive_sparse_matrix(shape=(500, 100), seed=42, poisson_mean=5)
    Y = test_fun(D)
    assert Y.shape == (D.shape[1], 10)
    assert np.allclose(Y[(np.arange(10), np.arange(10))], 1, atol=0)
    matrix.test_all_matrix_types(
        D,
        utils.assert_transform_equals,
        Y=Y,
        transform=test_fun,
        check=utils.assert_all_close,
    )
    matrix.test_all_matrix_types(
        D,
        utils.assert_transform_equals,
        Y=Y,
        transform=partial(
            scprep.stats.pairwise_correlation,
            Y=scprep.select.select_cols(D, idx=np.arange(10)),
        ),
        check=utils.assert_all_close,
    )

    def test_fun(X, *args, **kwargs):
        return scprep.stats.pairwise_correlation(X=D, Y=X, *args, **kwargs)

    matrix.test_all_matrix_types(
        scprep.select.select_cols(D, idx=np.arange(10)),
        utils.assert_transform_equals,
        Y=Y,
        transform=test_fun,
        check=utils.assert_all_close,
    )


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
    X = data.generate_positive_sparse_matrix(shape=(500, 2), seed=42, poisson_mean=5)
    Y = scprep.stats.mutual_information(X[:, 0], X[:, 1], bins=20)
    assert isinstance(Y, float)
    np.testing.assert_allclose(Y, calc_MI(X[:, 0], X[:, 1], bins=20))
    matrix.test_all_matrix_types(
        X,
        utils.assert_transform_equals,
        Y=Y,
        transform=partial(_test_fun_2d, fun=scprep.stats.mutual_information),
        check=utils.assert_all_close,
        bins=20,
    )


def test_knnDREMI():
    X = data.generate_positive_sparse_matrix(shape=(500, 2), seed=42, poisson_mean=5)
    Y = scprep.stats.knnDREMI(X[:, 0], X[:, 1])
    assert isinstance(Y, float)
    np.testing.assert_allclose(Y, 0.16238906)
    Y2, drevi = scprep.stats.knnDREMI(
        X[:, 0], X[:, 1], plot=True, filename="test.png", return_drevi=True
    )
    assert os.path.isfile("test.png")
    os.remove("test.png")
    assert Y2 == Y
    assert drevi.shape == (20, 20)
    matrix.test_all_matrix_types(
        X,
        utils.assert_transform_equals,
        Y=Y,
        transform=partial(_test_fun_2d, fun=scprep.stats.knnDREMI),
        check=utils.assert_all_close,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        assert scprep.stats.knnDREMI(
            X[:, 0], np.repeat(X[0, 1], X.shape[0]), return_drevi=True
        ) == (0, None)
    utils.assert_raises_message(
        ValueError,
        "Expected k as an integer. Got ",
        scprep.stats.knnDREMI,
        X[:, 0],
        X[:, 1],
        k="invalid",
    )
    utils.assert_raises_message(
        ValueError,
        "Expected n_bins as an integer. Got ",
        scprep.stats.knnDREMI,
        X[:, 0],
        X[:, 1],
        n_bins="invalid",
    )
    utils.assert_raises_message(
        ValueError,
        "Expected n_mesh as an integer. Got ",
        scprep.stats.knnDREMI,
        X[:, 0],
        X[:, 1],
        n_mesh="invalid",
    )
    utils.assert_warns_message(
        UserWarning,
        "Attempting to calculate kNN-DREMI on a constant array. " "Returning `0`",
        scprep.stats.knnDREMI,
        X[:, 0],
        np.zeros_like(X[:, 1]),
    )


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
            **kwargs,
        )

    matrix.test_all_matrix_types(
        X,
        utils.assert_transform_equals,
        Y=Y,
        transform=test_fun,
        check=utils.assert_all_close,
    )
    utils.assert_raises_message(
        ValueError,
        "Expected X and Y to have the same number of columns. "
        "Got shapes {}, {}".format(X.shape, X.iloc[:, :10].shape),
        scprep.stats.mean_difference,
        X,
        X.iloc[:, :10],
    )


@parameterized(
    [
        ("difference", "up"),
        ("difference", "down"),
        ("difference", "both"),
        ("emd", "up"),
        ("emd", "down"),
        ("emd", "both"),
        ("ttest", "up"),
        ("ttest", "down"),
        ("ttest", "both"),
        ("ranksum", "up"),
        ("ranksum", "down"),
        ("ranksum", "both"),
    ]
)
def test_differential_expression(measure, direction):
    X = data.load_10X()
    X = scprep.filter.filter_empty_genes(X)
    result = scprep.stats.differential_expression(
        X.iloc[:20], X.iloc[20:100], measure=measure, direction=direction
    )
    expected_results = {
        ("difference", "up"): ("Gstm5", 16.8125),
        ("difference", "down"): ("Slc2a3", -0.5625),
        ("difference", "both"): ("Gstm5", 16.8125),
        ("emd", "up"): ("Gstm5", 17.5625),
        ("emd", "down"): ("Slc2a3", -0.6875),
        ("emd", "both"): ("Gstm5", 17.5625),
        ("ttest", "up"): ("Trmt1", 2.6335),
        ("ttest", "down"): ("Dhfr", -1.93347),
        ("ttest", "both"): ("Trmt1", 2.6335),
        ("ranksum", "down"): ("Adam30", 796),
        ("ranksum", "up"): ("Gstm5", 339),
        ("ranksum", "both"): ("Adam30", 796),
    }
    assert result.index[0] == expected_results[(measure, direction)][0], result.index[0]
    assert np.allclose(
        result[measure][0], expected_results[(measure, direction)][1]
    ), result[measure][0]
    result_unnamed = scprep.stats.differential_expression(
        X.iloc[:20].sparse.to_coo(),
        X.iloc[20:100].sparse.to_coo(),
        measure=measure,
        direction=direction,
    )
    if direction != "both":
        values = result[measure]
    else:
        values = np.abs(result[measure])

    unique_values = ~np.isin(values, values[values.duplicated()])
    assert np.all(
        X.columns[result_unnamed.index][unique_values] == result.index[unique_values]
    )

    def test_fun(X, **kwargs):
        return scprep.stats.differential_expression(
            scprep.select.select_rows(X, idx=np.arange(20)),
            scprep.select.select_rows(X, idx=np.arange(20, 100)),
            **kwargs,
        )

    def check_fun(Y1, Y2):
        if direction == "both":
            Y1[measure] = np.abs(Y1[measure])
            Y2[measure] = np.abs(Y2[measure])
        np.testing.assert_allclose(Y1[measure], Y2[measure], atol=5e-4)
        Y1 = Y1.sort_index()
        Y2 = Y2.sort_index()
        np.testing.assert_allclose(Y1[measure], Y2[measure], atol=5e-4)

    matrix.test_all_matrix_types(
        X,
        utils.assert_transform_equals,
        Y=result,
        transform=test_fun,
        check=check_fun,
        gene_names=X.columns,
        measure=measure,
        direction=direction,
    )


def test_differential_expression_error():
    X = data.load_10X()
    utils.assert_raises_message(
        ValueError,
        "Expected `direction` in ['up', 'down', 'both']. " "Got invalid",
        scprep.stats.differential_expression,
        X,
        X,
        direction="invalid",
    )
    utils.assert_raises_message(
        ValueError,
        "Expected `measure` in ['difference', 'emd', 'ttest', 'ranksum']. "
        "Got invalid",
        scprep.stats.differential_expression,
        X,
        X,
        measure="invalid",
    )
    utils.assert_raises_message(
        ValueError,
        "Expected `X` and `Y` to be matrices. "
        "Got shapes {}, {}".format(X.shape, X.iloc[0].shape),
        scprep.stats.differential_expression,
        X,
        X.iloc[0],
    )
    utils.assert_raises_message(
        ValueError,
        "Expected gene_names to have length {}. "
        "Got {}".format(X.shape[0], X.shape[0] // 2),
        scprep.stats.differential_expression,
        X.sparse.to_coo(),
        X.sparse.to_coo(),
        gene_names=np.arange(X.shape[0] // 2),
    )
    utils.assert_raises_message(
        ValueError,
        "Expected gene_names to have length {}. "
        "Got {}".format(X.shape[0], X.shape[0] // 2),
        scprep.stats.differential_expression_by_cluster,
        X.sparse.to_coo(),
        np.random.choice(2, X.shape[0], replace=True),
        gene_names=np.arange(X.shape[0] // 2),
    )
    utils.assert_warns_message(
        UserWarning,
        "Input data has inconsistent column names. " "Subsetting to 20 common columns.",
        scprep.stats.differential_expression,
        X,
        X.iloc[:, :20],
    )


def test_differential_expression_by_cluster():
    measure = "difference"
    direction = "up"
    X = data.load_10X()
    np.random.seed(42)
    clusters = np.random.choice(4, X.shape[0], replace=True)
    result = scprep.stats.differential_expression_by_cluster(
        X, clusters, measure=measure, direction=direction
    )
    for cluster in range(4):
        r = scprep.stats.differential_expression(
            scprep.select.select_rows(X, idx=clusters == cluster),
            scprep.select.select_rows(X, idx=clusters != cluster),
            measure=measure,
            direction=direction,
        )
        assert np.all(result[cluster] == r)


def test_differential_expression_by_cluster_subset():
    measure = "difference"
    direction = "up"
    X = data.load_10X()
    np.random.seed(42)
    clusters = np.random.choice(4, X.shape[0], replace=True)
    result = scprep.stats.differential_expression_by_cluster(
        X,
        clusters,
        measure=measure,
        direction=direction,
        gene_names=X.columns[: X.shape[0] // 2],
    )
    for cluster in range(4):
        r = scprep.stats.differential_expression(
            scprep.select.select_rows(X, idx=clusters == cluster),
            scprep.select.select_rows(X, idx=clusters != cluster),
            measure=measure,
            direction=direction,
            gene_names=X.columns[: X.shape[0] // 2],
        )
        assert np.all(result[cluster] == r)
