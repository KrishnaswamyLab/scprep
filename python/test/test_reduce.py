from tools import utils, matrix, data
import scprep
from sklearn import decomposition
from functools import partial
import numpy as np


def test_pca():
    X = data.generate_positive_sparse_matrix(shape=[50, 3000])
    X = np.vstack([X, 10 * X, -10 * X])
    Y = decomposition.PCA(100, random_state=42).fit_transform(X)
    matrix.test_dense_matrix_types(
        X, utils.assert_transform_equals,
        Y=Y, transform=scprep.reduce.pca,
        n_components=100, seed=42)
    Y = decomposition.PCA(50, svd_solver='full').fit_transform(X)
    matrix.test_sparse_matrix_types(
        X, utils.assert_transform_equals,
        Y=Y, transform=scprep.reduce.pca,
        check=partial(utils.assert_all_close, rtol=1e-3, atol=1e-5),
        n_components=50, eps=0.15, seed=42)
