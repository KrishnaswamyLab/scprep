import scprep
from sklearn import decomposition
from load_tests import utils, matrix, data
from functools import partial


def test_pca():
    X = data.generate_positive_sparse_matrix(shape=[50, 1000])
    Y = decomposition.PCA(100, random_state=42).fit_transform(X)
    matrix.check_dense_matrix_types(
        X, utils.check_output_equivalent,
        Y=Y, transform=scprep.reduce.pca,
        seed=42)
    Y = decomposition.PCA(50, svd_solver='full').fit_transform(X)
    matrix.check_sparse_matrix_types(
        X, utils.check_output_equivalent,
        Y=Y, transform=scprep.reduce.pca,
        check=partial(utils.assert_all_close, rtol=1e-3, atol=1e-5),
        n_pca=50, svd_multiples=8, seed=42)
