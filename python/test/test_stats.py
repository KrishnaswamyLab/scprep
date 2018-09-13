import numpy as np
from sklearn.utils.testing import assert_warns_message, assert_raise_message
import scprep
from functools import partial
from load_tests import utils, matrix, data

def test_EMD():
    X = data.generate_positive_sparse_matrix(shape=(500,2))
    Y = scprep.stats.EMD(X[:,0], X[:,1])
    matrix.check_all_matrix_types(
        X, utils.check_transform_equivalent, Y=Y,
        transform=scprep.stats.EMD,
        check=utils.assert_all_close)

def test_mutual_information():
    X = data.generate_positive_sparse_matrix(shape=(500,2))
    Y = scprep.stats.mutual_information(X[:,0], X[:,1])
    matrix.check_all_matrix_types(
        X, utils.check_transform_equivalent, Y=Y,
        transform=scprep.stats.mutual_information,
        check=utils.assert_all_close)

def test_knnDREMI():
    X = data.generate_positive_sparse_matrix(shape=(500,2))
    Y = scprep.stats.knnDREMI(X[:,0], X[:,1])
    matrix.check_all_matrix_types(
        X, utils.check_transform_equivalent, Y=Y,
        transform=scprep.stats.knnDREMI,
        check=utils.assert_all_close)
