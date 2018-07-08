import numpy as np
from scipy import sparse
import pandas as pd
import scprep
from load_tests.utils import (
    all_equal, all_close,
    check_all_matrix_types,
    generate_positive_sparse_matrix,
    matrix_class_equivalent,
    check_transform_equivalent
)


def test_sqrt_transform():
    X = generate_positive_sparse_matrix()
    Y = np.sqrt(X)
    check_all_matrix_types(
        X, check_transform_equivalent,
        Y=Y, transform=scprep.transform.sqrt)


def test_log_transform():
    X = generate_positive_sparse_matrix()
    Y = np.log10(X + 1)
    check_all_matrix_types(
        X, check_transform_equivalent,
        Y=Y, transform=scprep.transform.log,
        base=10)
    Y = np.log(X + 1)
    check_all_matrix_types(
        X, check_transform_equivalent,
        Y=Y, transform=scprep.transform.log,
        base='e')
    Y = np.log2(X + 1)
    check_all_matrix_types(
        X, check_transform_equivalent,
        Y=Y, transform=scprep.transform.log,
        base=2)
    Y = np.log2(X + 5)
    check_all_matrix_types(
        X, check_transform_equivalent,
        Y=Y, transform=scprep.transform.log,
        base=2)


def test_arcsinh_transform():
    X = generate_positive_sparse_matrix()
    Y = np.arcsinh(X / 5)
    check_all_matrix_types(
        X, check_transform_equivalent,
        Y=Y, transform=scprep.transform.arcsinh,
        check=all_close)
