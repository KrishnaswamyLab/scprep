import numpy as np
from scipy import sparse
import pandas as pd
from preprocessing import transformation
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
    check_all_matrix_types(X, check_transform_equivalent,
                           Y=Y, transform=transformation.sqrt_transform)


def test_log_transform():
    X = generate_positive_sparse_matrix()
    Y = np.log(X + 1)
    check_all_matrix_types(X, check_transform_equivalent,
                           Y=Y, transform=transformation.log_transform)


def test_arcsinh_transform():
    X = generate_positive_sparse_matrix()
    Y = np.arcsinh(X / 5)
    check_all_matrix_types(X, check_transform_equivalent,
                           Y=Y, transform=transformation.arcsinh_transform,
                           check=all_close)
