import numpy as np
import scprep
from scipy import sparse
import pandas as pd
from sklearn.utils.testing import assert_warns_message
from load_tests.utils import (
    assert_all_close,
    check_all_matrix_types,
    check_dense_matrix_types,
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
    assert_warns_message(
        RuntimeWarning,
        "log transform on sparse data requires pseudocount=1",
        scprep.transform.log,
        data=sparse.csr_matrix(X), base=2, pseudocount=5)
    assert_warns_message(
        RuntimeWarning,
        "log transform on sparse data requires pseudocount=1",
        scprep.transform.log,
        data=pd.SparseDataFrame(X, default_fill_value=0.0),
        base=2, pseudocount=5)
    check_dense_matrix_types(
        X, check_transform_equivalent,
        Y=Y, transform=scprep.transform.log,
        base=2, pseudocount=5)


def test_arcsinh_transform():
    X = generate_positive_sparse_matrix()
    Y = np.arcsinh(X / 5)
    check_all_matrix_types(
        X, check_transform_equivalent,
        Y=Y, transform=scprep.transform.arcsinh,
        check=assert_all_close)


def test_deprecated():
    X = generate_positive_sparse_matrix()
    Y = scprep.transform.sqrt(X)
    check_transform_equivalent(Y=Y, transform=scprep.transform.sqrt_transform)
    assert_warns_message(
        FutureWarning,
        "scprep.transform.sqrt_transform is deprecated. Please use "
        "scprep.transform.sqrt in future.",
        scprep.transform.sqrt_transform,
        data=X)
    Y = scprep.transform.log(X)
    check_transform_equivalent(Y=Y, transform=scprep.transform.log_transform)
    assert_warns_message(
        FutureWarning,
        "scprep.transform.log_transform is deprecated. Please use "
        "scprep.transform.log in future.",
        scprep.transform.log_transform,
        data=X)
    Y = scprep.transform.arcsinh(X)
    check_transform_equivalent(
        Y=Y, transform=scprep.transform.arcsinh_transform)
    assert_warns_message(
        FutureWarning,
        "scprep.transform.arcsinh_transform is deprecated. Please use "
        "scprep.transform.arcsinh in future.",
        scprep.transform.arcsinh_transform,
        data=X)
