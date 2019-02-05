from tools import utils, matrix, data
import numpy as np
import scprep
from scipy import sparse
import pandas as pd
from sklearn.utils.testing import assert_warns_message, assert_raise_message
import warnings


def test_sqrt_transform():
    X = data.generate_positive_sparse_matrix()
    Y = np.sqrt(X)
    matrix.test_all_matrix_types(
        X, utils.assert_transform_equivalent,
        Y=Y, transform=scprep.transform.sqrt)


def test_log_transform():
    X = data.generate_positive_sparse_matrix()
    Y = np.log10(X + 1)
    matrix.test_all_matrix_types(
        X, utils.assert_transform_equivalent,
        Y=Y, transform=scprep.transform.log,
        base=10)
    Y = np.log(X + 1)
    matrix.test_all_matrix_types(
        X, utils.assert_transform_equivalent,
        Y=Y, transform=scprep.transform.log,
        base='e')
    Y = np.log2(X + 1)
    matrix.test_all_matrix_types(
        X, utils.assert_transform_equivalent,
        Y=Y, transform=scprep.transform.log,
        base=2)
    Y = np.log2(X + 5)
    assert_warns_message(
        RuntimeWarning,
        "log transform on sparse data requires pseudocount = 1",
        scprep.transform.log,
        data=sparse.csr_matrix(X), base=2, pseudocount=5)
    assert_warns_message(
        RuntimeWarning,
        "log transform on sparse data requires pseudocount = 1",
        scprep.transform.log,
        data=pd.SparseDataFrame(X, default_fill_value=0.0),
        base=2, pseudocount=5)
    matrix.test_dense_matrix_types(
        X, utils.assert_transform_equivalent,
        Y=Y, transform=scprep.transform.log,
        base=2, pseudocount=5)


def test_arcsinh_transform():
    X = data.generate_positive_sparse_matrix()
    Y = np.arcsinh(X / 5)
    matrix.test_all_matrix_types(
        X, utils.assert_transform_equivalent,
        Y=Y, transform=scprep.transform.arcsinh,
        check=utils.assert_all_close)
    assert_raise_message(
        ValueError,
        "Expected cofactor > 0 or None. "
        "Got 0",
        scprep.transform.arcsinh, data=X, cofactor=0)


def test_deprecated():
    X = data.generate_positive_sparse_matrix()
    Y = scprep.transform.sqrt(X)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        utils.assert_transform_equivalent(
            X, Y=Y, transform=scprep.transform.sqrt_transform)
    assert_warns_message(
        FutureWarning,
        "scprep.transform.sqrt_transform is deprecated. Please use "
        "scprep.transform.sqrt in future.",
        scprep.transform.sqrt_transform,
        data=X)
    Y = scprep.transform.log(X)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        utils.assert_transform_equivalent(
            X, Y=Y, transform=scprep.transform.log_transform)
    assert_warns_message(
        FutureWarning,
        "scprep.transform.log_transform is deprecated. Please use "
        "scprep.transform.log in future.",
        scprep.transform.log_transform,
        data=X)
    Y = scprep.transform.arcsinh(X)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        utils.assert_transform_equivalent(
            X, Y=Y, transform=scprep.transform.arcsinh_transform)
    assert_warns_message(
        FutureWarning,
        "scprep.transform.arcsinh_transform is deprecated. Please use "
        "scprep.transform.arcsinh in future.",
        scprep.transform.arcsinh_transform, data=X)


def test_sqrt_negative_value():
    X = np.arange(10) * -1
    assert_raise_message(
        ValueError,
        "Cannot square root transform negative values",
        scprep.transform.sqrt, data=X)


def test_log_error():
    X = np.arange(10)
    assert_raise_message(
        ValueError,
        "Required pseudocount + min(data) (-9) > 0. Got pseudocount = 1",
        scprep.transform.log, data=X * -1)
    assert_raise_message(
        ValueError,
        "Expected base in [2, 'e', 10]. Got 0",
        scprep.transform.log, data=X, base=0)
    assert_raise_message(
        ValueError,
        "Expected base in [2, 'e', 10]. Got none",
        scprep.transform.log, data=X, base='none')
