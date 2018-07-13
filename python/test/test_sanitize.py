import scprep
import numpy as np
from sklearn.utils.testing import assert_raise_message
from load_tests import utils, matrix, data


def test_check_numeric_copy():
    X = data.load_10X()
    matrix.check_all_matrix_types(
        X,
        utils.check_output_unchanged,
        transform=scprep.sanitize.check_numeric,
        copy=True)


def test_check_numeric_inplace():
    X = data.load_10X()
    matrix.check_matrix_types(
        X,
        utils.check_output_unchanged,
        matrix._scipy_matrix_types +
        matrix._numpy_matrix_types +
        matrix._pandas_dense_matrix_types,
        transform=scprep.sanitize.check_numeric,
        copy=False)
    assert_raise_message(
        TypeError,
        "pd.SparseDataFrame does not support "
        "copy=False. Please use copy=True.",
        scprep.sanitize.check_numeric,
        data=X, copy=True
    )


def test_check_numeric_bad_dtype():
    assert_raise_message(
        ValueError,
        "could not convert string to float: 'hello'",
        scprep.sanitize.check_numeric,
        np.array(['hello', 'world']))
