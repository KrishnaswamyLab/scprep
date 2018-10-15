from tools import utils, matrix, data
import scprep
import numpy as np
from sklearn.utils.testing import assert_raise_message


def test_check_numeric_copy():
    X = data.load_10X()
    matrix.test_all_matrix_types(
        X,
        utils.assert_transform_unchanged,
        transform=scprep.sanitize.check_numeric,
        copy=True)


def test_check_numeric_inplace():
    X = data.load_10X()
    matrix.test_matrix_types(
        X,
        utils.assert_transform_unchanged,
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
        data=X, copy=False)

    class TypeErrorClass(object):

        def astype(self, dtype):
            return
    X = TypeErrorClass()
    assert_raise_message(
        TypeError,
        "astype() got an unexpected keyword argument 'copy'",
        scprep.sanitize.check_numeric,
        data=X, copy=None)


def test_check_numeric_bad_dtype():
    assert_raise_message(
        ValueError,
        "could not convert string to float: ",
        scprep.sanitize.check_numeric,
        np.array(['hello', 'world']))
