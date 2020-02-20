from tools import utils, matrix, data
import scprep
import numpy as np
import pandas as pd
import warnings


def test_check_numeric_copy():
    X = data.load_10X()
    matrix.test_all_matrix_types(
        X,
        utils.assert_transform_unchanged,
        transform=scprep.sanitize.check_numeric,
        copy=True,
    )


def test_check_numeric_inplace():
    X = data.load_10X()
    matrix.test_matrix_types(
        X,
        utils.assert_transform_unchanged,
        matrix._scipy_matrix_types
        + matrix._numpy_matrix_types
        + matrix._pandas_dense_matrix_types
        + [matrix.SparseDataFrame],
        transform=scprep.sanitize.check_numeric,
        copy=False,
    )
    if matrix._pandas_0:
        matrix._ignore_pandas_sparse_warning()
        utils.assert_raises_message(
            TypeError,
            "pd.SparseDataFrame does not support " "copy=False. Please use copy=True.",
            scprep.sanitize.check_numeric,
            data=matrix.SparseDataFrame_deprecated(X),
            copy=False,
        )
        matrix._reset_warnings()

    class TypeErrorClass(object):
        def astype(self, dtype):
            return

    X = TypeErrorClass()
    utils.assert_raises_message(
        TypeError,
        "astype() got an unexpected keyword argument 'copy'",
        scprep.sanitize.check_numeric,
        data=X,
        copy=None,
    )


def test_check_numeric_bad_dtype():
    utils.assert_raises_message(
        ValueError,
        "could not convert string to float: ",
        scprep.sanitize.check_numeric,
        np.array(["hello", "world"]),
    )


def test_check_index():
    X = data.load_10X()
    scprep.sanitize.check_index(X)
    with utils.assert_warns_message(
        RuntimeWarning,
        "Renamed 2 copies of index GATGAGGCATTTCAGG-1 to (GATGAGGCATTTCAGG-1, GATGAGGCATTTCAGG-1.1)",
    ):
        scprep.sanitize.check_index(X.iloc[[0, 0]])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        Y = scprep.sanitize.check_index(X.iloc[[0, 0]])
        assert isinstance(Y.loc["GATGAGGCATTTCAGG-1"], pd.Series)
        assert isinstance(Y.loc["GATGAGGCATTTCAGG-1.1"], pd.Series)
        Y = X.iloc[[0, 0]]
        scprep.sanitize.check_index(Y)
        assert isinstance(Y.loc["GATGAGGCATTTCAGG-1"], pd.Series)
        assert isinstance(Y.loc["GATGAGGCATTTCAGG-1.1"], pd.Series)
        Y = X.iloc[[0, 0]]
        scprep.sanitize.check_index(Y, copy=True)
        assert isinstance(Y.loc["GATGAGGCATTTCAGG-1"], pd.DataFrame)
        assert Y.loc["GATGAGGCATTTCAGG-1"].shape[0] == 2
    with utils.assert_warns_message(
        RuntimeWarning,
        "Renamed 3 copies of index GTCATTTCATCTCGCT-1 to (GTCATTTCATCTCGCT-1, GTCATTTCATCTCGCT-1.1, GTCATTTCATCTCGCT-1.2)",
    ):
        scprep.sanitize.check_index(X.iloc[[1, 1, 1]])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        Y = scprep.sanitize.check_index(X.iloc[[1, 1, 1]])
        assert isinstance(Y.loc["GTCATTTCATCTCGCT-1"], pd.Series)
        assert isinstance(Y.loc["GTCATTTCATCTCGCT-1.1"], pd.Series)
        assert isinstance(Y.loc["GTCATTTCATCTCGCT-1.2"], pd.Series)
        Y = X.iloc[[1, 1, 1]]
        scprep.sanitize.check_index(Y)
        assert isinstance(Y.loc["GTCATTTCATCTCGCT-1"], pd.Series)
        assert isinstance(Y.loc["GTCATTTCATCTCGCT-1.1"], pd.Series)
        assert isinstance(Y.loc["GTCATTTCATCTCGCT-1.2"], pd.Series)
        Y = X.iloc[[1, 1, 1]]
        scprep.sanitize.check_index(Y, copy=True)
        assert isinstance(Y.loc["GTCATTTCATCTCGCT-1"], pd.DataFrame)
        assert Y.loc["GTCATTTCATCTCGCT-1"].shape[0] == 3


def test_check_index_multiindex():
    X = data.load_10X()
    X["i"] = [i for i in range(X.shape[0])]
    X["i+1"] = [i + 1 for i in range(X.shape[0])]
    X = X.set_index(["i", "i+1"])
    scprep.sanitize.check_index(X)
    with utils.assert_warns_message(
        RuntimeWarning, "Renamed 2 copies of index (0, 1) to ((0, 1), (0, '1.1'))"
    ):
        scprep.sanitize.check_index(X.iloc[[0, 0]])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        Y = scprep.sanitize.check_index(X.iloc[[0, 0]])
        assert isinstance(Y.loc[(0, 1)], pd.Series)
        assert isinstance(Y.loc[(0, "1.1")], pd.Series)
        Y = X.iloc[[0, 0]]
        scprep.sanitize.check_index(Y)
        assert isinstance(Y.loc[(0, 1)], pd.Series)
        assert isinstance(Y.loc[(0, "1.1")], pd.Series)
        Y = X.iloc[[0, 0]]
        scprep.sanitize.check_index(Y, copy=True)
        assert isinstance(Y.loc[(0, 1)], pd.DataFrame)
        assert Y.loc[(0, 1)].shape[0] == 2
    with utils.assert_warns_message(
        RuntimeWarning,
        "Renamed 3 copies of index (1, 2) to ((1, 2), (1, '2.1'), (1, '2.2'))",
    ):
        scprep.sanitize.check_index(X.iloc[[1, 1, 1]])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        Y = scprep.sanitize.check_index(X.iloc[[1, 1, 1]])
        assert isinstance(Y.loc[(1, 2)], pd.Series)
        assert isinstance(Y.loc[(1, "2.1")], pd.Series)
        assert isinstance(Y.loc[(1, "2.2")], pd.Series)
        Y = X.iloc[[1, 1, 1]]
        scprep.sanitize.check_index(Y)
        assert isinstance(Y.loc[(1, 2)], pd.Series)
        assert isinstance(Y.loc[(1, "2.1")], pd.Series)
        assert isinstance(Y.loc[(1, "2.2")], pd.Series)
        Y = X.iloc[[1, 1, 1]]
        scprep.sanitize.check_index(Y, copy=True)
        assert isinstance(Y.loc[(1, 2)], pd.DataFrame)
        assert Y.loc[(1, 2)].shape[0] == 3


def test_check_index_ndarray():
    with utils.assert_warns_message(
        UserWarning, "scprep.sanitize.check_index only accepts pandas input"
    ):
        scprep.sanitize.check_index(np.array([0, 1]))
