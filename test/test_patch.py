import numpy as np
import pandas as pd
import scprep  # noqa


def test_pandas_series_rmatmul():
    mat = np.random.random(size=(100, 100))
    arr = np.random.random(size=100)
    df = pd.DataFrame(mat)
    ser = pd.Series(arr)
    np.testing.assert_array_equal(mat @ ser, (df @ ser).values)


def test_pandas_sparse_iloc():
    X = pd.DataFrame([[0, 1, 1], [0, 0, 1], [0, 0, 0]]).astype(
        pd.SparseDtype(float, fill_value=0.0)
    )
    assert np.all(~np.isnan(X.iloc[[0, 1]].to_numpy()))
