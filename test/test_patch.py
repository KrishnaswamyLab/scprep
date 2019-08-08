import scprep
import numpy as np
import pandas as pd


def test_pandas_series_rmatmul():
    mat = np.random.random(size=(100, 100))
    arr = np.random.random(size=100)
    df = pd.DataFrame(mat)
    ser = pd.Series(arr)
    np.testing.assert_array_equal(mat @ ser, (df @ ser).values)
