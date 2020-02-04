import scprep
import numpy as np
import pandas as pd
from pandas.core.internals.blocks import ExtensionBlock


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


class CustomBlock(ExtensionBlock):
    _holder = np.ndarray


def test_fill_value():
    values = pd.Series(np.arange(3), dtype=pd.UInt16Dtype())
    custom_block = CustomBlock(values, placement=slice(1, 2))
    assert pd.isna(custom_block.fill_value)
    values = pd.Series(np.arange(3), dtype=pd.SparseDtype(float, 0.0))
    custom_block = CustomBlock(values, placement=slice(1, 2))
    assert not pd.isna(custom_block.fill_value)
