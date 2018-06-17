from preprocessing import io
from sklearn.utils.testing import assert_warns_message
import pandas as pd
import numpy as np
import os

if os.getcwd().strip("/").endswith("test"):
    data_dir = os.path.join("..", "..", "data", "test_data")
else:
    data_dir = os.path.join("..", "data", "test_data")


def load_10X(**kwargs):
    return io.load_10X(os.path.join(data_dir, "test_10X"), **kwargs)

# TODO: write tests
