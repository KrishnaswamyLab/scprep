import numpy as np
import pandas as pd
from scipy import sparse


def matrix_any(condition):
    """np.any doesn't handle data frames
    """
    return np.sum(np.sum(condition)) > 0
