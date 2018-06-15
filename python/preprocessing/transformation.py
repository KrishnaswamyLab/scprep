# author: Scott Gigante <scott.gigante@yale.edu>
# (C) 2018 Krishnaswamy Lab GPLv2

import numpy as np
from scipy import sparse


def sqrt_transform():
    pass


def log_transform(data, pseudocount=1):
    pass


def arcsinh_transform(data, cofactor=5):
    if cofactor > 0:
        data = np.arcsinh(data / cofactor)
