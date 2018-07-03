# author: Scott Gigante <scott.gigante@yale.edu>
# (C) 2018 Krishnaswamy Lab GPLv2

from __future__ import print_function, division


def check_numeric(data, copy=False):
    return data.astype('float', copy=copy)
