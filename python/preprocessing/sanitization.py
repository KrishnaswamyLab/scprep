# author: Scott Gigante <scott.gigante@yale.edu>
# (C) 2018 Krishnaswamy Lab GPLv2

import pandas as pd
import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    pass


def remove_empty_genes(data):
    gene_sums = data.sum(axis=0)
    to_keep = gene_sums > 0
    if isinstance(data, pd.DataFrame):
        data = data[data.columns[to_keep].tolist()]
    else:
        data = data[:, to_keep]
    return data


def remove_empty_cells(data):
    cell_sums = library_size(data)
    to_keep = cell_sums > 0
    if isinstance(data, pd.DataFrame):
        data = data.loc[to_keep]
    else:
        data = data[to_keep, :]
    return data


def library_size(data):
    if isinstance(data, pd.SparseDataFrame):
        # densifies matrix if you take the sum
        cell_sums = pd.Series(
            np.array(data.to_coo().sum(axis=1)).reshape(-1),
            index=data.index)
    else:
        cell_sums = data.sum(axis=1)
    return cell_sums


def plot_library_size(data, bins=30, cutoff=None):
    try:
        plt
    except NameError:
        print("matplotlib not found. "
              "Please install it with e.g. `pip install --user matplotlib`")
    cell_sums = library_size(data)
    plt.hist(cell_sums, bins=bins)
    if cutoff is not None:
        plt.vline(cutoff, color='red')
    plt.show()


def filter_library_size(data, libsize=2000):
    cell_sums = library_size(data)
    to_keep = cell_sums > libsize
    if isinstance(data, pd.DataFrame):
        data = data.loc[to_keep]
    else:
        data = data[:, to_keep]
    return data
