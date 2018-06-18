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
    try:
        cell_sums = data.sum(axis=1)
    except MemoryError:
        # pandas sparse dataframes do weird stuff
        if False:  # isinstance(data, pd.SparseDataFrame):
            split = np.arange(0, data.shape[0], data.shape[0] // 10)
            split = np.concatenate([split, [data.shape[0]]])
            cell_sums = []
            for i in range(len(split)):
                cell_sums.append(library_size(
                    data.iloc[split[i]:split[i + 1]]))
            cell_sums = pd.concat(cell_sums)
        else:
            raise
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
