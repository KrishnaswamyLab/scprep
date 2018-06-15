# author: Scott Gigante <scott.gigante@yale.edu>
# (C) 2018 Krishnaswamy Lab GPLv2

import numpy as np


def remove_empty(data):

    # Remove empty cells
    print('Removing empty cells')
    cell_sums = data.sum(axis=1)
    to_keep = np.where(cell_sums > 0)[0]
    data = data.ix[
        data.index[to_keep], :].astype(np.float32)

    # Remove empty genes
    print('Removing empty genes')
    gene_sums = data.sum(axis=0)
    to_keep = np.where(gene_sums > 0)[0]
    data = data.ix[:, to_keep].astype(np.float32)
