# scprep

[![Latest PyPI version](https://img.shields.io/pypi/v/scprep.svg)](https://pypi.org/project/scprep/)
[![Travis CI Build](https://api.travis-ci.com/KrishnaswamyLab/scprep.svg?branch=master)](https://travis-ci.com/KrishnaswamyLab/scprep)
[![Read the Docs](https://img.shields.io/readthedocs/scprep.svg)](https://scprep.readthedocs.io/)
[![Coverage Status](https://coveralls.io/repos/github/KrishnaswamyLab/scprep/badge.svg?branch=master)](https://coveralls.io/github/KrishnaswamyLab/scprep?branch=master)
[![Twitter](https://img.shields.io/twitter/follow/KrishnaswamyLab.svg?style=social&label=Follow)](https://twitter.com/KrishnaswamyLab)

Tools for loading and preprocessing biological matrices in Python and MATLAB.

## Python

### Installation

preprocessing is available on `pip`. Install by running the following in a terminal::

        pip install --user scprep

### Quick Start

You can use `scprep` with your single cell data as follows::

    import scprep
    # Load data
    data_path = "~/mydata/my_10X_data"
    data = scprep.io.load_10X(data_path)
    # Remove empty columns and rows
    data = scprep.filter.remove_empty_cells(data)
    data = scprep.filter.remove_empty_genes(data)
    # Filter by library size to remove background
    scprep.plot.plot_library_size(data, cutoff=500)
    data = scprep.filter.filter_library_size(data, cutoff=500)
    # Filter by mitochondrial expression to remove dead cells
    mt_genes = scprep.utils.get_gene_set(data, starts_with="MT")
    scprep.plot.plot_gene_set_expression(data, mt_genes, percentile=90)
    data = scprep.filter.filter_gene_set_expression(data, mt_genes, percentile=90)
    # Library size normalize
    data = scprep.normalize.library_size_normalize(data)
    # Square root transform
    data = scprep.transform.sqrt(data)

### Help

Read the docs at If you have any questions or require assistance using scprep, please read the documentation at https://scprep.readthedocs.io/ or contact us at https://krishnaswamylab.org/get-help