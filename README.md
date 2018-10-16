# scprep

[![Latest PyPI version](https://img.shields.io/pypi/v/scprep.svg)](https://pypi.org/project/scprep/)
[![install with bioconda](https://img.shields.io/badge/install%20with-bioconda-brightgreen.svg?style=flat-square)](http://bioconda.github.io/recipes/scprep/README.html)
[![Travis CI Build](https://api.travis-ci.com/KrishnaswamyLab/scprep.svg?branch=master)](https://travis-ci.com/KrishnaswamyLab/scprep)
[![Read the Docs](https://img.shields.io/readthedocs/scprep.svg)](https://scprep.readthedocs.io/)
[![Coverage Status](https://coveralls.io/repos/github/KrishnaswamyLab/scprep/badge.svg?branch=master)](https://coveralls.io/github/KrishnaswamyLab/scprep?branch=master)
[![Twitter](https://img.shields.io/twitter/follow/KrishnaswamyLab.svg?style=social&label=Follow)](https://twitter.com/KrishnaswamyLab)

Tools for loading and preprocessing biological matrices in Python and MATLAB.

## Python

### Installation

Scprep is available on `pip`. Install by running the following in a terminal::

    pip install --user scprep
    
Alternatively, Scprep can be installed using [Conda](https://conda.io/docs/) (most easily obtained via the [Miniconda Python distribution](https://conda.io/miniconda.html)):

    conda install -c bioconda scprep

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
    data = scprep.filter.filter_gene_set_expression(data, mt_genes, 
                                                    percentile=90)
    # Library size normalize
    data = scprep.normalize.library_size_normalize(data)
    # Square root transform
    data = scprep.transform.sqrt(data)

### Help

Read the docs at If you have any questions or require assistance using scprep, please read the documentation at https://scprep.readthedocs.io/ or contact us at https://krishnaswamylab.org/get-help
