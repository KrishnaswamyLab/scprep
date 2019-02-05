=============
scprep
=============

.. image:: https://img.shields.io/pypi/v/scprep.svg
    :target: https://pypi.org/project/scprep/
    :alt: Latest PyPi version
.. image:: https://anaconda.org/bioconda/scprep/badges/version.svg
    :target: https://anaconda.org/bioconda/scprep/
    :alt: Latest Conda version
.. image:: https://api.travis-ci.com/KrishnaswamyLab/scprep.svg?branch=master
    :target: https://travis-ci.com/KrishnaswamyLab/scprep
    :alt: Travis CI Build
.. image:: https://img.shields.io/readthedocs/scprep.svg
    :target: https://scprep.readthedocs.io/
    :alt: Read the Docs
.. image:: https://coveralls.io/repos/github/KrishnaswamyLab/scprep/badge.svg?branch=master
    :target: https://coveralls.io/github/KrishnaswamyLab/scprep?branch=master
    :alt: Coverage Status
.. image:: https://img.shields.io/twitter/follow/KrishnaswamyLab.svg?style=social&label=Follow
    :target: https://twitter.com/KrishnaswamyLab
    :alt: Twitter
.. image:: https://img.shields.io/github/stars/KrishnaswamyLab/scprep.svg?style=social&label=Stars
    :target: https://github.com/KrishnaswamyLab/scprep/
    :alt: GitHub stars


Tools for loading and preprocessing biological matrices in Python.

Installation
------------

preprocessing is available on `pip`. Install by running the following in a terminal::

    pip install --user scprep

Alternatively, scprep can be installed using `Conda <https://conda.io/docs/>`_ (most easily obtained via the `Miniconda Python distribution <https://conda.io/miniconda.html>`_)::

    conda install -c bioconda scprep

Usage example
-------------

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
    mt_genes = scprep.select.get_gene_set(data, starts_with="MT")
    scprep.plot.plot_gene_set_expression(data, genes=mt_genes, percentile=90)
    data = scprep.filter.filter_gene_set_expression(data, genes=mt_genes, 
                                                    percentile=90)
    # Library size normalize
    data = scprep.normalize.library_size_normalize(data)
    # Square root transform
    data = scprep.transform.sqrt(data)

Help
----

If you have any questions or require assistance using scprep, please read the documentation at https://scprep.readthedocs.io/ or contact us at https://krishnaswamylab.org/get-help