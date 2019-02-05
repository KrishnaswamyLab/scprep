===========================================================================
scprep
===========================================================================

.. raw:: html

    <a href="https://pypi.org/project/scprep/"><img src="https://img.shields.io/pypi/v/scprep.svg" alt="Latest PyPi version"></a>

.. raw:: html

    <a href="https://travis-ci.com/KrishnaswamyLab/scprep"><img src="https://api.travis-ci.com/KrishnaswamyLab/scprep.svg?branch=master" alt="Travis CI Build"></a>

.. raw:: html

    <a href="https://scprep.readthedocs.io/"><img src="https://img.shields.io/readthedocs/scprep.svg" alt="Read the Docs"></img></a>

.. raw:: html

    <a href="https://coveralls.io/github/KrishnaswamyLab/scprep?branch=master"><img src="https://coveralls.io/repos/github/KrishnaswamyLab/scprep/badge.svg?branch=master" alt="Coverage Status"></img></a>

.. raw:: html

    <a href="https://twitter.com/KrishnaswamyLab"><img src="https://img.shields.io/twitter/follow/KrishnaswamyLab.svg?style=social&label=Follow" alt="Twitter"></a>

.. raw:: html

    <a href="https://github.com/KrishnaswamyLab/scprep/"><img src="https://img.shields.io/github/stars/KrishnaswamyLab/scprep.svg?style=social&label=Stars" alt="GitHub stars"></a>

Tools for building and manipulating graphs in Python.

.. toctree::
    :maxdepth: 2

    installation
    reference

Quick Start
===========

You can use `scprep` with your single cell data as follows::

	import scprep
	# Load data
	data_path = "~/mydata/my_10X_data"
	data = scprep.io.load_10X(data_path)
	# Remove empty columns and rows
	data = scprep.filter.filter_empty_cells(data)
	data = scprep.filter.filter_empty_genes(data)
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
====

If you have any questions or require assistance using scprep, please contact us at https://krishnaswamylab.org/get-help