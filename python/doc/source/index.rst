===========================================================================
graphtools
===========================================================================

.. raw:: html

    <a href="https://pypi.org/project/graphtools/"><img src="https://img.shields.io/pypi/v/graphtools.svg" alt="Latest PyPi version"></a>

.. raw:: html

    <a href="https://travis-ci.com/KrishnaswamyLab/graphtools"><img src="https://api.travis-ci.com/KrishnaswamyLab/graphtools.svg?branch=master" alt="Travis CI Build"></a>

.. raw:: html

    <a href="https://graphtools.readthedocs.io/"><img src="https://img.shields.io/readthedocs/graphtools.svg" alt="Read the Docs"></img></a>

.. raw:: html

    <a href="https://twitter.com/KrishnaswamyLab"><img src="https://img.shields.io/twitter/follow/KrishnaswamyLab.svg?style=social&label=Follow" alt="Twitter"></a>

.. raw:: html

    <a href="https://github.com/KrishnaswamyLab/graphtools/"><img src="https://img.shields.io/github/stars/KrishnaswamyLab/graphtools.svg?style=social&label=Stars" alt="GitHub stars"></a>

Tools for building and manipulating graphs in Python.

.. toctree::
    :maxdepth: 2

    installation
    reference

Quick Start
===========

To use `graphtools`, create a `graphtools.Graph` class::

    from sklearn import datasets
    import graphtools
    digits = datasets.load_digits()
    G = graphtools.Graph(digits['data'])
    K = G.kernel
    P = G.diff_op
    G = graphtools.Graph(digits['data'], n_landmark=300)
    L = G.landmark_op

To use `graphtools` with `pygsp`, create a `graphtools.Graph` class with `use_pygsp=True`::

    from sklearn import datasets
    import graphtools
    digits = datasets.load_digits()
    G = graphtools.Graph(digits['data'], use_pygsp=True)
    N = G.N
    W = G.W
    basis = G.compute_fourier_basis()
