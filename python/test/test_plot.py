from tools import data
import scprep
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.testing import assert_raise_message, assert_warns_message
import unittest


class Test10X(unittest.TestCase):

    def setUp(self):
        self.X = data.load_10X(sparse=False)
        self.X_pca, self.S = scprep.reduce.pca(self.X, n_components=10,
                                               return_singular_values=True)

    def test_histogram(self):
        scprep.plot.plot_library_size(self.X, cutoff=1000, log=True)
        scprep.plot.plot_library_size(self.X, cutoff=1000, log=True,
                                      xlabel="x label", ylabel="y label")

    def test_histogram_custom_axis(self):
        fig, ax = plt.subplots()
        scprep.plot.plot_gene_set_expression(
            self.X, genes=scprep.utils.get_gene_set(self.X, starts_with="D"),
            percentile=90, log='y', ax=ax)

    def test_histogram_invalid_axis(self):
        assert_raise_message(
            TypeError,
            "Expected ax as a matplotlib.axes.Axes. Got ",
            scprep.plot.plot_library_size,
            self.X, ax="invalid")

    def test_scree(self):
        scprep.plot.scree_plot(self.S)
        scprep.plot.scree_plot(self.S, cumulative=True,
                               xlabel="x label", ylabel="y label")

    def test_scree_custom_axis(self):
        fig, ax = plt.subplots()
        scprep.plot.scree_plot(self.S, ax=ax)

    def test_scree_invalid_axis(self):
        assert_raise_message(
            TypeError,
            "Expected ax as a matplotlib.axes.Axes. Got ",
            scprep.plot.scree_plot,
            self.S, ax="invalid")

    def test_scatter_continuous(self):
        scprep.plot.scatter2d(self.X_pca, c=self.X_pca[:, 0])

    def test_scatter_discrete(self):
        scprep.plot.scatter2d(self.X_pca, c=np.random.choice(
            ['hello', 'world'], self.X_pca.shape[0], replace=True))

    def test_scatter_dict(self):
        scprep.plot.scatter2d(self.X_pca, c=np.random.choice(
            ['hello', 'world'], self.X_pca.shape[0], replace=True),
            cmap={'hello': 'red', 'world': 'green'})

    def test_scatter_dict_c_none(self):
        assert_raise_message(
            ValueError,
            "Expected list-like `c` with dictionary cmap. Got <class 'NoneType'>",
            scprep.plot.scatter2d, self.X_pca, c=None,
            cmap={'hello': 'red', 'world': 'green'})

    def test_scatter_dict_continuous(self):
        assert_raise_message(
            ValueError,
            "Cannot use dictionary cmap with continuous data",
            scprep.plot.scatter2d, self.X_pca, c=self.X_pca[:, 0],
            discrete=False, cmap={'hello': 'red', 'world': 'green'})

    def test_scatter_dict_missing(self):
        assert_raise_message(
            ValueError,
            "Dictionary cmap requires a color for every unique entry in `c`. "
            "Missing colors for [world]",
            scprep.plot.scatter2d, self.X_pca, c=np.random.choice(
                ['hello', 'world'], self.X_pca.shape[0], replace=True),
            cmap={'hello': 'red'})

    def test_scatter_list_discrete(self):
        scprep.plot.scatter2d(self.X_pca, c=np.random.choice(
            ['hello', 'world'], self.X_pca.shape[0], replace=True),
            cmap=['red', 'green'])

    def test_scatter_list_continuous(self):
        scprep.plot.scatter2d(self.X_pca, c=self.X_pca[:, 0],
                              cmap=['red', 'green'], legend_title="test")

    def test_scatter_list_c_none(self):
        assert_raise_message(
            ValueError,
            "Expected list-like `c` with list cmap. Got <class 'NoneType'>",
            scprep.plot.scatter2d, self.X_pca, c=None,
            cmap=['red', 'green'])

    def test_scatter_list_missing(self):
        assert_raise_message(
            ValueError,
            "List cmap with discrete data requires a color for every unique "
            "entry in `c` (2). Got 1",
            scprep.plot.scatter2d, self.X_pca, c=np.random.choice(
                ['hello', 'world'], self.X_pca.shape[0], replace=True),
            cmap=['red'])

    def test_scatter_discrete_greater_than_10(self):
        scprep.plot.scatter2d(
            self.X_pca, c=np.arange(self.X_pca.shape[0]) % 11)

    def test_scatter_solid(self):
        scprep.plot.scatter3d(self.X_pca, c='green')

    def test_scatter_none(self):
        scprep.plot.scatter2d(self.X_pca, c=None)

    def test_scatter_invalid_data(self):
        assert_raise_message(
            ValueError, "Expected all axes of data to have the same length. "
            "Got {}".format([self.X_pca.shape[0], self.X_pca.shape[1]]),
            scprep.plot.scatter, x=self.X_pca[:, 0], y=self.X_pca[0, :])
        assert_raise_message(
            ValueError, "Expected all axes of data to have the same length. "
            "Got {}".format([self.X_pca.shape[0], self.X_pca.shape[0],
                             self.X_pca.shape[1]]),
            scprep.plot.scatter, x=self.X_pca[:, 0], y=self.X_pca[:, 0],
            z=self.X_pca[0, :])

    def test_scatter_invalid_c(self):
        assert_raise_message(
            ValueError, "Expected c of length {} or 1. Got {}".format(
                self.X_pca.shape[0], self.X_pca.shape[1]),
            scprep.plot.scatter2d, self.X_pca,
            c=self.X_pca[0, :])

    def test_scatter_invalid_discrete(self):
        assert_raise_message(
            ValueError, "Cannot treat non-numeric data as continuous",
            scprep.plot.scatter2d, self.X_pca, discrete=False,
            c=np.random.choice(
                ['hello', 'world'], self.X_pca.shape[0], replace=True))

    def test_scatter_invalid_legend(self):
        assert_warns_message(
            UserWarning, "`c` is a color array and cannot be used to create a "
            "legend. To interpret these values as labels instead, "
            "provide a `cmap` dictionary with label-color pairs.",
            scprep.plot.scatter2d, self.X_pca, legend=True,
            c=np.random.choice(['red', 'blue'],
                               self.X_pca.shape[0], replace=True))
        assert_warns_message(
            UserWarning, "Cannot create a legend with `c=red`",
            scprep.plot.scatter2d, self.X_pca, legend=True,
            c='red')
