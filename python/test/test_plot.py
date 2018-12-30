from tools import data
import scprep
import matplotlib.pyplot as plt
from sklearn.utils.testing import assert_raise_message
import unittest


class Test10X(unittest.TestCase):

    def setUp(self):
        self.X = data.load_10X(sparse=False)
        self.S = scprep.reduce.pca(self.X, n_components=10,
                                   return_singular_values=True)[1]

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
