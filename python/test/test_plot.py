from tools import data
import scprep
import matplotlib.pyplot as plt
from sklearn.utils.testing import assert_raise_message


def test_plot_histogram():
    X = data.load_10X()
    scprep.plot.plot_library_size(X, cutoff=1000, log=True)
    fig, ax = plt.subplots()
    scprep.plot.plot_gene_set_expression(
        X, genes=scprep.utils.get_gene_set(X, starts_with="D"),
        percentile=90, log='y', ax=ax)
    assert_raise_message(
        TypeError,
        "Expected ax as a matplotlib.axes.Axes. Got ",
        scprep.plot.plot_library_size,
        X, ax="invalid")
