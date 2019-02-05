from tools import data
import scprep
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.utils.testing import assert_raise_message, assert_warns_message
import unittest

self = unittest.TestCase()
self.X_pca = np.random.normal(0, 1, [100, 3])


def try_remove(filename):
    try:
        os.remove(filename)
    except FileNotFoundError:
        pass


def test_default_matplotlibrc():
    for key in ['axes.labelsize',
                'axes.titlesize',
                'figure.titlesize',
                'legend.fontsize',
                'legend.title_fontsize',
                'xtick.labelsize',
                'ytick.labelsize']:
        assert scprep.plot.utils._is_default_matplotlibrc() is True
        default = plt.rcParams[key]
        plt.rcParams[key] = 'xx-large'
        assert scprep.plot.utils._is_default_matplotlibrc() is False
        plt.rcParams[key] = default
    assert scprep.plot.utils._is_default_matplotlibrc() is True


def test_parse_fontsize():
    for key in ['axes.labelsize',
                'axes.titlesize',
                'figure.titlesize',
                'legend.fontsize',
                'legend.title_fontsize',
                'xtick.labelsize',
                'ytick.labelsize']:
        assert scprep.plot.utils.parse_fontsize(
            'x-large', 'large') == 'x-large'
        assert scprep.plot.utils.parse_fontsize(None, 'large') == 'large'
        default = plt.rcParams[key]
        plt.rcParams[key] = 'xx-large'
        assert scprep.plot.utils.parse_fontsize(
            'x-large', 'large') == 'x-large'
        assert scprep.plot.utils.parse_fontsize(None, 'large') is None
        plt.rcParams[key] = default
    assert scprep.plot.utils.parse_fontsize('x-large', 'large') == 'x-large'
    assert scprep.plot.utils.parse_fontsize(None, 'large') == 'large'


class Test10X(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.X = data.load_10X(sparse=False)
        self.X_pca, self.S = scprep.reduce.pca(self.X, n_components=10,
                                               return_singular_values=True)

    @classmethod
    def tearDownClass(self):
        try_remove("test.png")
        try_remove("test.gif")
        try_remove("test.mp4")

    def tearDown(self):
        plt.close('all')

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
        scprep.plot.scatter2d(self.X_pca, c=self.X_pca[:, 0],
                              legend_title="test", title="title test")

    def test_scatter_discrete(self):
        ax = scprep.plot.scatter2d(self.X_pca, c=np.random.choice(
            ['hello', 'world'], self.X_pca.shape[0], replace=True),
            legend_title="test", legend_loc='center left',
            legend_anchor=(1.02, 0.5))
        assert ax.get_legend().get_title().get_text() == 'test'

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

    def test_scatter_list_discrete_missing(self):
        scprep.plot.scatter2d(self.X_pca, c=np.random.choice(
            ['hello', 'great', 'world'], self.X_pca.shape[0], replace=True),
            cmap=['red', 'green'])

    def test_scatter_list_continuous(self):
        scprep.plot.scatter2d(self.X_pca, c=self.X_pca[:, 0],
                              cmap=['red', 'green'])

    def test_scatter_list_single(self):
        scprep.plot.scatter2d(self.X_pca, c=self.X_pca[:, 0],
                              cmap=['red'])

    def test_scatter_list_c_none(self):
        assert_raise_message(
            ValueError,
            "Expected list-like `c` with list cmap. Got <class 'NoneType'>",
            scprep.plot.scatter2d, self.X_pca, c=None,
            cmap=['red', 'green'])

    def test_scatter_discrete_greater_than_10(self):
        scprep.plot.scatter2d(
            self.X_pca, c=np.arange(self.X_pca.shape[0]) % 11)

    def test_scatter_solid(self):
        scprep.plot.scatter3d(self.X_pca, c='green')

    def test_scatter_none(self):
        scprep.plot.scatter2d(self.X_pca, c=None)

    def test_scatter_no_ticks(self):
        ax = scprep.plot.scatter3d(self.X_pca, zticks=False)
        assert len(ax.get_zticks()) == 0

    def test_scatter_no_ticklabels(self):
        ax = scprep.plot.scatter3d(self.X_pca, zticklabels=False)
        assert np.all([lab.get_text() == '' for lab in ax.get_zticklabels()])

    def test_scatter_custom_ticks(self):
        ax = scprep.plot.scatter2d(self.X_pca, xticks=[0, 1, 2])
        assert np.all(ax.get_xticks() == np.array([0, 1, 2]))
        ax = scprep.plot.scatter3d(self.X_pca, zticks=False)
        assert np.all(ax.get_zticks() == np.array([]))

    def test_scatter_custom_ticklabels(self):
        ax = scprep.plot.scatter2d(self.X_pca, xticks=[0, 1, 2],
                                   xticklabels=['a', 'b', 'c'])
        assert np.all(ax.get_xticks() == np.array([0, 1, 2]))
        xticklabels = np.array([lab.get_text()
                                for lab in ax.get_xticklabels()])
        assert np.all(xticklabels == np.array(['a', 'b', 'c']))

    def test_scatter_axis_labels(self):
        ax = scprep.plot.scatter3d(
            self.X_pca, label_prefix="test")
        assert ax.get_xlabel() == "test1"
        assert ax.get_ylabel() == "test2"
        assert ax.get_zlabel() == "test3"
        ax = scprep.plot.scatter2d(
            self.X_pca, label_prefix="test", xlabel="override")
        assert ax.get_xlabel() == "override"
        assert ax.get_ylabel() == "test2"

    def test_scatter_axis_savefig(self):
        scprep.plot.scatter2d(
            self.X_pca, filename="test.png")
        assert os.path.exists("test.png")

    def test_scatter_viewinit(self):
        ax = scprep.plot.scatter3d(
            self.X_pca, elev=80, azim=270)
        assert ax.elev == 80
        assert ax.azim == 270

    def test_scatter_rotate_gif(self):
        scprep.plot.rotate_scatter3d(self.X_pca, fps=5, dpi=50,
                                     filename="test.gif")
        assert os.path.exists("test.gif")

    def test_scatter_rotate_mp4(self):
        scprep.plot.rotate_scatter3d(self.X_pca, fps=5, dpi=50,
                                     filename="test.mp4")
        assert os.path.exists("test.mp4")

    def test_scatter_rotate_invalid_filename(self):
        assert_raise_message(
            ValueError,
            "filename must end in .gif or .mp4. Got test.invalid",
            scprep.plot.rotate_scatter3d,
            self.X_pca, fps=5, dpi=50, filename="test.invalid")

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
            UserWarning, "Cannot create a legend with constant `c=red`",
            scprep.plot.scatter2d, self.X_pca, legend=True,
            c='red')
        assert_warns_message(
            UserWarning, "Cannot create a legend with constant `c=None`",
            scprep.plot.scatter2d, self.X_pca, legend=True,
            c=None)

    def test_scatter_invalid_axis(self):
        fig, ax = plt.subplots()
        assert_raise_message(
            TypeError, "Expected ax with projection='3d'. "
            "Got 2D axis instead.",
            scprep.plot.scatter3d, self.X_pca, ax=ax)

    def test_scatter_colorbar(self):
        scprep.plot.scatter3d(self.X_pca, c=self.X_pca[:, 0], colorbar=True)

    def test_scatter_colorbar_log(self):
        scprep.plot.scatter2d(self.X_pca, c=np.abs(self.X_pca[:, 0]) + 1e-7,
                              colorbar=True, cmap_scale='log')

    def test_scatter_colorbar_log_constant_c(self):
        assert_warns_message(
            UserWarning,
            "Cannot use non-linear `cmap_scale` with constant `c=blue`",
            scprep.plot.scatter2d, self.X_pca, c='blue',
            colorbar=True, cmap_scale='log')

    def test_scatter_colorbar_log_discrete(self):
        assert_warns_message(
            UserWarning,
            "Cannot use non-linear `cmap_scale` with discrete data.",
            scprep.plot.scatter2d, self.X_pca,
            c=np.random.choice(['hello', 'world'], self.X_pca.shape[0]),
            colorbar=True, cmap_scale='log')

    def test_scatter_colorbar_log_negative(self):
        assert_raise_message(
            ValueError, "`vmin` must be positive for `cmap_scale='log'`. "
            "Got {}".format(self.X_pca[:, 0].min()),
            scprep.plot.scatter2d, self.X_pca,
            c=self.X_pca[:, 0],
            colorbar=True, cmap_scale='log')

    def test_scatter_colorbar_symlog(self):
        scprep.plot.scatter2d(self.X_pca, c=self.X_pca[:, 0],
                              colorbar=True, cmap_scale='symlog')

    def test_scatter_colorbar_sqrt(self):
        scprep.plot.scatter2d(self.X_pca, c=self.X_pca[:, 0],
                              colorbar=True, cmap_scale='sqrt')

    def test_scatter_colorbar_invalid(self):
        assert_raise_message(
            ValueError, "Expected norm in ['linear', 'log', 'symlog',"
            "'sqrt'] or a matplotlib.colors.Normalize object."
            " Got invalid",
            scprep.plot.scatter2d,
            self.X_pca, c=self.X_pca[:, 0],
            colorbar=True, cmap_scale='invalid')

    def test_scatter_legend_and_colorbar(self):
        assert_raise_message(
            ValueError, "Received conflicting values for synonyms "
            "`legend=True` and `colorbar=False`",
            scprep.plot.scatter2d, self.X_pca, c=self.X_pca[:, 0],
            legend=True, colorbar=False)

    def test_scatter_vmin_vmax(self):
        scprep.plot.scatter2d(
            self.X_pca, c=self.X_pca[:, 0], vmin=1, vmax=2)

    def test_scatter_vmin_vmax_discrete(self):
        assert_warns_message(
            UserWarning, "Cannot set `vmin` or `vmax` with discrete data. "
            "Setting to `None`.", scprep.plot.scatter3d,
            self.X_pca, c=np.random.choice(
                ['hello', 'world'], self.X_pca.shape[0], replace=True),
            vmin=1, vmax=2)

    def test_scatter_vmin_vmax_solid_color(self):
        assert_warns_message(
            UserWarning, "Cannot set `vmin` or `vmax` with constant `c=red`. "
            "Setting `vmin = vmax = None`.", scprep.plot.scatter3d,
            self.X_pca, c='red', vmin=1, vmax=2)

    def test_generate_colorbar_n_ticks(self):
        cb = scprep.plot.tools.generate_colorbar('inferno', vmin=0, vmax=1,
                                                 n_ticks=4)
        assert len(cb.get_ticks()) == 4

    def test_generate_colorbar_vmin_vmax_none(self):
        cb = scprep.plot.tools.generate_colorbar('inferno')
        assert_warns_message(
            UserWarning,
            "Cannot set `n_ticks` without setting `vmin` and `vmax`.",
            scprep.plot.tools.generate_colorbar,
            n_ticks=4)

    def test_generate_colorbar_mappable(self):
        im = plt.imshow([np.arange(10), np.arange(10)])
        scprep.plot.tools.generate_colorbar(mappable=im)
        assert_warns_message(
            UserWarning,
            "Cannot set `vmin` or `vmax` when `mappable` is given.",
            scprep.plot.tools.generate_colorbar,
            mappable=im, vmin=10, vmax=20)
        assert_warns_message(
            UserWarning,
            "Cannot set `cmap` when `mappable` is given.",
            scprep.plot.tools.generate_colorbar,
            mappable=im, cmap='inferno')
        assert_warns_message(
            UserWarning,
            "Cannot set `scale` when `mappable` is given.",
            scprep.plot.tools.generate_colorbar,
            mappable=im, scale='log')

    def test_generate_colorbar_vmin_none_vmax_given(self):
        assert_raise_message(
            ValueError,
            "Either both or neither of `vmax` and `vmin` should be set. "
            "Got `vmax=None, vmin=0`",
            scprep.plot.tools.generate_colorbar, 'inferno', vmin=0)

    def test_marker_plot(self):
        scprep.plot.marker_plot(
            data=self.X,
            clusters=np.random.choice(
                np.arange(10), replace=True, size=self.X.shape[0]),
            gene_names=self.X.columns,
            markers={'tissue': [self.X.columns[0]]})

    def test_marker_plot_bad_gene_names(self):
        assert_raise_message(
            ValueError,
            'All genes in `markers` must appear '
            'in gene_names. Did not find: {}'.format('z'),
            scprep.plot.marker_plot,
            data=self.X,
            clusters=np.random.choice(
                np.arange(10), replace=True, size=self.X.shape[0]),
            gene_names=self.X.columns,
            markers={'tissue': ['z']})

    def test_style_phate(self):
        ax = scprep.plot.scatter2d(self.X_pca)
        scprep.plot.style.style_phate(ax)
        assert len(ax.get_xticks()) == 0
        assert len(ax.get_yticks()) == 0
        ax = scprep.plot.scatter3d(self.X_pca)
        scprep.plot.style.style_phate(ax)
        assert len(ax.get_xticks()) == 0
        assert len(ax.get_yticks()) == 0
        assert len(ax.get_zticks()) == 0

    def test_label_axis_va(self):
        ax = scprep.plot.scatter2d(self.X_pca)
        scprep.plot.tools.label_axis(
            ax.yaxis, ticklabel_vertical_alignment="top")
        for tick in ax.yaxis.get_ticklabels():
            assert tick.get_va() == "top"
        scprep.plot.tools.label_axis(
            ax.yaxis, ticklabel_vertical_alignment="bottom")
        for tick in ax.yaxis.get_ticklabels():
            assert tick.get_va() == "bottom"

    def test_label_axis_ha(self):
        ax = scprep.plot.scatter2d(self.X_pca)
        scprep.plot.tools.label_axis(
            ax.xaxis, ticklabel_horizontal_alignment="left")
        for tick in ax.xaxis.get_ticklabels():
            assert tick.get_ha() == "left"
        scprep.plot.tools.label_axis(
            ax.xaxis, ticklabel_horizontal_alignment="right")
        for tick in ax.xaxis.get_ticklabels():
            assert tick.get_ha() == "right"

    def test_label_axis_rotation(self):
        ax = scprep.plot.scatter2d(self.X_pca)
        scprep.plot.tools.label_axis(
            ax.xaxis, ticklabel_rotation=45)
        for tick in ax.xaxis.get_ticklabels():
            assert tick.get_rotation() == 45
        scprep.plot.tools.label_axis(
            ax.xaxis, ticklabel_rotation=90)
        for tick in ax.xaxis.get_ticklabels():
            assert tick.get_rotation() == 90
