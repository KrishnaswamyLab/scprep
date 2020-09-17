import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import scprep
import matplotlib

import os
import sys
import numbers
import unittest

from packaging.version import Version

from scprep.plot.scatter import _ScatterParams
from scprep.plot.jitter import _JitterParams
from scprep.plot.histogram import _symlog_bins

from tools import data, utils


def try_remove(filename):
    try:
        os.remove(filename)
    except FileNotFoundError:
        pass


def test_default_matplotlibrc():
    for key in [
        "axes.labelsize",
        "axes.titlesize",
        "figure.titlesize",
        "legend.fontsize",
        "legend.title_fontsize",
        "xtick.labelsize",
        "ytick.labelsize",
    ]:
        assert scprep.plot.utils._is_default_matplotlibrc() is True
        default = plt.rcParams[key]
        plt.rcParams[key] = "xx-large"
        assert scprep.plot.utils._is_default_matplotlibrc() is False
        plt.rcParams[key] = default
    assert scprep.plot.utils._is_default_matplotlibrc() is True


def test_parse_fontsize():
    for key in [
        "axes.labelsize",
        "axes.titlesize",
        "figure.titlesize",
        "legend.fontsize",
        "legend.title_fontsize",
        "xtick.labelsize",
        "ytick.labelsize",
    ]:
        assert scprep.plot.utils.parse_fontsize("x-large", "large") == "x-large"
        assert scprep.plot.utils.parse_fontsize(None, "large") == "large"
        default = plt.rcParams[key]
        plt.rcParams[key] = "xx-large"
        assert scprep.plot.utils.parse_fontsize("x-large", "large") == "x-large"
        assert scprep.plot.utils.parse_fontsize(None, "large") is None
        plt.rcParams[key] = default
    assert scprep.plot.utils.parse_fontsize("x-large", "large") == "x-large"
    assert scprep.plot.utils.parse_fontsize(None, "large") == "large"


def test_generate_colorbar_str():
    cb = scprep.plot.tools.generate_colorbar(cmap="viridis")
    assert cb.cmap.name == "viridis"


def test_generate_colorbar_colormap():
    cb = scprep.plot.tools.generate_colorbar(cmap=plt.cm.viridis)
    assert cb.cmap.name == "viridis"


def test_generate_colorbar_list():
    cb = scprep.plot.tools.generate_colorbar(cmap=["red", "blue"])
    assert cb.cmap.name == "scprep_custom_cmap"


def test_generate_colorbar_dict():
    if Version(matplotlib.__version__) >= Version("3.2"):
        errtype = ValueError
        msg = "is not a valid value for name; supported values are"
    else:
        errtype = TypeError
        msg = "unhashable type: 'dict'"
    utils.assert_raises_message(
        errtype,
        msg,
        scprep.plot.tools.generate_colorbar,
        cmap={"+": "r", "-": "b"},
    )


def test_tab30():
    cmap = scprep.plot.colors.tab30()
    np.testing.assert_array_equal(
        cmap.colors[:15],
        np.array(matplotlib.cm.tab20c.colors)[
            [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18]
        ],
    )
    np.testing.assert_array_equal(
        cmap.colors[15:],
        np.array(matplotlib.cm.tab20b.colors)[
            [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18]
        ],
    )


def test_tab40():
    cmap = scprep.plot.colors.tab40()
    np.testing.assert_array_equal(cmap.colors[:20], matplotlib.cm.tab20c.colors)
    np.testing.assert_array_equal(cmap.colors[20:], matplotlib.cm.tab20b.colors)


def test_tab10_continuous():
    cmap = scprep.plot.colors.tab10_continuous(n_colors=10, n_step=2, reverse=True)
    np.testing.assert_allclose(
        cmap.colors,
        np.hstack([matplotlib.cm.tab20.colors, np.ones((20, 1))]),
        atol=0.06,
    )


def test_tab10_continuous_no_reverse():
    cmap = scprep.plot.colors.tab10_continuous(n_colors=10, n_step=2, reverse=False)
    colors = np.array(cmap.colors)
    for i in range(len(colors) // 2):
        tmp = np.array(colors[2 * i])
        colors[2 * i] = colors[2 * i + 1]
        colors[2 * i + 1] = tmp
    np.testing.assert_allclose(
        colors, np.hstack([matplotlib.cm.tab20.colors, np.ones((20, 1))]), atol=0.06
    )


def test_tab10_continuous_invalid_n_colors():
    utils.assert_raises_message(
        ValueError,
        "Expected 0 < n_colors <= 10. Got 0",
        scprep.plot.colors.tab10_continuous,
        n_colors=0,
    )
    utils.assert_raises_message(
        ValueError,
        "Expected 0 < n_colors <= 10. Got 11",
        scprep.plot.colors.tab10_continuous,
        n_colors=11,
    )
    utils.assert_raises_message(
        ValueError,
        "Expected n_step >= 2. Got 1",
        scprep.plot.colors.tab10_continuous,
        n_step=1,
    )


def test_tab_exact():
    assert scprep.plot.colors.tab(1) is plt.cm.tab10
    np.testing.assert_array_equal(
        scprep.plot.colors.tab(10).colors, plt.cm.tab10.colors
    )
    np.testing.assert_array_equal(
        scprep.plot.colors.tab(20).colors, plt.cm.tab20.colors
    )
    np.testing.assert_array_equal(
        scprep.plot.colors.tab(30).colors, scprep.plot.colors.tab30().colors
    )
    np.testing.assert_array_equal(
        scprep.plot.colors.tab(40).colors, scprep.plot.colors.tab40().colors
    )
    np.testing.assert_array_equal(
        scprep.plot.colors.tab(50).colors,
        scprep.plot.colors.tab10_continuous(n_colors=10, n_step=5).colors,
    )


def test_tab_first10():
    np.testing.assert_array_equal(
        scprep.plot.colors.tab(19).colors[:10], plt.cm.tab10.colors
    )
    np.testing.assert_array_equal(
        scprep.plot.colors.tab(29).colors[:10], scprep.plot.colors.tab30().colors[::3]
    )
    np.testing.assert_array_equal(
        scprep.plot.colors.tab(39).colors[:10], scprep.plot.colors.tab40().colors[::4]
    )
    np.testing.assert_array_equal(
        scprep.plot.colors.tab(49).colors[:10],
        scprep.plot.colors.tab10_continuous(n_colors=10, n_step=5).colors[::5],
    )


def test_tab_first20():
    np.testing.assert_array_equal(
        scprep.plot.colors.tab(29).colors[10:20],
        scprep.plot.colors.tab30().colors[1::3],
    )
    np.testing.assert_array_equal(
        scprep.plot.colors.tab(39).colors[10:20],
        scprep.plot.colors.tab40().colors[1::4],
    )


def test_tab_first30():
    np.testing.assert_array_equal(
        scprep.plot.colors.tab(39).colors[20:30],
        scprep.plot.colors.tab40().colors[2::4],
    )


def test_tab_overhang():
    np.testing.assert_array_equal(
        scprep.plot.colors.tab(9).colors, plt.cm.tab10.colors[:9]
    )
    np.testing.assert_array_equal(
        scprep.plot.colors.tab(19).colors[10:], plt.cm.tab20.colors[1:-1:2]
    )
    np.testing.assert_array_equal(
        scprep.plot.colors.tab(29).colors[20:],
        scprep.plot.colors.tab30().colors[2:-1:3],
    )
    np.testing.assert_array_equal(
        scprep.plot.colors.tab(39).colors[30:],
        scprep.plot.colors.tab40().colors[3:-1:4],
    )
    np.testing.assert_array_equal(
        scprep.plot.colors.tab(49).colors[40:],
        scprep.plot.colors.tab10_continuous(n_colors=10, n_step=5).colors[4:-1:5],
    )


def test_tab_invalid():
    utils.assert_raises_message(
        ValueError, "Expected n >= 1. Got 0", scprep.plot.colors.tab, n=0
    )


def test_is_color_array_none():
    assert not scprep.plot.utils._is_color_array(None)


def test_symlog_bins():
    # all negative
    assert np.all(_symlog_bins(-10, -1, 1, 10) < 0)
    # all positive
    assert np.all(_symlog_bins(1, 10, 1, 10) > 0)
    # ends at zero
    assert np.all(_symlog_bins(-10, 0, 1, 10) <= 0)
    assert _symlog_bins(-10, 0, 1, 10)[-1] == 0
    # starts at zero
    assert np.all(_symlog_bins(0, 10, 1, 10) >= 0)
    assert _symlog_bins(0, 10, 1, 10)[0] == 0
    # identically zero
    assert np.all(_symlog_bins(0, 0, 0.1, 10) == [-1, -0.1, 0.1, 1])


class TestScatterParams(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.X = np.random.normal(0, 1, [500, 4])
        self.x = self.X[:, 0]
        self.y = self.X[:, 1]
        self.z = self.X[:, 2]
        self.c = self.X[:, 3]
        self.array_c = np.vstack([self.c, self.c, self.c, self.c]).T
        self.array_c = self.array_c - np.min(self.array_c)
        self.array_c = self.array_c / np.max(self.array_c)

    def test_size(self):
        params = _ScatterParams(x=self.x, y=self.y)
        assert params.size == len(self.x)

    def test_plot_idx_shuffle(self):
        params = _ScatterParams(
            x=self.x, y=self.y, z=self.z, c=self.c, s=np.abs(self.x)
        )
        assert not np.all(params.plot_idx == np.arange(params.size))
        np.testing.assert_equal(params.x, self.x[params.plot_idx])
        np.testing.assert_equal(params.y, self.y[params.plot_idx])
        np.testing.assert_equal(params.z, self.z[params.plot_idx])
        np.testing.assert_equal(params.c, self.c[params.plot_idx])
        np.testing.assert_equal(params.s, np.abs(self.x)[params.plot_idx])

    def test_plot_idx_no_shuffle(self):
        params = _ScatterParams(
            x=self.x, y=self.y, z=self.z, c=self.c, s=np.abs(self.x), shuffle=False
        )
        np.testing.assert_equal(params.plot_idx, np.arange(params.size))
        np.testing.assert_equal(params.x, self.x)
        np.testing.assert_equal(params.y, self.y)
        np.testing.assert_equal(params.z, self.z)
        np.testing.assert_equal(params.c, self.c)
        np.testing.assert_equal(params.s, np.abs(self.x))

    def test_plot_idx_mask(self):
        params = _ScatterParams(
            x=self.x, y=self.y, z=self.z, c=self.c, mask=self.x > 0, shuffle=False
        )
        np.testing.assert_equal(params.plot_idx, np.arange(params.size)[self.x > 0])
        np.testing.assert_equal(params.x, self.x[self.x > 0])
        np.testing.assert_equal(params.y, self.y[self.x > 0])
        np.testing.assert_equal(params.z, self.z[self.x > 0])
        np.testing.assert_equal(params.c, self.c[self.x > 0])

    def test_plot_idx_mask_shuffle(self):
        params = _ScatterParams(x=self.x, y=self.y, mask=self.x > 0, shuffle=True)
        np.testing.assert_equal(
            np.sort(params.plot_idx), np.arange(params.size)[self.x > 0]
        )
        assert np.all(params.x > 0)

    def test_data_int(self):
        params = _ScatterParams(x=1, y=2)
        np.testing.assert_equal(params._data, [np.array([1]), np.array([2])])
        assert params.subplot_kw == {}

    def test_data_2d(self):
        params = _ScatterParams(x=self.x, y=self.y)
        np.testing.assert_equal(params._data, [self.x, self.y])
        np.testing.assert_equal(
            params.data, [self.x[params.plot_idx], self.y[params.plot_idx]]
        )
        assert params.subplot_kw == {}

    def test_data_3d(self):
        params = _ScatterParams(x=self.x, y=self.y, z=self.z)
        np.testing.assert_equal(params._data, [self.x, self.y, self.z])
        np.testing.assert_equal(
            params.data,
            [self.x[params.plot_idx], self.y[params.plot_idx], self.z[params.plot_idx]],
        )
        assert params.subplot_kw == {"projection": "3d"}

    def test_s_default(self):
        params = _ScatterParams(x=self.x, y=self.y)
        assert params.s == 200 / np.sqrt(params.size)

    def test_s_given(self):
        params = _ScatterParams(x=self.x, y=self.y, s=3)
        assert params.s == 3

    def test_c_none(self):
        params = _ScatterParams(x=self.x, y=self.y)
        assert params.constant_c()
        assert not params.array_c()
        assert params.discrete is None
        assert params.legend is False
        assert params.vmin is None
        assert params.vmax is None
        assert params.cmap is None
        assert params.cmap_scale is None
        assert params.extend is None

    def test_constant_c(self):
        params = _ScatterParams(x=self.x, y=self.y, c="blue")
        assert params.constant_c()
        assert not params.array_c()
        assert params.discrete is None
        assert params.legend is False
        assert params.vmin is None
        assert params.vmax is None
        assert params.cmap is None
        assert params.cmap_scale is None
        assert params.extend is None
        assert params.labels is None

    def test_array_c(self):
        params = _ScatterParams(x=self.x, y=self.y, c=self.array_c)
        assert params.array_c()
        assert not params.constant_c()
        np.testing.assert_array_equal(params.x, params._x[params.plot_idx])
        np.testing.assert_array_equal(params.y, params._y[params.plot_idx])
        np.testing.assert_array_equal(params.c, params._c[params.plot_idx])
        assert params.discrete is None
        assert params.legend is False
        assert params.vmin is None
        assert params.vmax is None
        assert params.cmap is None
        assert params.cmap_scale is None
        assert params.extend is None
        assert params.labels is None

    def test_continuous(self):
        params = _ScatterParams(x=self.x, y=self.y, c=self.c)
        assert not params.array_c()
        assert not params.constant_c()
        np.testing.assert_array_equal(params.x, params._x[params.plot_idx])
        np.testing.assert_array_equal(params.y, params._y[params.plot_idx])
        np.testing.assert_array_equal(params.c, params._c[params.plot_idx])
        assert params.discrete is False
        assert params.legend is True
        assert params.cmap_scale == "linear"
        assert params.cmap is plt.cm.inferno
        params = _ScatterParams(
            x=self.x, y=self.y, discrete=False, c=np.round(self.c % 1, 1)
        )
        assert not params.array_c()
        assert not params.constant_c()
        np.testing.assert_array_equal(params.x, params._x[params.plot_idx])
        np.testing.assert_array_equal(params.y, params._y[params.plot_idx])
        np.testing.assert_array_equal(params.c, params._c[params.plot_idx])
        assert params.discrete is False
        assert params.legend is True
        assert params.labels is None
        assert params.cmap_scale == "linear"
        assert params.cmap is plt.cm.inferno

    def test_discrete_tab10(self):
        params = _ScatterParams(x=self.x, y=self.y, c=np.where(self.c > 0, "+", "-"))
        assert not params.array_c()
        assert not params.constant_c()
        np.testing.assert_array_equal(params.x, params._x[params.plot_idx])
        np.testing.assert_array_equal(params.y, params._y[params.plot_idx])
        np.testing.assert_array_equal(params.c, params.c_discrete[params.plot_idx])
        assert params.discrete is True
        assert params.legend is True
        assert params.vmin is None
        assert params.vmax is None
        assert params.cmap_scale is None
        np.testing.assert_equal(params.cmap.colors, plt.cm.tab10.colors[:2])

    def test_discrete_tab20(self):
        params = _ScatterParams(x=self.x, y=self.y, c=10 * np.round(self.c % 1, 1))
        assert not params.array_c()
        assert not params.constant_c()
        assert params.discrete is True
        assert params.legend is True
        assert params.vmin is None
        assert params.vmax is None
        assert params.cmap_scale is None
        assert params.extend is None
        assert isinstance(params.cmap, matplotlib.colors.ListedColormap)
        np.testing.assert_equal(params.cmap.colors[:10], plt.cm.tab10.colors)
        np.testing.assert_equal(
            params.cmap.colors[10:],
            plt.cm.tab20.colors[1 : 1 + (len(params.cmap.colors) - 10) * 2 : 2],
        )

    def test_continuous_less_than_20(self):
        params = _ScatterParams(x=self.x, y=self.y, c=np.round(self.c % 1, 1))
        assert not params.array_c()
        assert not params.constant_c()
        assert params.discrete is False
        assert params.legend is True
        assert params.vmin == 0
        assert params.vmax == 1
        assert params.cmap_scale == "linear"
        assert params.extend == "neither"
        assert params.cmap is matplotlib.cm.inferno

    def test_continuous_tab20_str(self):
        params = _ScatterParams(
            x=self.x, y=self.y, discrete=False, cmap="tab20", c=np.round(self.c % 1, 1)
        )
        assert params.cmap is plt.cm.tab20

    def test_continuous_tab20_obj(self):
        params = _ScatterParams(
            x=self.x,
            y=self.y,
            discrete=False,
            cmap=plt.get_cmap("tab20"),
            c=np.round(self.c % 1, 1),
        )
        assert params.cmap is plt.cm.tab20

    def test_discrete_dark2(self):
        params = _ScatterParams(
            x=self.x,
            y=self.y,
            discrete=True,
            cmap="Dark2",
            c=np.where(self.c > 0, "+", "-"),
        )
        assert not params.array_c()
        assert not params.constant_c()
        assert params.discrete is True
        assert params.legend is True
        assert params.vmin is None
        assert params.vmax is None
        assert params.cmap_scale is None
        assert params.extend is None
        assert isinstance(params.cmap, matplotlib.colors.ListedColormap)
        np.testing.assert_equal(params.cmap.colors, plt.cm.Dark2.colors[:2])

    def test_c_discrete(self):
        c = np.where(self.c > 0, "a", "b")
        params = _ScatterParams(x=self.x, y=self.y, c=c)
        np.testing.assert_equal(params.c_discrete, np.where(c == "a", 0, 1))
        np.testing.assert_equal(params.labels, ["a", "b"])

    def test_legend(self):
        params = _ScatterParams(x=self.x, y=self.y, c=self.c, legend=False)
        assert params.legend is False
        params = _ScatterParams(x=self.x, y=self.y, c=self.c, colorbar=False)
        assert params.legend is False

    def test_vmin_given(self):
        params = _ScatterParams(x=self.x, y=self.y, c=self.c, vmin=0)
        assert params.vmin == 0

    def test_vmin_default(self):
        params = _ScatterParams(x=self.x, y=self.y, c=self.c)
        assert params.vmin == np.min(self.c)

    def test_vmax_given(self):
        params = _ScatterParams(x=self.x, y=self.y, c=self.c, vmax=0)
        assert params.vmax == 0

    def test_vmax_default(self):
        params = _ScatterParams(x=self.x, y=self.y, c=self.c)
        assert params.vmax == np.max(self.c)

    def test_list_cmap(self):
        params = _ScatterParams(x=self.x, y=self.y, c=self.c, cmap=["red", "black"])
        assert params.list_cmap()
        np.testing.assert_equal(params.cmap([0, 255]), [[1, 0, 0, 1], [0, 0, 0, 1]])

    def test_dict_cmap_fwd(self):
        params = _ScatterParams(
            x=self.x,
            y=self.y,
            c=np.where(self.c > 0, "+", "-"),
            cmap={"+": "k", "-": "r"},
        )
        assert not params.list_cmap()
        if sys.version_info[1] > 5:
            np.testing.assert_equal(params.cmap.colors, [[0, 0, 0, 1], [1, 0, 0, 1]])
            assert np.all(params._labels == np.array(["+", "-"]))
        else:
            try:
                np.testing.assert_equal(
                    params.cmap.colors, [[0, 0, 0, 1], [1, 0, 0, 1]]
                )
                assert np.all(params._labels == np.array(["+", "-"]))
            except AssertionError:
                np.testing.assert_equal(
                    params.cmap.colors, [[1, 0, 0, 1], [0, 0, 0, 1]]
                )
                assert np.all(params._labels == np.array(["-", "+"]))

    def test_dict_cmap_rev(self):
        params = _ScatterParams(
            x=self.x,
            y=self.y,
            c=np.where(self.c > 0, "+", "-"),
            cmap={"-": "k", "+": "r"},
        )
        if sys.version_info[1] > 5:
            np.testing.assert_equal(params.cmap.colors, [[0, 0, 0, 1], [1, 0, 0, 1]])
            assert np.all(params._labels == np.array(["-", "+"]))
        else:
            try:
                np.testing.assert_equal(
                    params.cmap.colors, [[0, 0, 0, 1], [1, 0, 0, 1]]
                )
                assert np.all(params._labels == np.array(["-", "+"]))
            except AssertionError:
                np.testing.assert_equal(
                    params.cmap.colors, [[1, 0, 0, 1], [0, 0, 0, 1]]
                )
                assert np.all(params._labels == np.array(["+", "-"]))

    def test_dict_cmap_constant(self):
        params = _ScatterParams(
            x=self.x,
            y=self.y,
            c=np.full_like(self.c, "+", dtype=str),
            cmap={"-": "k", "+": "r"},
        )
        np.testing.assert_equal(params.cmap.colors, [[1, 0, 0, 1]])
        assert np.all(params._labels == np.array(["+"]))

    def test_cmap_given(self):
        params = _ScatterParams(x=self.x, y=self.y, c=self.c, cmap="viridis")
        assert params.cmap is matplotlib.cm.viridis
        assert not params.list_cmap()

    def test_cmap_scale_symlog(self):
        params = _ScatterParams(x=self.x, y=self.y, c=self.c, cmap_scale="symlog")
        assert params.cmap_scale == "symlog"
        assert isinstance(params.norm, matplotlib.colors.SymLogNorm)

    def test_cmap_scale_log(self):
        params = _ScatterParams(
            x=self.x, y=self.y, c=np.abs(self.c) + 1, cmap_scale="log"
        )
        assert params.cmap_scale == "log"
        assert isinstance(params.norm, matplotlib.colors.LogNorm)

    def test_cmap_scale_sqrt(self):
        params = _ScatterParams(x=self.x, y=self.y, c=self.c, cmap_scale="sqrt")
        assert params.cmap_scale == "sqrt"
        assert isinstance(params.norm, matplotlib.colors.PowerNorm)
        assert params.norm.gamma == 0.5

    def test_extend(self):
        params = _ScatterParams(x=self.x, y=self.y, c=self.c, vmin=np.mean(self.c))
        assert params.extend == "min"
        params = _ScatterParams(x=self.x, y=self.y, c=self.c, vmax=np.mean(self.c))
        assert params.extend == "max"
        params = _ScatterParams(
            x=self.x,
            y=self.y,
            c=self.c,
            vmin=(np.min(self.c) + np.mean(self.c)) / 2,
            vmax=(np.max(self.c) + np.mean(self.c)) / 2,
        )
        assert params.extend == "both"
        params = _ScatterParams(x=self.x, y=self.y, c=self.c)
        assert params.extend == "neither"

    def test_check_vmin_vmax(self):
        utils.assert_warns_message(
            UserWarning,
            "Cannot set `vmin` or `vmax` with constant `c=None`. "
            "Setting `vmin = vmax = None`.",
            _ScatterParams,
            x=self.x,
            y=self.y,
            vmin=0,
        )
        utils.assert_warns_message(
            UserWarning,
            "Cannot set `vmin` or `vmax` with discrete data. " "Setting to `None`.",
            _ScatterParams,
            x=self.x,
            y=self.y,
            c=np.where(self.c > 0, "+", "-"),
            vmin=0,
        )

    def test_check_legend(self):
        utils.assert_raises_message(
            ValueError,
            "Received conflicting values for synonyms "
            "`legend=True` and `colorbar=False`",
            _ScatterParams,
            x=self.x,
            y=self.y,
            legend=True,
            colorbar=False,
        )
        utils.assert_warns_message(
            UserWarning,
            "`c` is a color array and cannot be used to create a "
            "legend. To interpret these values as labels instead, "
            "provide a `cmap` dictionary with label-color pairs.",
            _ScatterParams,
            x=self.x,
            y=self.y,
            c=self.array_c,
            legend=True,
        )
        utils.assert_warns_message(
            UserWarning,
            "Cannot create a legend with constant `c=None`",
            _ScatterParams,
            x=self.x,
            y=self.y,
            c=None,
            legend=True,
        )

    def test_check_size(self):
        utils.assert_raises_message(
            ValueError,
            "Expected all axes of data to have the same length" ". Got [500, 100]",
            _ScatterParams,
            x=self.x,
            y=self.y[:100],
        )
        utils.assert_raises_message(
            ValueError,
            "Expected all axes of data to have the same length" ". Got [500, 500, 100]",
            _ScatterParams,
            x=self.x,
            y=self.y,
            z=self.z[:100],
        )

    def test_check_c(self):
        utils.assert_raises_message(
            ValueError,
            "Expected c of length 500 or 1. Got 100",
            _ScatterParams,
            x=self.x,
            y=self.y,
            c=self.c[:100],
        )

    def test_check_discrete(self):
        utils.assert_raises_message(
            ValueError,
            "Cannot treat non-numeric data as continuous.",
            _ScatterParams,
            x=self.x,
            y=self.y,
            c=np.where(self.c > 0, "+", "-"),
            discrete=False,
        )

    def test_check_cmap(self):
        utils.assert_raises_message(
            ValueError,
            "Expected list-like `c` with dictionary cmap." " Got <class 'str'>",
            _ScatterParams,
            x=self.x,
            y=self.y,
            c="black",
            cmap={"+": "k", "-": "r"},
        )
        utils.assert_raises_message(
            ValueError,
            "Cannot use dictionary cmap with " "continuous data.",
            _ScatterParams,
            x=self.x,
            y=self.y,
            c=self.c,
            discrete=False,
            cmap={"+": "k", "-": "r"},
        )
        utils.assert_raises_message(
            ValueError,
            "Dictionary cmap requires a color "
            "for every unique entry in `c`. "
            "Missing colors for [+]",
            _ScatterParams,
            x=self.x,
            y=self.y,
            c=np.where(self.c > 0, "+", "-"),
            cmap={"-": "r"},
        )
        utils.assert_raises_message(
            ValueError,
            "Expected list-like `c` with list cmap. " "Got <class 'str'>",
            _ScatterParams,
            x=self.x,
            y=self.y,
            c="black",
            cmap=["k", "r"],
        )

    def test_check_cmap_scale(self):
        utils.assert_warns_message(
            UserWarning,
            "Cannot use non-linear `cmap_scale` with " "`c` as a color array.",
            _ScatterParams,
            x=self.x,
            y=self.y,
            c=self.array_c,
            cmap_scale="log",
        )
        utils.assert_warns_message(
            UserWarning,
            "Cannot use non-linear `cmap_scale` with constant " "`c=black`.",
            _ScatterParams,
            x=self.x,
            y=self.y,
            c="black",
            cmap_scale="log",
        )
        utils.assert_warns_message(
            UserWarning,
            "Cannot use non-linear `cmap_scale` with discrete data.",
            _ScatterParams,
            x=self.x,
            y=self.y,
            cmap_scale="log",
            c=np.where(self.c > 0, "+", "-"),
        )

    def test_series_labels(self):
        params = _ScatterParams(x=pd.Series(self.x, name="x"), y=self.y, c=self.c)
        assert params.xlabel == "x"
        assert params.ylabel is None
        assert params.zlabel is None
        params = _ScatterParams(x=self.x, y=pd.Series(self.y, name="y"), c=self.c)
        assert params.xlabel is None
        assert params.ylabel == "y"
        assert params.zlabel is None
        params = _ScatterParams(
            x=self.x, y=self.y, z=pd.Series(self.y, name="z"), c=self.c
        )
        assert params.xlabel is None
        assert params.ylabel is None
        assert params.zlabel == "z"
        # xlabel overrides series
        params = _ScatterParams(
            x=pd.Series(self.x, name="x"), y=self.y, c=self.c, xlabel="y"
        )
        assert params.xlabel == "y"
        assert params.ylabel is None
        assert params.zlabel is None
        # label_prefix overrides series
        params = _ScatterParams(
            x=pd.Series(self.x, name="x"), y=self.y, c=self.c, label_prefix="y"
        )
        assert params.xlabel == "y1"
        assert params.ylabel == "y2"
        assert params.zlabel is None
        # xlabel overrides label_prefix
        params = _ScatterParams(
            x=pd.Series(self.x, name="x"),
            y=self.y,
            z=self.y,
            c=self.c,
            label_prefix="y",
            xlabel="test",
        )
        assert params.xlabel == "test"
        assert params.ylabel == "y2"
        assert params.zlabel == "y3"

    def test_jitter_x(self):
        params = _JitterParams(x=np.where(self.x > 0, "+", "-"), y=self.y)
        np.testing.assert_array_equal(params.x_labels, ["+", "-"])
        np.testing.assert_array_equal(
            params.x_coords, np.where(self.x > 0, 0, 1)[params.plot_idx]
        )


class Test10X(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.X = data.load_10X(sparse=False)
        self.X_filt = scprep.filter.filter_empty_cells(self.X)
        self.X_pca, self.S = scprep.reduce.pca(
            scprep.utils.toarray(self.X), n_components=10, return_singular_values=True
        )

    @classmethod
    def tearDownClass(self):
        try_remove("test.png")
        try_remove("test.gif")
        try_remove("test.mp4")
        try_remove("test_jitter.png")
        try_remove("test_histogram.png")
        try_remove("test_library_size.png")
        try_remove("test_variable_genes.png")
        try_remove("test_gene_expression.png")
        try_remove("test_scree.png")

    def tearDown(self):
        plt.close("all")

    def test_histogram(self):
        scprep.plot.plot_library_size(self.X_filt, cutoff=1000, log=True)
        scprep.plot.plot_library_size(
            self.X_filt, cutoff=1000, log=True, xlabel="x label", ylabel="y label"
        )

    def test_histogram_list_of_lists(self):
        scprep.plot.plot_library_size(scprep.utils.toarray(self.X_filt).tolist())

    def test_histogram_array(self):
        scprep.plot.plot_library_size(scprep.utils.toarray(self.X_filt))

    def test_histogram_multiple(self):
        scprep.plot.histogram(
            [scprep.select.select_rows(self.X, idx=0), [1, 2, 2, 2, 3]],
            color=["r", "b"],
        )

    def test_histogram_multiple_cutoff(self):
        scprep.plot.plot_library_size(self.X_filt, cutoff=[500, 1000], log=True)

    def test_histogram_multiple_percentile(self):
        scprep.plot.plot_library_size(self.X_filt, percentile=[10, 90], log=True)

    def test_histogram_log_negative_min(self):
        scprep.plot.histogram([-1, 1, 1, 1], log="x")
        scprep.plot.histogram([-1, 1, 1, 1], log=True)
        scprep.plot.histogram([-1, -0.1, -0.1, 1], log="x")
        scprep.plot.histogram([-1, -0.1, -0.1, 1], log=True)

    def test_histogram_log_negative_max(self):
        scprep.plot.histogram([-1, -1, -1, -1], log="x")
        scprep.plot.histogram([-1, -1, -1, -1], log=True)
        scprep.plot.histogram([-1, -1, -1, -2], log="x")
        scprep.plot.histogram([-1, -1, -1, -2], log=True)

    def test_histogram_log_zero_min(self):
        scprep.plot.histogram([0, 1, 1, 1], log="x")
        scprep.plot.histogram([0, 1, 1, 1], log=True)
        scprep.plot.histogram([0, 0, -0.1, 1], log="x")
        scprep.plot.histogram([0, 0, -0.1, 1], log=True)

    def test_histogram_log_zero_max(self):
        scprep.plot.histogram([-1, -1, 0, -1], log="x")
        scprep.plot.histogram([-1, -1, 0, -1], log=True)
        scprep.plot.histogram([-1, -1, 0, -2], log="x")
        scprep.plot.histogram([-1, -1, 0, -2], log=True)

    def test_plot_library_size_multiple(self):
        scprep.plot.plot_library_size(
            [
                self.X_filt,
                scprep.select.select_rows(
                    self.X_filt, idx=np.arange(self.X_filt.shape[0] // 2)
                ),
            ],
            color=["r", "b"],
            filename="test_library_size.png",
        )
        assert os.path.exists("test_library_size.png")

    def test_plot_gene_set_expression_multiple(self):
        scprep.plot.plot_gene_set_expression(
            [
                self.X,
                scprep.select.select_rows(self.X, idx=np.arange(self.X.shape[0] // 2)),
            ],
            starts_with="D",
            color=["r", "b"],
        )

    def test_gene_set_expression_list_of_lists(self):
        scprep.plot.plot_gene_set_expression(
            scprep.utils.toarray(self.X).tolist(), genes=[0, 1]
        )

    def test_gene_set_expression_array(self):
        scprep.plot.plot_gene_set_expression(scprep.utils.toarray(self.X), genes=[0, 1])

    def test_plot_gene_set_expression_single_gene(self):
        scprep.plot.plot_gene_set_expression(
            self.X, color=["red"], genes="Arl8b", filename="test_gene_expression.png"
        )
        assert os.path.exists("test_gene_expression.png")

    def test_plot_variable_genes(self):
        scprep.plot.plot_gene_variability(self.X, filename="test_variable_genes.png")
        assert os.path.exists("test_variable_genes.png")

    def test_variable_genes_list_of_lists(self):
        scprep.plot.plot_gene_variability(scprep.utils.toarray(self.X).tolist())

    def test_histogram_single_gene_dataframe(self):
        scprep.plot.histogram(
            scprep.select.select_cols(self.X, idx=["Arl8b"]), color=["red"]
        )

    def test_histogram_single_gene_series(self):
        scprep.plot.histogram(
            scprep.select.select_cols(self.X, idx="Arl8b"), color=["red"]
        )

    def test_histogram_custom_axis(self):
        fig, ax = plt.subplots()
        scprep.plot.plot_gene_set_expression(
            self.X,
            genes=scprep.select.get_gene_set(self.X, starts_with="D"),
            percentile=90,
            log="y",
            ax=ax,
            title="histogram",
            filename="test_histogram.png",
        )
        assert os.path.exists("test_histogram.png")
        assert ax.get_title() == "histogram"

    def test_histogram_invalid_axis(self):
        utils.assert_raises_message(
            TypeError,
            "Expected ax as a matplotlib.axes.Axes. Got ",
            scprep.plot.plot_library_size,
            self.X,
            ax="invalid",
        )

    def test_scree(self):
        ax = scprep.plot.scree_plot(self.S)
        assert all([t == int(t) for t in ax.get_xticks()]), ax.get_xticks()
        ax = scprep.plot.scree_plot(
            self.S,
            cumulative=True,
            xlabel="x label",
            ylabel="y label",
            filename="test_scree.png",
        )
        assert all([t == int(t) for t in ax.get_xticks()]), ax.get_xticks()
        assert os.path.isfile("test_scree.png")

    def test_scree_custom_axis(self):
        fig, ax = plt.subplots()
        scprep.plot.scree_plot(self.S, ax=ax)
        assert all([t == int(t) for t in ax.get_xticks()]), ax.get_xticks()

    def test_scree_invalid_axis(self):
        utils.assert_raises_message(
            TypeError,
            "Expected ax as a matplotlib.axes.Axes. Got ",
            scprep.plot.scree_plot,
            self.S,
            ax="invalid",
        )

    def test_scatter_continuous(self):
        scprep.plot.scatter2d(
            self.X_pca, c=self.X_pca[:, 0], legend_title="test", title="title test"
        )

    def test_scatter2d_one_point(self):
        scprep.plot.scatter2d(self.X_pca[0], c=["red"])

    def test_scatter3d_one_point(self):
        scprep.plot.scatter3d(self.X_pca[0], c=["red"])

    def test_scatter_discrete(self):
        ax = scprep.plot.scatter2d(
            self.X_pca,
            c=np.random.choice(["hello", "world"], self.X_pca.shape[0], replace=True),
            legend_title="test",
            legend_loc="center left",
            legend_anchor=(1.02, 0.5),
        )
        assert ax.get_legend().get_title().get_text() == "test"

    def test_scatter_discrete_str_int(self):
        ax = scprep.plot.scatter2d(
            self.X_pca,
            c=np.random.choice(["1", "2", "3"], self.X_pca.shape[0], replace=True),
            legend_title="test",
            legend_loc="center left",
            legend_anchor=(1.02, 0.5),
        )
        assert ax.get_legend().get_title().get_text() == "test"

    def test_jitter_discrete(self):
        ax = scprep.plot.jitter(
            np.where(self.X_pca[:, 0] > 0, "+", "-"),
            self.X_pca[:, 1],
            c=np.random.choice(["hello", "world"], self.X_pca.shape[0], replace=True),
            legend_title="test",
            title="jitter",
            filename="test_jitter.png",
        )
        assert os.path.exists("test_jitter.png")
        assert ax.get_legend().get_title().get_text() == "test"
        assert ax.get_title() == "jitter"
        assert ax.get_xlim() == (-0.5, 1.5)
        assert [t.get_text() for t in ax.get_xticklabels()] == ["+", "-"]

    def test_jitter_continuous(self):
        ax = scprep.plot.jitter(
            np.where(self.X_pca[:, 0] > 0, "+", "-"),
            self.X_pca[:, 1],
            c=self.X_pca[:, 1],
            title="jitter",
            legend_title="test",
        )
        assert ax.get_figure().get_axes()[1].get_ylabel() == "test"
        assert ax.get_title() == "jitter"
        assert ax.get_xlim() == (-0.5, 1.5)
        assert [t.get_text() for t in ax.get_xticklabels()] == ["+", "-"]

    def test_jitter_axis_labels(self):
        ax = scprep.plot.jitter(
            np.where(self.X_pca[:, 0] > 0, "+", "-"), self.X_pca[:, 1], xlabel="test"
        )
        assert ax.get_xlabel() == "test"
        assert ax.get_ylabel() == ""
        ax = scprep.plot.jitter(
            pd.Series(np.where(self.X_pca[:, 0] > 0, "+", "-"), name="x"),
            pd.Series(self.X_pca[:, 1], name="y"),
            ylabel="override",
        )
        assert ax.get_xlabel() == "x"
        assert ax.get_ylabel() == "override"

    def test_scatter_dict(self):
        scprep.plot.scatter2d(
            self.X_pca,
            c=np.random.choice(["hello", "world"], self.X_pca.shape[0], replace=True),
            cmap={"hello": "red", "world": "green"},
        )

    def test_scatter_dict_c_none(self):
        utils.assert_raises_message(
            ValueError,
            "Expected list-like `c` with dictionary cmap. Got <class 'NoneType'>",
            scprep.plot.scatter2d,
            self.X_pca,
            c=None,
            cmap={"hello": "red", "world": "green"},
        )

    def test_scatter_dict_continuous(self):
        utils.assert_raises_message(
            ValueError,
            "Cannot use dictionary cmap with continuous data",
            scprep.plot.scatter2d,
            self.X_pca,
            c=self.X_pca[:, 0],
            discrete=False,
            cmap={"hello": "red", "world": "green"},
        )

    def test_scatter_dict_missing(self):
        utils.assert_raises_message(
            ValueError,
            "Dictionary cmap requires a color for every unique entry in `c`. "
            "Missing colors for [world]",
            scprep.plot.scatter2d,
            self.X_pca,
            c=np.random.choice(["hello", "world"], self.X_pca.shape[0], replace=True),
            cmap={"hello": "red"},
        )

    def test_scatter_list_discrete(self):
        scprep.plot.scatter2d(
            self.X_pca,
            c=np.random.choice(["hello", "world"], self.X_pca.shape[0], replace=True),
            cmap=["red", "green"],
        )

    def test_scatter_list_discrete_missing(self):
        scprep.plot.scatter2d(
            self.X_pca,
            c=np.random.choice(
                ["hello", "great", "world"], self.X_pca.shape[0], replace=True
            ),
            cmap=["red", "green"],
        )

    def test_scatter_list_continuous(self):
        scprep.plot.scatter2d(self.X_pca, c=self.X_pca[:, 0], cmap=["red", "green"])

    def test_scatter_list_single(self):
        scprep.plot.scatter2d(self.X_pca, c=self.X_pca[:, 0], cmap=["red"])

    def test_scatter_list_c_none(self):
        utils.assert_raises_message(
            ValueError,
            "Expected list-like `c` with list cmap. Got <class 'NoneType'>",
            scprep.plot.scatter2d,
            self.X_pca,
            c=None,
            cmap=["red", "green"],
        )

    def test_scatter_discrete_greater_than_10(self):
        scprep.plot.scatter2d(self.X_pca, c=np.arange(self.X_pca.shape[0]) % 11)

    def test_scatter_solid(self):
        scprep.plot.scatter3d(self.X_pca, c="green")

    def test_scatter_none(self):
        scprep.plot.scatter2d(self.X_pca, c=None)

    def test_scatter_no_ticks(self):
        ax = scprep.plot.scatter3d(self.X_pca, zticks=False)
        assert len(ax.get_zticks()) == 0

    def test_scatter_no_ticklabels(self):
        ax = scprep.plot.scatter3d(self.X_pca, zticklabels=False)
        assert np.all([lab.get_text() == "" for lab in ax.get_zticklabels()])

    def test_scatter_custom_ticks(self):
        ax = scprep.plot.scatter2d(self.X_pca, xticks=[0, 1, 2])
        assert np.all(ax.get_xticks() == np.array([0, 1, 2]))
        ax = scprep.plot.scatter3d(self.X_pca, zticks=False)
        assert np.all(ax.get_zticks() == np.array([]))

    def test_scatter_custom_ticklabels(self):
        ax = scprep.plot.scatter2d(
            self.X_pca, xticks=[0, 1, 2], xticklabels=["a", "b", "c"]
        )
        assert np.all(ax.get_xticks() == np.array([0, 1, 2]))
        xticklabels = np.array([lab.get_text() for lab in ax.get_xticklabels()])
        assert np.all(xticklabels == np.array(["a", "b", "c"]))

    def test_scatter_axis_labels(self):
        ax = scprep.plot.scatter2d(self.X_pca.tolist(), label_prefix="test")
        assert ax.get_xlabel() == "test1"
        assert ax.get_ylabel() == "test2"
        ax = scprep.plot.scatter3d(self.X_pca.tolist(), label_prefix="test")
        assert ax.get_xlabel() == "test1"
        assert ax.get_ylabel() == "test2"
        assert ax.get_zlabel() == "test3"
        ax = scprep.plot.scatter2d(self.X_pca, label_prefix="test", xlabel="override")
        assert ax.get_xlabel() == "override"
        assert ax.get_ylabel() == "test2"
        ax = scprep.plot.scatter(
            x=self.X_pca[:, 0],
            y=pd.Series(self.X_pca[:, 1], name="y"),
            z=pd.Series(self.X_pca[:, 2], name="z"),
            ylabel="override",
        )
        assert ax.get_xlabel() == ""
        assert ax.get_ylabel() == "override"
        assert ax.get_zlabel() == "z"
        ax = scprep.plot.scatter(
            x=self.X_pca[:, 0],
            y=pd.Series(self.X_pca[:, 1], name="y"),
            z=pd.Series(self.X_pca[:, 2], name="z"),
            zlabel="override",
        )
        assert ax.get_xlabel() == ""
        assert ax.get_ylabel() == "y"
        assert ax.get_zlabel() == "override"

    def test_scatter_axis_savefig(self):
        scprep.plot.scatter2d(self.X_pca, filename="test.png")
        assert os.path.exists("test.png")

    def test_scatter_viewinit(self):
        ax = scprep.plot.scatter3d(self.X_pca, elev=80, azim=270)
        assert ax.elev == 80
        assert ax.azim == 270

    def test_scatter3d_data_2d(self):
        utils.assert_raises_message(
            ValueError,
            "Expected data.shape[1] >= 3. Got 2",
            scprep.plot.scatter3d,
            self.X_pca[:, :2],
        )

    def test_scatter3d_data_2d_list(self):
        utils.assert_raises_message(
            ValueError,
            "Expected data.shape[1] >= 3. Got 2",
            scprep.plot.scatter3d,
            self.X_pca[:, :2].tolist(),
        )

    def test_scatter_rotate_gif(self):
        scprep.plot.rotate_scatter3d(self.X_pca, fps=3, dpi=20, filename="test.gif")
        assert os.path.exists("test.gif")

    def test_scatter_rotate_mp4(self):
        scprep.plot.rotate_scatter3d(self.X_pca, fps=3, dpi=20, filename="test.mp4")
        assert os.path.exists("test.mp4")

    def test_scatter_rotate_invalid_filename(self):
        utils.assert_raises_message(
            ValueError,
            "filename must end in .gif or .mp4. Got test.invalid",
            scprep.plot.rotate_scatter3d,
            self.X_pca,
            fps=3,
            dpi=20,
            filename="test.invalid",
        )

    def test_scatter_invalid_data(self):
        utils.assert_raises_message(
            ValueError,
            "Expected all axes of data to have the same length. "
            "Got {}".format([self.X_pca.shape[0], self.X_pca.shape[1]]),
            scprep.plot.scatter,
            x=self.X_pca[:, 0],
            y=self.X_pca[0, :],
        )
        utils.assert_raises_message(
            ValueError,
            "Expected all axes of data to have the same length. "
            "Got {}".format(
                [self.X_pca.shape[0], self.X_pca.shape[0], self.X_pca.shape[1]]
            ),
            scprep.plot.scatter,
            x=self.X_pca[:, 0],
            y=self.X_pca[:, 0],
            z=self.X_pca[0, :],
        )

    def test_scatter_invalid_c(self):
        utils.assert_raises_message(
            ValueError,
            "Expected c of length {} or 1. Got {}".format(
                self.X_pca.shape[0], self.X_pca.shape[1]
            ),
            scprep.plot.scatter2d,
            self.X_pca,
            c=self.X_pca[0, :],
        )

    def test_scatter_invalid_s(self):
        utils.assert_raises_message(
            ValueError,
            "Expected s of length {} or 1. Got {}".format(
                self.X_pca.shape[0], self.X_pca.shape[1]
            ),
            scprep.plot.scatter2d,
            self.X_pca,
            s=self.X_pca[0, :],
        )

    def test_scatter_invalid_mask(self):
        utils.assert_raises_message(
            ValueError,
            "Expected mask of length {}. Got {}".format(
                self.X_pca.shape[0], self.X_pca.shape[1]
            ),
            scprep.plot.scatter2d,
            self.X_pca,
            mask=self.X_pca[0, :] > 0,
        )

    def test_scatter_invalid_discrete(self):
        utils.assert_raises_message(
            ValueError,
            "Cannot treat non-numeric data as continuous",
            scprep.plot.scatter2d,
            self.X_pca,
            discrete=False,
            c=np.random.choice(["hello", "world"], self.X_pca.shape[0], replace=True),
        )

    def test_scatter_invalid_legend(self):
        utils.assert_warns_message(
            UserWarning,
            "`c` is a color array and cannot be used to create a "
            "legend. To interpret these values as labels instead, "
            "provide a `cmap` dictionary with label-color pairs.",
            scprep.plot.scatter2d,
            self.X_pca,
            legend=True,
            c=np.random.choice(["red", "blue"], self.X_pca.shape[0], replace=True),
        )
        utils.assert_warns_message(
            UserWarning,
            "Cannot create a legend with constant `c=red`",
            scprep.plot.scatter2d,
            self.X_pca,
            legend=True,
            c="red",
        )
        utils.assert_warns_message(
            UserWarning,
            "Cannot create a legend with constant `c=None`",
            scprep.plot.scatter2d,
            self.X_pca,
            legend=True,
            c=None,
        )

    def test_scatter_invalid_axis(self):
        fig, ax = plt.subplots()
        utils.assert_raises_message(
            TypeError,
            "Expected ax with projection='3d'. " "Got 2D axis instead.",
            scprep.plot.scatter3d,
            self.X_pca,
            ax=ax,
        )

    def test_scatter_colorbar(self):
        scprep.plot.scatter3d(self.X_pca, c=self.X_pca[:, 0], colorbar=True)

    def test_scatter_colorbar_log(self):
        scprep.plot.scatter2d(
            self.X_pca,
            c=np.abs(self.X_pca[:, 0]) + 1e-7,
            colorbar=True,
            cmap_scale="log",
        )

    def test_scatter_colorbar_log_constant_c(self):
        utils.assert_warns_message(
            UserWarning,
            "Cannot use non-linear `cmap_scale` with constant `c=blue`",
            scprep.plot.scatter2d,
            self.X_pca,
            c="blue",
            colorbar=True,
            cmap_scale="log",
        )

    def test_scatter_colorbar_log_discrete(self):
        utils.assert_warns_message(
            UserWarning,
            "Cannot use non-linear `cmap_scale` with discrete data.",
            scprep.plot.scatter2d,
            self.X_pca,
            c=np.random.choice(["hello", "world"], self.X_pca.shape[0]),
            colorbar=True,
            cmap_scale="log",
        )

    def test_scatter_colorbar_log_negative(self):
        utils.assert_raises_message(
            ValueError,
            "`vmin` must be positive for `cmap_scale='log'`. "
            "Got {}".format(self.X_pca[:, 0].min()),
            scprep.plot.scatter2d,
            self.X_pca,
            c=self.X_pca[:, 0],
            colorbar=True,
            cmap_scale="log",
        )

    def test_scatter_colorbar_symlog(self):
        scprep.plot.scatter2d(
            self.X_pca, c=self.X_pca[:, 0], colorbar=True, cmap_scale="symlog"
        )

    def test_scatter_colorbar_sqrt(self):
        scprep.plot.scatter2d(
            self.X_pca, c=self.X_pca[:, 0], colorbar=True, cmap_scale="sqrt"
        )

    def test_scatter_colorbar_invalid(self):
        utils.assert_raises_message(
            ValueError,
            "Expected norm in ['linear', 'log', 'symlog',"
            "'sqrt'] or a matplotlib.colors.Normalize object."
            " Got invalid",
            scprep.plot.scatter2d,
            self.X_pca,
            c=self.X_pca[:, 0],
            colorbar=True,
            cmap_scale="invalid",
        )

    def test_scatter_legend_and_colorbar(self):
        utils.assert_raises_message(
            ValueError,
            "Received conflicting values for synonyms "
            "`legend=True` and `colorbar=False`",
            scprep.plot.scatter2d,
            self.X_pca,
            c=self.X_pca[:, 0],
            legend=True,
            colorbar=False,
        )

    def test_scatter_vmin_vmax(self):
        scprep.plot.scatter2d(self.X_pca, c=self.X_pca[:, 0], vmin=1, vmax=2)

    def test_scatter_vmin_vmax_discrete(self):
        utils.assert_warns_message(
            UserWarning,
            "Cannot set `vmin` or `vmax` with discrete data. " "Setting to `None`.",
            scprep.plot.scatter3d,
            self.X_pca,
            c=np.random.choice(["hello", "world"], self.X_pca.shape[0], replace=True),
            vmin=1,
            vmax=2,
        )

    def test_scatter_vmin_vmax_solid_color(self):
        utils.assert_warns_message(
            UserWarning,
            "Cannot set `vmin` or `vmax` with constant `c=red`. "
            "Setting `vmin = vmax = None`.",
            scprep.plot.scatter3d,
            self.X_pca,
            c="red",
            vmin=1,
            vmax=2,
        )

    def test_generate_colorbar_n_ticks(self):
        cb = scprep.plot.tools.generate_colorbar("inferno", vmin=0, vmax=1, n_ticks=4)
        assert len(cb.get_ticks()) == 4

    def test_generate_colorbar_vmin_vmax_none(self):
        cb = scprep.plot.tools.generate_colorbar("inferno")
        utils.assert_warns_message(
            UserWarning,
            "Cannot set `n_ticks` without setting `vmin` and `vmax`.",
            scprep.plot.tools.generate_colorbar,
            n_ticks=4,
        )

    def test_generate_colorbar_mappable(self):
        im = plt.imshow([np.arange(10), np.arange(10)])
        scprep.plot.tools.generate_colorbar(mappable=im)
        utils.assert_warns_message(
            UserWarning,
            "Cannot set `vmin` or `vmax` when `mappable` is given.",
            scprep.plot.tools.generate_colorbar,
            mappable=im,
            vmin=10,
            vmax=20,
        )
        utils.assert_warns_message(
            UserWarning,
            "Cannot set `cmap` when `mappable` is given.",
            scprep.plot.tools.generate_colorbar,
            mappable=im,
            cmap="inferno",
        )
        utils.assert_warns_message(
            UserWarning,
            "Cannot set `scale` when `mappable` is given.",
            scprep.plot.tools.generate_colorbar,
            mappable=im,
            scale="log",
        )

    def test_generate_colorbar_vmin_none_vmax_given(self):
        utils.assert_raises_message(
            ValueError,
            "Either both or neither of `vmax` and `vmin` should be set. "
            "Got `vmax=None, vmin=0`",
            scprep.plot.tools.generate_colorbar,
            "inferno",
            vmin=0,
        )

    def test_marker_plot_dict(self):
        scprep.plot.marker_plot(
            data=self.X,
            clusters=np.random.choice(
                np.arange(10), replace=True, size=self.X.shape[0]
            ),
            gene_names=self.X.columns,
            markers={"tissue": self.X.columns[:2], "other tissue": self.X.columns[2:4]},
        )

    def test_marker_plot_single_marker(self):
        scprep.plot.marker_plot(
            data=self.X,
            clusters=np.random.choice(
                np.arange(10), replace=True, size=self.X.shape[0]
            ),
            gene_names=self.X.columns,
            markers={
                "tissue": [self.X.columns[0]],
                "other tissue": self.X.columns[2:4],
            },
        )

    def test_marker_plot_repeat_marker(self):
        scprep.plot.marker_plot(
            data=self.X,
            clusters=np.random.choice(
                np.arange(10), replace=True, size=self.X.shape[0]
            ),
            gene_names=self.X.columns,
            markers={"tissue": self.X.columns[:3], "other tissue": self.X.columns[2:4]},
        )

    def test_marker_plot_list(self):
        scprep.plot.marker_plot(
            data=self.X,
            clusters=np.random.choice(
                np.arange(10), replace=True, size=self.X.shape[0]
            ),
            markers=self.X.columns,
            normalize_emd=False,
            normalize_expression=False,
        )

    def test_marker_plot_bad_gene_names(self):
        utils.assert_raises_message(
            ValueError,
            "All genes in `markers` must appear "
            "in gene_names. Did not find: {}".format("z"),
            scprep.plot.marker_plot,
            data=self.X,
            clusters=np.random.choice(
                np.arange(10), replace=True, size=self.X.shape[0]
            ),
            gene_names=self.X.columns,
            markers={"tissue": ["z"]},
        )

    def test_marker_plot_pandas_gene_names(self):
        scprep.plot.marker_plot(
            data=self.X,
            clusters=np.random.choice(
                np.arange(10), replace=True, size=self.X.shape[0]
            ),
            markers={"tissue": self.X.columns[:2], "other tissue": self.X.columns[2:4]},
            reorder_tissues=False,
            reorder_markers=False,
        )

    def test_marker_plot_no_gene_names(self):
        utils.assert_raises_message(
            ValueError,
            "Either `data` must be a pd.DataFrame, or gene_names must "
            "be provided. "
            "Got gene_names=None, data as a <class 'numpy.ndarray'>",
            scprep.plot.marker_plot,
            data=self.X.to_numpy(),
            clusters=np.random.choice(
                np.arange(10), replace=True, size=self.X.shape[0]
            ),
            markers={"tissue": ["z"]},
        )

    def test_label_axis_va(self):
        ax = scprep.plot.scatter2d(self.X_pca)
        scprep.plot.tools.label_axis(ax.yaxis, ticklabel_vertical_alignment="top")
        for tick in ax.yaxis.get_ticklabels():
            assert tick.get_va() == "top"
        scprep.plot.tools.label_axis(ax.yaxis, ticklabel_vertical_alignment="bottom")
        for tick in ax.yaxis.get_ticklabels():
            assert tick.get_va() == "bottom"

    def test_label_axis_ha(self):
        ax = scprep.plot.scatter2d(self.X_pca)
        scprep.plot.tools.label_axis(ax.xaxis, ticklabel_horizontal_alignment="left")
        for tick in ax.xaxis.get_ticklabels():
            assert tick.get_ha() == "left"
        scprep.plot.tools.label_axis(ax.xaxis, ticklabel_horizontal_alignment="right")
        for tick in ax.xaxis.get_ticklabels():
            assert tick.get_ha() == "right"

    def test_label_axis_rotation(self):
        ax = scprep.plot.scatter2d(self.X_pca)
        scprep.plot.tools.label_axis(ax.xaxis, ticklabel_rotation=45)
        for tick in ax.xaxis.get_ticklabels():
            assert tick.get_rotation() == 45
        scprep.plot.tools.label_axis(ax.xaxis, ticklabel_rotation=90)
        for tick in ax.xaxis.get_ticklabels():
            assert tick.get_rotation() == 90
