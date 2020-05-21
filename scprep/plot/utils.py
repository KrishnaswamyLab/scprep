import numpy as np
import platform

from .. import utils

from .._lazyload import matplotlib as mpl
from .._lazyload import mpl_toolkits

plt = mpl.pyplot


def _with_default(param, default):
    return param if param is not None else default


def _mpl_is_gui_backend():
    backend = mpl.get_backend()
    if backend in ["module://ipykernel.pylab.backend_inline", "agg"]:
        return False
    else:
        return True


def _get_figure(ax=None, figsize=None, subplot_kw=None):
    if subplot_kw is None:
        subplot_kw = {}
    if ax is None:
        if "projection" in subplot_kw and subplot_kw["projection"] == "3d":
            # ensure mplot3d is loaded
            mpl_toolkits.mplot3d.Axes3D
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=subplot_kw)
        show_fig = True
    else:
        try:
            fig = ax.get_figure()
        except AttributeError as e:
            if not isinstance(ax, mpl.axes.Axes):
                raise TypeError(
                    "Expected ax as a matplotlib.axes.Axes. " "Got {}".format(type(ax))
                )
            else:
                raise e
        if "projection" in subplot_kw:
            if subplot_kw["projection"] == "3d" and not isinstance(
                ax, mpl_toolkits.mplot3d.Axes3D
            ):
                raise TypeError(
                    "Expected ax with projection='3d'. " "Got 2D axis instead."
                )
        show_fig = False
    return fig, ax, show_fig


def _is_color_array(c):
    if c is None:
        return False
    else:
        try:
            c_rgb = mpl.colors.to_rgba_array(c)
            if np.any(c_rgb > 1):
                # possibly https://github.com/matplotlib/matplotlib/issues/13912
                for i in np.argwhere(c_rgb > 1).flatten():
                    if isinstance(c[i], str):
                        return False
            return True
        except ValueError:
            return False


def _in_ipynb():
    """Check if we are running in a Jupyter Notebook

    Credit to https://stackoverflow.com/a/24937408/3996580
    """
    __VALID_NOTEBOOKS = [
        "<class 'google.colab._shell.Shell'>",
        "<class 'ipykernel.zmqshell.ZMQInteractiveShell'>",
    ]
    try:
        return str(type(get_ipython())) in __VALID_NOTEBOOKS
    except NameError:
        return False


@utils._with_pkg(pkg="matplotlib", min_version=3)
def show(fig):
    """Show a matplotlib Figure correctly, regardless of platform

    If running a Jupyter notebook, we avoid running `fig.show`. If running
    in Windows, it is necessary to run `plt.show` rather than `fig.show`.

    Parameters
    ----------
    fig : matplotlib.Figure
        Figure to show
    """
    fig.tight_layout()
    if _mpl_is_gui_backend():
        if platform.system() == "Windows":
            plt.show(block=True)
        else:
            fig.show()


def _is_default_matplotlibrc():
    __defaults = {
        "axes.labelsize": "medium",
        "axes.titlesize": "large",
        "figure.titlesize": "large",
        "legend.fontsize": "medium",
        "legend.title_fontsize": None,
        "xtick.labelsize": "medium",
        "ytick.labelsize": "medium",
    }
    for k, v in __defaults.items():
        if plt.rcParams[k] != v:
            return False
    return True


def parse_fontsize(size=None, default=None):
    """Parse the user's input font size

    Returns `size` if explicitly set by user,
    `default` if not set by user and the user's matplotlibrc is also default,
    or `None` otherwise (falling back to mpl defaults)

    Parameters
    ----------
    size
        Fontsize explicitly set by user
    default
        Desired default font size in
        xx-small, x-small, small, medium, large, x-large, xx-large,
        larger, smaller
    """
    if size is not None:
        return size
    elif _is_default_matplotlibrc():
        return default
    else:
        return None


class temp_fontsize(object):
    def __init__(self, size=None):
        if size is None:
            size = plt.rcParams["font.size"]
        self.size = size

    def __enter__(self):
        self.old_size = plt.rcParams["font.size"]
        plt.rcParams["font.size"] = self.size

    def __exit__(self, type, value, traceback):
        plt.rcParams["font.size"] = self.old_size


@utils._with_pkg(pkg="matplotlib", min_version=3)
def shift_ticklabels(axis, dx=0, dy=0):
    """Shifts ticklabels on an axis

    Parameters
    ----------
    axis : matplotlib.axis.{X,Y}Axis, mpl_toolkits.mplot3d.axis3d.{X,Y,Z}Axis
        Axis on which to draw labels and ticks
    dx : float, optional (default: 0)
        Horizontal shift
    dy : float, optional (default: 0)
    """
    # Create offset transform by 5 points in x direction
    offset = mpl.transforms.ScaledTranslation(dx, dy, axis.get_figure().dpi_scale_trans)
    # apply offset transform to all ticklabels.
    for label in axis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)
