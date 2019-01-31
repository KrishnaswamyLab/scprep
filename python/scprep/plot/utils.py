import numpy as np
from decorator import decorator
import platform
try:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    pass


@decorator
def _with_matplotlib(fun, *args, **kwargs):
    try:
        plt
    except NameError:
        raise ImportError(
            "matplotlib not found. "
            "Please install it with e.g. `pip install --user matplotlib`")
    return fun(*args, **kwargs)


def _mpl_is_gui_backend():
    backend = mpl.get_backend()
    if backend in ['module://ipykernel.pylab.backend_inline', 'agg']:
        return False
    else:
        return True


def _get_figure(ax=None, figsize=None, subplot_kw=None):
    if subplot_kw is None:
        subplot_kw = {}
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=subplot_kw)
        show_fig = True
    else:
        try:
            fig = ax.get_figure()
        except AttributeError as e:
            if not isinstance(ax, mpl.axes.Axes):
                raise TypeError("Expected ax as a matplotlib.axes.Axes. "
                                "Got {}".format(type(ax)))
            else:
                raise e
        if 'projection' in subplot_kw:
            if subplot_kw['projection'] == '3d' and not isinstance(ax, Axes3D):
                raise TypeError("Expected ax with projection='3d'. "
                                "Got 2D axis instead.")
        show_fig = False
    return fig, ax, show_fig


def _is_color_array(c):
    return c is not None and np.all([mpl.colors.is_color_like(val) for val in c])


def _in_ipynb():
    """Check if we are running in a Jupyter Notebook

    Credit to https://stackoverflow.com/a/24937408/3996580
    """
    __VALID_NOTEBOOKS = ["<class 'google.colab._shell.Shell'>",
                         "<class 'ipykernel.zmqshell.ZMQInteractiveShell'>"]
    try:
        return str(type(get_ipython())) in __VALID_NOTEBOOKS
    except NameError:
        return False


@_with_matplotlib
def show(fig):
    """Show a matplotlib Figure correctly, regardless of platform

    If running a Jupyter notebook, we avoid running `fig.show`. If running
    in Windows, it is necessary to run `plt.show` rather than `fig.show`.

    Parameters
    ----------
    fig : matplotlib.Figure
        Figure to show
    """
    if _mpl_is_gui_backend():
        plt.tight_layout()
        if platform.system() == "Windows":
            plt.show(block=True)
        else:
            fig.show()
