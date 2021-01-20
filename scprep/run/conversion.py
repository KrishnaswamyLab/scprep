import numpy as np
import warnings

from .. import utils
from .._lazyload import rpy2, anndata2ri


def _rpylist2py(robject):
    if not isinstance(robject, rpy2.robjects.vectors.ListVector):
        raise NotImplementedError
    names = rpy2py(robject.names)
    if names is None or len(names) > len(np.unique(names)):
        # list
        robject = np.array([rpy2py(obj) for obj in robject])
    else:
        # dictionary
        robject = {name: rpy2py(obj) for name, obj in zip(robject.names, robject)}
    return robject


def _rpynull2py(robject):
    if robject is rpy2.rinterface.NULL:
        robject = None
    return robject


def _pynull2rpy(pyobject):
    if pyobject is None:
        pyobject = rpy2.rinterface.NULL
    return pyobject


def _rpysce2py(robject):
    if utils._try_import("anndata2ri"):
        robject = anndata2ri.rpy2py(robject)
        if hasattr(robject, "uns"):
            for k, v in robject.uns.items():
                robject.uns[k] = rpy2py(v)
    return robject


def _pysce2rpy(pyobject):
    if utils._try_import("anndata2ri"):
        pyobject = anndata2ri.py2rpy(pyobject)
    return pyobject


def _is_r_object(obj):
    return (
        "rpy2.robjects" in str(type(obj))
        or "rpy2.rinterface" in str(type(obj))
        or obj is rpy2.rinterface.NULL
    )


def _is_builtin(obj):
    return isinstance(obj, (int, str, float))


@utils._with_pkg(pkg="rpy2", min_version="3.0")
def rpy2py(robject):
    """Convert an rpy2 object to Python.

    Attempts the following, in order: data.frame -> pd.DataFrame, named list -> dict,
    unnamed list -> list, SingleCellExperiment -> anndata.AnnData, vector -> np.ndarray,
    rpy2 generic converter, NULL -> None.

    Parameters
    ----------
    robject : rpy2 object
        Object to be converted

    Returns
    -------
    pyobject : python object
        Converted object
    """
    for converter in [
        _rpynull2py,
        _rpysce2py,
        rpy2.robjects.pandas2ri.rpy2py,
        _rpylist2py,
        rpy2.robjects.numpy2ri.rpy2py,
        rpy2.robjects.conversion.rpy2py,
    ]:
        if _is_r_object(robject):
            try:
                robject = converter(robject)
            except NotImplementedError:
                pass
        else:
            break
    if _is_r_object(robject):
        warnings.warn(
            "Object not converted: {} (type {})".format(
                robject, type(robject).__name__
            ),
            RuntimeWarning,
        )
    return robject


@utils._with_pkg(pkg="rpy2", min_version="3.0")
def py2rpy(pyobject):
    """Convert an Python object to rpy2.

    Attempts the following, in order: data.frame -> pd.DataFrame, named list -> dict,
    unnamed list -> list, SingleCellExperiment -> anndata.AnnData, vector -> np.ndarray,
    rpy2 generic converter, NULL -> None.

    Parameters
    ----------
    pyobject : python object
        Converted object

    Returns
    -------
    robject : rpy2 object
        Object to be converted
    """
    for converter in [
        _pynull2rpy,
        _pysce2rpy,
        rpy2.robjects.pandas2ri.py2rpy,
        rpy2.robjects.numpy2ri.py2rpy,
        rpy2.robjects.conversion.py2rpy,
    ]:
        if not _is_r_object(pyobject):
            try:
                pyobject = converter(pyobject)
            except NotImplementedError:
                pass
        else:
            break
    if not _is_r_object(pyobject) and not _is_builtin(pyobject):
        warnings.warn(
            "Object not converted: {} (type {})".format(
                pyobject, type(pyobject).__name__
            ),
            RuntimeWarning,
        )
    return pyobject
