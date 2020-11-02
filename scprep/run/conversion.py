import numpy as np

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
        for k, v in robject.uns.items():
            robject.uns[k] = rpy2py(v)
    return robject


def _pysce2rpy(pyobject):
    if utils._try_import("anndata2ri"):
        pyobject = anndata2ri.py2rpy(pyobject)
    return pyobject


def _is_r_object(obj):
    return "rpy2.robjects" in str(type(obj)) or obj is rpy2.rinterface.NULL


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
        rpy2.robjects.pandas2ri.rpy2py,
        _rpylist2py,
        _rpysce2py,
        rpy2.robjects.numpy2ri.rpy2py,
        rpy2.robjects.conversion.rpy2py,
        _rpynull2py,
    ]:
        if _is_r_object(robject):
            try:
                robject = converter(robject)
            except NotImplementedError:
                pass
        else:
            break
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
        rpy2.robjects.pandas2ri.py2rpy,
        _pysce2rpy,
        rpy2.robjects.numpy2ri.py2rpy,
        rpy2.robjects.conversion.py2rpy,
        _pynull2rpy,
    ]:
        if not _is_r_object(pyobject):
            try:
                pyobject = converter(pyobject)
            except NotImplementedError:
                pass
        else:
            break
    return pyobject
