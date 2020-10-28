import numpy as np

from .. import utils
from .._lazyload import rpy2


def activate():
    rpy2.robjects.numpy2ri.activate()
    rpy2.robjects.pandas2ri.activate()
    try:
        import anndata2ri

        anndata2ri.activate()
    except ModuleNotFoundError:
        pass


def rpylist2py(robject):
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


def rpynull2py(robject):
    if robject is rpy2.rinterface.NULL:
        return None
    else:
        raise NotImplementedError


def rpysce2py(robject):
    try:
        import anndata2ri

        return anndata2ri.rpy2py(robject)
    except ModuleNotFoundError:
        raise NotImplementedError


def is_r_object(obj):
    return "rpy2.robjects" in str(type(obj)) or obj is rpy2.rinterface.NULL


@utils._with_pkg(pkg="rpy2", min_version="3.0")
def rpy2py(robject):
    for converter in [
        rpy2.robjects.pandas2ri.rpy2py,
        rpylist2py,
        rpysce2py,
        rpy2.robjects.numpy2ri.rpy2py,
        rpy2.robjects.conversion.rpy2py,
        rpynull2py,
    ]:
        if is_r_object(robject):
            try:
                robject = converter(robject)
            except NotImplementedError:
                pass
        else:
            break
    return robject
