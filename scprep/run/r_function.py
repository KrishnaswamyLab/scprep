import numpy as np
import warnings

from .. import utils

from .._lazyload import rpy2

_formatwarning = warnings.formatwarning


def _quiet_rwarning(message, category, *args, **kwargs):
    if category == rpy2.rinterface.RRuntimeWarning:
        return 'RRuntimeWarning: ' + str(message)
    else:
        return _formatwarning(message, category, *args, **kwargs)


class RFunction(object):
    """Run an R function from Python
    """

    def __init__(self, name, args, setup, body, quiet_setup=True):
        self.name = name
        self.args = args
        self.setup = setup
        self.body = body
        if quiet_setup:
            self.setup = """
                suppressPackageStartupMessages(suppressMessages(
                    suppressWarnings({{
                        {setup}
                    }})))""".format(setup=self.setup)

    @utils._with_pkg(pkg="rpy2")
    def _build(self):
        function_text = """
        {setup}
        {name} <- function({args}) {{
          {body}
        }}
        """.format(setup=self.setup, name=self.name,
                   args=self.args, body=self.body)
        fun = getattr(rpy2.robjects.packages.STAP(
            function_text, self.name), self.name)
        rpy2.robjects.numpy2ri.activate()
        return fun

    @property
    def function(self):
        try:
            return self._function
        except AttributeError:
            self._function = self._build()
            return self._function

    def is_r_object(self, obj):
        return "rpy2.robjects" in str(type(obj)) or obj is rpy2.rinterface.NULL

    @utils._with_pkg(pkg="rpy2")
    def convert(self, robject):
        if self.is_r_object(robject):
            if isinstance(robject, rpy2.robjects.vectors.ListVector):
                names = self.convert(robject.names)
                if names is rpy2.rinterface.NULL or \
                        len(names) > len(np.unique(names)):
                    # list
                    robject = [self.convert(obj) for obj in robject]
                else:
                    # dictionary
                    robject = {name: self.convert(
                        obj) for name, obj in zip(robject.names, robject)}
            else:
                # try numpy first
                robject = rpy2.robjects.numpy2ri.ri2py(robject)
                if self.is_r_object(robject):
                    # try regular conversion
                    robject = rpy2.robjects.conversion.ri2py(robject)
                if robject is rpy2.rinterface.NULL:
                    robject = None
        return robject

    def __call__(self, *args, **kwargs):
        # monkey patch warnings
        warnings.formatwarning = _quiet_rwarning
        robject = self.function(*args, **kwargs)
        robject = self.convert(robject)
        warnings.formatwarning = _formatwarning
        return robject
