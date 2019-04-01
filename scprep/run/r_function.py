import numpy as np

from .. import utils
from .._lazyload import rpy2


loud_console_warning = rpy2.rinterface_lib.callbacks.consolewrite_warnerror


def quiet_console_warning(s: str) -> None:
    rpy2.rinterface_lib.callbacks.logger.warning(
        rpy2.rinterface_lib.callbacks._WRITECONSOLE_EXCEPTION_LOG, s.strip())


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

    @utils._with_pkg(pkg="rpy2", min_version="2-3")
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

    @utils._with_pkg(pkg="rpy2", min_version="2-3")
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
                robject = rpy2.robjects.numpy2ri.rpy2py(robject)
                if self.is_r_object(robject):
                    # try regular conversion
                    robject = rpy2.robjects.conversion.rpy2py(robject)
                if robject is rpy2.rinterface.NULL:
                    robject = None
        return robject

    def __call__(self, *args, **kwargs):
        # monkey patch warnings
        rpy2.rinterface_lib.callbacks.consolewrite_warnerror = quiet_console_warning
        robject = self.function(*args, **kwargs)
        robject = self.convert(robject)
        rpy2.rinterface_lib.callbacks.consolewrite_warnerror = loud_console_warning
        return robject
