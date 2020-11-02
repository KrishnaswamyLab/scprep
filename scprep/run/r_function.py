from . import conversion
from .. import utils
from .._lazyload import rpy2


class _ConsoleWarning(object):
    def __init__(self, verbose=1):
        if verbose is True:
            verbose = 1
        elif verbose is False:
            verbose = 0
        self.verbose = verbose

    @staticmethod
    def warning(s: str) -> None:
        rpy2.rinterface_lib.callbacks.logger.warning(
            rpy2.rinterface_lib.callbacks._WRITECONSOLE_EXCEPTION_LOG, s.strip()
        )

    @staticmethod
    def debug(s: str) -> None:
        rpy2.rinterface_lib.callbacks.logger.debug(
            rpy2.rinterface_lib.callbacks._WRITECONSOLE_EXCEPTION_LOG, s.strip()
        )

    @staticmethod
    def set(fun):
        if not hasattr(_ConsoleWarning, "builtin_warning"):
            _ConsoleWarning.builtin_warning = (
                rpy2.rinterface_lib.callbacks.consolewrite_warnerror
            )
        rpy2.rinterface_lib.callbacks.consolewrite_warnerror = fun

    @staticmethod
    def set_warning():
        _ConsoleWarning.set(_ConsoleWarning.warning)

    @staticmethod
    def set_debug():
        _ConsoleWarning.set(_ConsoleWarning.debug)

    @staticmethod
    def set_builtin():
        _ConsoleWarning.set(_ConsoleWarning.builtin_warning)

    def __enter__(self):
        self.previous_warning = rpy2.rinterface_lib.callbacks.consolewrite_warnerror
        if self.verbose > 0:
            self.set_warning()
        else:
            self.set_debug()

    def __exit__(self, type, value, traceback):
        self.set(self.previous_warning)


class RFunction(object):
    """Run an R function from Python

    Parameters
    ----------
    args : str, optional (default: "")
        Comma-separated R argument names and optionally default parameters
    setup : str, optional (default: "")
        R code to run prior to function definition (e.g. loading libraries)
    body : str, optional (default: "")
        R code to run in the body of the function
    cleanup : boolean, optional (default: True)
        If true, clear the R workspace after the function is complete.
        If false, this could result in memory leaks.
    verbose : int, optional (default: 1)
        R script verbosity. For verbose==0, all messages are printed.
        For verbose==1, messages from the function body are printed.
        For verbose==2, messages from the function setup and body are printed.
    """

    def __init__(self, args="", setup="", body="", cleanup=True, verbose=1):
        self.name = "fun"
        self.args = args
        self.setup = setup
        self.body = body
        self.cleanup = cleanup
        self.verbose = verbose

    @utils._with_pkg(pkg="rpy2", min_version="3.0")
    def _build(self):
        if self.setup != "":
            with _ConsoleWarning(self.verbose - 1):
                rpy2.robjects.r(self.setup)
        function_text = """
        {name} <- function({args}) {{
          {body}
        }}
        """.format(
            name=self.name, args=self.args, body=self.body
        )
        fun = getattr(rpy2.robjects.packages.STAP(function_text, self.name), self.name)
        return fun

    @property
    def function(self):
        try:
            return self._function
        except AttributeError:
            self._function = self._build()
            return self._function

    @utils._with_pkg(pkg="rpy2", min_version="3.0")
    def __call__(self, *args, rpy_cleanup=None, rpy_verbose=None, **kwargs):
        default_verbose = self.verbose
        if rpy_verbose is None:
            rpy_verbose = self.verbose
        else:
            self.verbose = rpy_verbose
        if rpy_cleanup is None:
            rpy_cleanup = self.cleanup
        args = [conversion.py2rpy(a) for a in args]
        kwargs = {k: conversion.py2rpy(v) for k, v in kwargs.items()}
        with _ConsoleWarning(rpy_verbose):
            robject = self.function(*args, **kwargs)
            robject = conversion.rpy2py(robject)
            if rpy_cleanup:
                rpy2.robjects.r("rm(list=ls())")
        self.verbose = default_verbose
        return robject


_install_bioconductor = RFunction(
    args="package = character(), site_repository = character(), update = FALSE, version = BiocManager::version()",
    body="""
        if (!require('BiocManager')) install.packages("BiocManager")
        ask <- !update
        if (length(package) == 0) {
          BiocManager::install(site_repository=site_repository,
                               update=update, ask=ask, version=version)
        } else {
          for (pkg in package) {
            if (update || !require(pkg, character.only = TRUE)) {
              BiocManager::install(pkg, site_repository=site_repository,
                                   update=update, ask=ask, version=version)
            }
          }
        }
    """,
)


def install_bioconductor(
    package=None, site_repository=None, update=False, version=None, verbose=True
):
    """Install a Bioconductor package

    Parameters
    ----------
    site_repository : string, optional (default: None)
        additional repository in which to look for packages to install.
        This repository will be prepended to the default repositories
    update : boolean, optional (default: False)
        When False, don't attempt to update old packages.
        When True, update old packages automatically.
    version : string, optional (default: None)
        Bioconductor version to install, e.g., version = "3.8".
        The special symbol version = "devel" installs the current 'development' version.
        If None, installs from the current version.
    verbose : boolean, optional (default: True)
        Install script verbosity.
    """
    kwargs = {"update": update, "rpy_verbose": verbose}
    if package is not None:
        kwargs["package"] = package
    if site_repository is not None:
        kwargs["site_repository"] = site_repository
    if version is not None:
        kwargs["version"] = version
    _install_bioconductor(**kwargs)
