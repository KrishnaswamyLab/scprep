from .. import utils
from .._lazyload import rpy2
from . import conversion

import functools


def _console_warning(s, log_fn):
    s = s.strip()
    if s == "=":
        return
    else:
        return log_fn(rpy2.rinterface_lib.callbacks._WRITECONSOLE_EXCEPTION_LOG, s)


class _ConsoleWarning(object):
    def __init__(self, verbose=1):
        if verbose is True:
            verbose = 1
        elif verbose is False:
            verbose = 0
        self.verbose = verbose

    @staticmethod
    def warning(s: str) -> None:
        _console_warning(s, rpy2.rinterface_lib.callbacks.logger.warning)

    @staticmethod
    def debug(s: str) -> None:
        _console_warning(s, rpy2.rinterface_lib.callbacks.logger.debug)

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


@functools.lru_cache(None)
def setup_rlang():
    rpy2.robjects.r(
        """
    if (!require('rlang')) install.packages('rlang')
    options(error = rlang::entrace)
    """
    )


class RFunction(object):
    """Run an R function from Python.

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
        setup_rlang()
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
            try:
                robject = self.function(*args, **kwargs)
            except rpy2.rinterface_lib.embedded.RRuntimeError as e:
                # Attempt to capture the traceback from R.
                # Credit: https://stackoverflow.com/a/40002973
                try:
                    r_traceback = rpy2.robjects.r(
                        "format(rlang::last_trace(), simplify='none', fields=TRUE)"
                    )[0]
                except Exception as traceback_exc:
                    r_traceback = (
                        "\n(an error occurred while getting traceback "
                        f"from R){traceback_exc}"
                    )
                e.args = (f"{e.args[0]}\n{r_traceback}",)
                raise

            robject = conversion.rpy2py(robject)
            if rpy_cleanup:
                rpy2.robjects.r("rm(list=ls())")
        self.verbose = default_verbose
        return robject


_install_bioconductor = RFunction(
    args="package = character(), site_repository = character(), update = FALSE, "
    'type="binary", version = BiocManager::version()',
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
                                   update=update, ask=ask, version=version, type=type)
            }
          }
        }
    """,
)


def install_bioconductor(
    package=None,
    site_repository=None,
    update=False,
    type="binary",
    version=None,
    verbose=True,
):
    """Install a Bioconductor package.

    Parameters
    ----------
    site_repository : string, optional (default: None)
        additional repository in which to look for packages to install.
        This repository will be prepended to the default repositories
    update : boolean, optional (default: False)
        When False, don't attempt to update old packages.
        When True, update old packages automatically.
    type : {"binary", "source", "both"}, optional (default: "binary")
        Which package version to install if a newer version is available as source.
        "both" tries source first and uses binary as a fallback.
    version : string, optional (default: None)
        Bioconductor version to install, e.g., version = "3.8".
        The special symbol version = "devel" installs the current 'development' version.
        If None, installs from the current version.
    verbose : boolean, optional (default: True)
        Install script verbosity.
    """
    kwargs = {"update": update, "rpy_verbose": verbose, "type": type}
    if package is not None:
        kwargs["package"] = package
    if site_repository is not None:
        kwargs["site_repository"] = site_repository
    if version is not None:
        kwargs["version"] = version
    _install_bioconductor(**kwargs)


_install_github = RFunction(
    args="""repo=character(), lib=.libPaths()[1], dependencies=NA,
            update=FALSE, type="binary",
            build_vignettes=FALSE, force=FALSE, verbose=TRUE""",
    body="""
        quiet <- !verbose

        if (!require('remotes', quietly=TRUE)) install.packages('remotes', lib=lib)
        library(remotes)
        install_github(repo=repo,
                         lib=lib, dependencies=dependencies,
                         upgrade=update,
                         build_vignettes=build_vignettes,
                         force=force, quiet=quiet)

        # prepend path to libPaths if new library
        if (lib != .libPaths()[1]) .libPaths(c(lib, .libPaths()))

        if (verbose) cat('.libPaths():', .libPaths())
    """,
)


def install_github(
    repo,
    lib=None,
    dependencies=None,
    update=False,
    type="binary",
    build_vignettes=False,
    force=False,
    verbose=True,
):
    """Install a Github repository.

    Parameters
    ----------
    repo: string
        Github repository name to install.
    lib: string
        Directory to install the package.
        If missing, defaults to the first element of .libPaths().
    dependencies: boolean, optional (default: None/NA)
        When True, installs all packages specified under "Depends", "Imports",
        "LinkingTo" and "Suggests".
        When False, installs no dependencies.
        When None/NA, installs all packages specified under "Depends", "Imports"
        and "LinkingTo".
    update: string or boolean, optional (default: False)
        One of "default", "ask", "always", or "never". "default"
        Respects R_REMOTES_UPGRADE variable if set, falls back to "ask" if unset.
        "ask" prompts the user for which out of date packages to upgrade.
        For non-interactive sessions "ask" is equivalent to "always".
        TRUE and FALSE also accepted, correspond to "always" and "never" respectively.
    type : {"binary", "source", "both"}, optional (default: "binary")
        Which package version to install if a newer version is available as source.
        "both" tries source first and uses binary as a fallback.
    build_vignettes: boolean, optional (default: False)
        Builds Github vignettes.
    force: boolean, optional (default: False)
        Forces installation even if remote state has not changed since previous install.
    verbose: boolean, optional (default: True)
        Install script verbosity.
    """
    kwargs = {"type": type}
    if lib is not None:
        kwargs["lib"] = lib
    if dependencies is not None:
        kwargs["dependencies"] = dependencies

    _install_github(
        repo=repo,
        update=update,
        build_vignettes=build_vignettes,
        force=force,
        verbose=verbose,
        **kwargs,
    )
