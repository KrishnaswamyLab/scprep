import numpy
import pandas
import scipy
import sys


def test_lazyload():
    preloaded_modules = set([m.split(".")[0] for m in sys.modules.keys()])
    assert "scprep" not in preloaded_modules
    import scprep

    postloaded_modules = set([m.split(".")[0] for m in sys.modules.keys()])
    scprep_loaded = postloaded_modules.difference(preloaded_modules)
    for module in scprep._lazyload._importspec.keys():
        if module == "anndata2ri" and sys.version_info[:2] < (3, 6):
            continue
        assert module not in scprep_loaded, module
        if module in postloaded_modules:
            assert getattr(scprep._lazyload, module).__class__ is type(scprep), module
        else:
            assert (
                getattr(scprep._lazyload, module).__class__
                is scprep._lazyload.AliasModule
            ), module
