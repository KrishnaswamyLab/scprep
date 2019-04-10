import scprep
import sys


def test_lazyload():
    loaded_modules = set([m.split('.')[0] for m in sys.modules.keys()])
    exceptions = ['matplotlib', 'mpl_toolkits']
    for module in scprep._lazyload._importspec.keys():
        assert module not in loaded_modules.difference(exceptions)
