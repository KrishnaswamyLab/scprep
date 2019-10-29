import subprocess
import os
import scprep
import sys
from tools import data


def test_lazyload():
    proc = subprocess.Popen(
        ["nose2", "--quiet", "_test_lazyload"],
        cwd=os.path.join(data._get_root_dir(), "test"),
        stderr=subprocess.PIPE,
    )
    return_code = proc.wait()
    try:
        assert return_code == 0
    except AssertionError:
        lines = proc.stderr.read().decode().split("\n")
        lines = lines[4:-6]
        raise AssertionError("\n".join(lines))
    finally:
        proc.stderr.close()


def test_builtins():
    for module in scprep._lazyload._importspec.keys():
        try:
            del sys.modules[module]
        except KeyError:
            pass
        assert (
            getattr(scprep._lazyload, module).__class__ is scprep._lazyload.AliasModule
        )
        try:
            getattr(scprep._lazyload, module).__version__
        except AttributeError:
            pass
        assert getattr(scprep._lazyload, module).__class__ is type(scprep)
