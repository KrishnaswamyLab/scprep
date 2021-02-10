import subprocess
import mock
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
    for module_name in scprep._lazyload._importspec.keys():
        if module_name == "anndata2ri" and sys.version_info[:2] < (3, 6):
            continue
        with mock.patch.dict(sys.modules):
            if module_name in sys.modules:
                del sys.modules[module_name]
            module = getattr(scprep._lazyload, module_name)
            assert module.__class__ is scprep._lazyload.AliasModule, (
                module_name,
                module,
            )
            module.__package__
            assert module.__class__ is type(scprep), (module_name, module)
