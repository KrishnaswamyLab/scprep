import subprocess
import os
from tools import data


def test_lazyload():
    proc = subprocess.Popen(
        ['nose2', '--quiet', '_test_lazyload'],
        cwd=os.path.join(data._get_root_dir(), "test"), stderr=subprocess.PIPE)
    return_code = proc.wait()
    try:
        assert return_code == 0
    except AssertionError:
        lines = proc.stderr.read().decode().split('\n')
        lines = lines[4:-6]
        raise AssertionError("\n".join(lines))
    finally:
        proc.stderr.close()
