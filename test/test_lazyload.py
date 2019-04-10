import subprocess
import os


def test_lazyload():
    return_code = subprocess.call(
        ['nose2', '--quiet', '_test_lazyload'],
        cwd=os.path.dirname(os.path.realpath(__file__)))
    assert return_code == 0
