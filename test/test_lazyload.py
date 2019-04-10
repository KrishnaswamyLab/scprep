import subprocess
import os
from tools import data


def test_lazyload():
    return_code = subprocess.call(
        ['nose2', '--quiet', '_test_lazyload'],
        cwd=os.path.join(data._get_root_dir(), "test"))
    assert return_code == 0
