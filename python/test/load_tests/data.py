import scprep
import os

data_dir = os.getcwd().split(os.path.sep)
while data_dir[-1] in ["load_tests", "test", "python"]:
    data_dir = data_dir[:-1]
data_dir = data_dir + ["data", "test_data"]


def load_10X(**kwargs):
    return scprep.io.load_10X(
        os.path.join(data_dir, "test_10X"), **kwargs)
