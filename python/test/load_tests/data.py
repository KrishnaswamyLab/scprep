import scprep
import os

data_dir = os.getcwd().split(os.path.sep)
while data_dir[-1] in ["load_tests", "test", "python"]:
    data_dir = data_dir[:-1]

data_dir = data_dir + ["data", "test_data"]
end = data_dir[1:] if len(data_dir) > 2 else [data_dir[1]]
data_dir = [data_dir[0]] + [os.path.sep] + end
data_dir = os.path.join(*data_dir)


def load_10X(**kwargs):
    return scprep.io.load_10X(
        os.path.join(data_dir, "test_10X"), **kwargs)
