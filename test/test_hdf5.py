from tools import data, utils
import os
import mock
import scprep
import sys
import h5py
import tables


@scprep.io.hdf5.with_HDF5
def hdf5_available():
    return True


def test_failed_import_tables():
    h5_file = os.path.join(data.data_dir, "test_10X.h5")
    with tables.File(h5_file, "r") as f_tables:
        with mock.patch.dict(sys.modules, {"tables": None}):
            assert hdf5_available() is True
            assert scprep.io.hdf5._is_tables(f_tables) is False
            with scprep.io.hdf5.open_file(h5_file) as f_h5py:
                assert scprep.io.hdf5._is_h5py(f_h5py)


def test_failed_import_h5py():
    h5_file = os.path.join(data.data_dir, "test_10X.h5")
    with h5py.File(h5_file, "r") as f_h5py:
        with mock.patch.dict(sys.modules, {"h5py": None}):
            assert hdf5_available() is True
            assert scprep.io.hdf5._is_h5py(f_h5py) is False
            with scprep.io.hdf5.open_file(h5_file) as f_tables:
                assert scprep.io.hdf5._is_tables(f_tables)


def test_failed_import_both():
    with mock.patch.dict(sys.modules, {"tables": None, "h5py": None}):
        utils.assert_raises_message(
            ImportError,
            "Found neither tables nor h5py. "
            "Please install one of them with e.g. "
            "`pip install --user tables` or "
            "`pip install --user h5py`",
            hdf5_available,
        )


def test_list_nodes_invalid():
    utils.assert_raises_message(
        TypeError,
        "Expected h5py.File, tables.File, h5py.Group or "
        "tables.Group. Got <class 'str'>",
        scprep.io.hdf5.list_nodes,
        "invalid",
    )


def test_get_node_invalid():
    utils.assert_raises_message(
        TypeError,
        "Expected h5py.File, tables.File, h5py.Group or "
        "tables.Group. Got <class 'str'>",
        scprep.io.hdf5.get_node,
        "invalid",
        "node",
    )


def test_get_values_invalid():
    utils.assert_raises_message(
        TypeError,
        "Expected h5py.Dataset or tables.CArray. " "Got <class 'str'>",
        scprep.io.hdf5.get_values,
        "invalid",
    )
