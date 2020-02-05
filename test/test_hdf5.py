from tools import data, utils
import os
import scprep


@scprep.io.hdf5.with_HDF5
def hdf5_available():
    return True


def test_failed_import_tables():
    h5_file = os.path.join(data.data_dir, "test_10X.h5")
    tables = scprep.io.hdf5.tables
    del scprep.io.hdf5.tables
    assert hdf5_available() is True
    with tables.File(h5_file, "r") as f:
        assert scprep.io.hdf5._is_tables(f) is False
    with scprep.io.hdf5.open_file(h5_file) as f:
        assert scprep.io.hdf5._is_h5py(f)
    scprep.io.hdf5.tables = tables


def test_failed_import_h5py():
    h5_file = os.path.join(data.data_dir, "test_10X.h5")
    h5py = scprep.io.hdf5.h5py
    del scprep.io.hdf5.h5py
    assert hdf5_available() is True
    with h5py.File(h5_file, "r") as f:
        assert scprep.io.hdf5._is_h5py(f) is False
    scprep.io.hdf5.h5py = h5py


def test_failed_import_both():
    tables = scprep.io.hdf5.tables
    del scprep.io.hdf5.tables
    h5py = scprep.io.hdf5.h5py
    del scprep.io.hdf5.h5py
    utils.assert_raises_message(
        ImportError,
        "Found neither tables nor h5py. "
        "Please install one of them with e.g. "
        "`pip install --user tables` or "
        "`pip install --user h5py`",
        hdf5_available,
    )
    scprep.io.hdf5.tables = tables
    scprep.io.hdf5.h5py = h5py


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
