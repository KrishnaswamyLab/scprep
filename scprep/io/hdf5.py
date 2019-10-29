# author: Scott Gigante <scott.gigante@yale.edu>
# (C) 2018 Krishnaswamy Lab GPLv2
from decorator import decorator

from .._lazyload import tables
from .._lazyload import h5py

try:
    ModuleNotFoundError
except NameError:
    # python 3.5
    ModuleNotFoundError = ImportError


@decorator
def with_HDF5(fun, *args, **kwargs):
    try:
        tables
    except NameError:
        try:
            h5py
        except NameError:
            raise ModuleNotFoundError(
                "Found neither tables nor h5py. "
                "Please install one of them with e.g. "
                "`pip install --user tables` or `pip install --user h5py`"
            )
    return fun(*args, **kwargs)


@with_HDF5
def open_file(filename, mode="r", backend=None):
    """Open an HDF5 file with either tables or h5py

    Gives a simple, unified interface for both tables and h5py

    Parameters
    ----------
    filename : str
        Name of the HDF5 file
    mode : str, optional (default: 'r')
        Read/write mode. Choose from ['r', 'w', 'a' 'r+']
    backend : str, optional (default: None)
        HDF5 backend to use. Choose from ['h5py', 'tables']. If not given,
        scprep will detect which backend is available, using tables if
        both are installed.

    Returns
    -------
    f : tables.File or h5py.File
        Open HDF5 file handle.
    """
    if backend is None:
        try:
            tables
            backend = "tables"
        except NameError:
            backend = "h5py"
    if backend == "tables":
        return tables.open_file(filename, mode)
    elif backend == "h5py":
        return h5py.File(filename, mode)
    else:
        raise ValueError(
            "Expected backend in ['tables', 'h5py']. Got {}".format(backend)
        )


def _is_tables(obj, allow_file=True, allow_group=True, allow_dataset=True):
    try:
        types = []
        if allow_file:
            types.append(tables.File)
        if allow_group:
            types.append(tables.Group)
        if allow_dataset:
            types.append(tables.CArray)
            types.append(tables.Array)
    except NameError:
        return False
    return isinstance(obj, tuple(types))


def _is_h5py(obj, allow_file=True, allow_group=True, allow_dataset=True):
    try:
        types = []
        if allow_file:
            types.append(h5py.File)
        if allow_group:
            types.append(h5py.Group)
        if allow_dataset:
            types.append(h5py.Dataset)
    except NameError:
        return False
    return isinstance(obj, tuple(types))


@with_HDF5
def list_nodes(f):
    """List all first-level nodes in a HDF5 file.

    Parameters
    ----------
    f : tables.File or h5py.File
        Open HDF5 file handle.

    Returns
    -------
    nodes : list
        List of names of first-level nodes below f
    """
    if _is_h5py(f, allow_dataset=False):
        return [node for node in f.keys()]
    elif _is_tables(f, allow_dataset=False):
        return [node._v_name for node in f.list_nodes(f.root)]
    else:
        raise TypeError(
            "Expected h5py.File, tables.File, h5py.Group or tables.Group. Got {}".format(
                type(f)
            )
        )


@with_HDF5
def get_node(f, node):
    """Get a subnode from a HDF5 file or group.

    Parameters
    ----------
    f : tables.File, h5py.File, tables.Group or h5py.Group
        Open HDF5 file handle or node
    node : str
        Name of subnode to retrieve

    Returns
    -------
    g : tables.Group, h5py.Group, tables.CArray or hdf5.Dataset
        Requested HDF5 node.
    """
    if _is_h5py(f, allow_dataset=False):
        return f[node]
    elif _is_tables(f, allow_dataset=False):
        if isinstance(f, tables.File):
            return f.get_node(f.root, node)
        else:
            return f[node]
    else:
        raise TypeError(
            "Expected h5py.File, tables.File, h5py.Group or tables.Group. Got {}".format(
                type(f)
            )
        )


@with_HDF5
def get_values(dataset):
    """Read values from a HDF5 dataset.

    Parameters
    ----------
    dataset : tables.CArray or h5py.Dataset

    Returns
    -------
    data : np.ndarray
        Data read from HDF5 dataset
    """
    if _is_h5py(dataset, allow_file=False, allow_group=False):
        return dataset[()]
    elif _is_tables(dataset, allow_file=False, allow_group=False):
        return dataset.read()
    else:
        raise TypeError(
            "Expected h5py.Dataset or tables.CArray. Got {}".format(type(dataset))
        )
