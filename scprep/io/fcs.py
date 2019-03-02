# author: Scott Gigante <scott.gigante@yale.edu>
# (C) 2018 Krishnaswamy Lab GPLv2

import pandas as pd
import numpy as np
from decorator import decorator
import struct
from io import BytesIO

from .utils import _matrix_to_data_frame

try:
    import fcsparser
except ImportError:
    pass


def _fcsextract(filename):
    # FCS parser:
    # https://github.com/jbkinney/16_titeseq/blob/master/fcsextract.py

    fcs_file_name = filename

    fcs = open(fcs_file_name, 'rb')
    header = fcs.read(58)
    version = header[0:6].strip()
    text_start = int(header[10:18].strip())
    text_end = int(header[18:26].strip())
    data_start = int(header[26:34].strip())
    data_end = int(header[34:42].strip())
    analysis_start = int(header[42:50].strip())
    analysis_end = int(header[50:58].strip())

    # print "Parsing TEXT segment"
    # read TEXT portion
    fcs.seek(text_start)
    delimeter = fcs.read(1)
    # First byte of the text portion defines the delimeter
    # print "delimeter:",delimeter
    text = fcs.read(text_end - text_start + 1)

    # Variables in TEXT poriton are stored "key/value/key/value/key/value"
    keyvalarray = text.split(delimeter)
    fcs_vars = {}
    fcs_var_list = []
    # Iterate over every 2 consecutive elements of the array
    for k, v in zip(keyvalarray[::2], keyvalarray[1::2]):
        fcs_vars[k.decode()] = v.decode()
        # Keep a list around so we can print them in order
        fcs_var_list.append((k, v))

    # from pprint import pprint; pprint(fcs_var_list)
    if data_start == 0 and data_end == 0:
        data_start = int(fcs_vars['$DATASTART'])
        data_end = int(fcs_vars['$DATAEND'])

    num_dims = int(fcs_vars['$PAR'])
    # print "Number of dimensions:",num_dims

    num_events = int(fcs_vars['$TOT'])
    # print "Number of events:",num_events

    # Read DATA portion
    fcs.seek(data_start)
    # print "# of Data bytes",data_end-data_start+1
    data = fcs.read(data_end - data_start + 1)

    # Determine data format
    datatype = fcs_vars['$DATATYPE']
    if datatype == 'F':
        datatype = 'f'  # set proper data mode for struct module
        # print "Data stored as single-precision (32-bit) floating point
        # numbers"
    elif datatype == 'D':
        datatype = 'd'  # set proper data mode for struct module
        # print "Data stored as double-precision (64-bit) floating point
        # numbers"
    else:
        assert False, "Error: Unrecognized $DATATYPE '%s'" % datatype

    # Determine endianess
    endian = fcs_vars['$BYTEORD']
    if endian == "4,3,2,1":
        endian = ">"  # set proper data mode for struct module
        # print "Big endian data format"
    elif endian == "1,2,3,4":
        # print "Little endian data format"
        endian = "<"  # set proper data mode for struct module
    else:
        assert False, "Error: This script can only read data encoded with $BYTEORD = 1,2,3,4 or 4,3,2,1"

    # Put data in StringIO so we can read bytes like a file
    data = BytesIO(data)

    # print "Parsing DATA segment"
    # Create format string based on endianeness and the specified data type
    format = endian + str(num_dims) + datatype
    datasize = struct.calcsize(format)
    # print "Data format:",format
    # print "Data size:",datasize
    events = []
    # Read and unpack all the events from the data
    for e in range(num_events):
        event = struct.unpack(format, data.read(datasize))
        events.append(event)

    fcs.close()

    events = np.array(events)
    return fcs_vars, events


def _fcs_to_dataframe(f):
    fcs_vars, events = _fcsextract(f)
    gene_names = {}
    for i in range(1, events.shape[1] + 1):
        key = '$P{}S'.format(i)
        if key in fcs_vars:
            if '_' in fcs_vars[key]:
                gene_names[i] = fcs_vars[key].split('_')[1]
            else:
                gene_names[i] = fcs_vars[key]
        else:
            key = '$P{}N'.format(i)
            gene_names[i] = fcs_vars[key]

    gene_names = np.array([*gene_names.values()])
    return pd.DataFrame(events, columns=gene_names,
                        index=np.arange(events.shape[0]))


@decorator
def _with_fcsparser(fun, *args, **kwargs):
    try:
        fcsparser
    except NameError:
        raise ImportError(
            "fcsparser not found. "
            "Please install it with e.g. `pip install --user fcsparser`")
    return fun(*args, **kwargs)


@_with_fcsparser
def load_fcs(filename, gene_names=True, cell_names=True,
             sparse=None,
             metadata_channels=['Time', 'Event_length', 'DNA1', 'DNA2',
                                'Cisplatin', 'beadDist', 'bead1'],
             reformat_meta=True, override=False,
             **kwargs):
    """Load a fcs file

    Parameters
    ----------
    filename : str
        The name of the fcs file to be loaded
    gene_names : `bool`, `str`, array-like, or `None` (default: True)
        If `True`, we assume gene names are contained in the file. Otherwise
        expects a filename or an array containing a list of gene symbols or ids
    cell_names : `bool`, `str`, array-like, or `None` (default: True)
        If `True`, we assume cell names are contained in the file. Otherwise
        expects a filename or an array containing a list of cell barcodes.
    sparse : bool, optional (default: None)
        If True, loads the data as a pd.SparseDataFrame. This uses less memory
        but more CPU.
    metadata_channels : list-like, optional, shape=[n_meta] (default: ['Time', 'Event_length', 'DNA1', 'DNA2', 'Cisplatin', 'beadDist', 'bead1'])
        Channels to be excluded from the data
    reformat_meta : bool, optional (default: True)
        If true, the meta data is reformatted with the channel information
        organized into a DataFrame and moved into the '_channels_' key
    **kwargs : optional arguments for `fcsparser.parse`.

    Returns
    -------
    channel_metadata : dict
        FCS metadata
    cell_metadata : array-like, shape=[n_samples, n_meta]
        Values from metadata channels
    data : array-like, shape=[n_samples, n_features]
        If either gene or cell names are given, data will be a pd.DataFrame or
        pd.SparseDataFrame. If no names are given, data will be a np.ndarray
        or scipy.sparse.spmatrix
    """
    if cell_names is True:
        cell_names = None
    if gene_names is True:
        gene_names = None
    # Parse the fcs file
    if override:
        data = _fcs_to_dataframe(filename)
        channel_metadata = None
    else:
        channel_metadata, data = fcsparser.parse(
            filename, reformat_meta=reformat_meta, **kwargs)
    metadata_channels = data.columns.intersection(metadata_channels)
    data_channels = data.columns.difference(metadata_channels)
    cell_metadata = data[metadata_channels]
    data = data[data_channels]
    data = _matrix_to_data_frame(data, gene_names=gene_names,
                                 cell_names=cell_names, sparse=sparse)
    return channel_metadata, cell_metadata, data
