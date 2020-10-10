# author: Scott Gigante <scott.gigante@yale.edu>
# (C) 2018 Krishnaswamy Lab GPLv2

import pandas as pd
import numpy as np
import struct
from io import BytesIO
import string
import warnings

from .utils import _matrix_to_data_frame
from .._lazyload import fcsparser
from .. import utils


def _channel_names_from_meta(meta, channel_numbers, naming="N"):
    try:
        return tuple([meta["$P{0}{1}".format(i, naming)] for i in channel_numbers])
    except KeyError:
        return []


def _get_channel_names(meta, channel_numbers, channel_naming="$PnS"):
    """Get list of channel names. Raises a warning if the names are not unique.

    Credit: https://github.com/eyurtsev/fcsparser/blob/master/fcsparser/api.py
    """
    names_n = _channel_names_from_meta(meta, channel_numbers, "N")
    names_s = _channel_names_from_meta(meta, channel_numbers, "S")

    # Figure out which channel names to use
    if channel_naming == "$PnS":
        channel_names, channel_names_alternate = names_s, names_n
    elif channel_naming == "$PnN":
        channel_names, channel_names_alternate = names_n, names_s
    else:
        raise ValueError(
            "Expected channel_naming in ['$PnS', '$PnN']. "
            "Got '{}'".format(channel_naming)
        )

    if len(channel_names) == 0:
        channel_names = channel_names_alternate

    if len(set(channel_names)) != len(channel_names):
        warnings.warn(
            "The default channel names (defined by the {} "
            "parameter in the FCS file) were not unique. To avoid "
            "problems in downstream analysis, the channel names "
            "have been switched to the alternate channel names "
            "defined in the FCS file. To avoid "
            "seeing this warning message, explicitly instruct "
            "the FCS parser to use the alternate channel names by "
            "specifying the channel_naming parameter.".format(channel_naming),
            RuntimeWarning,
        )
        channel_names = channel_names_alternate

    return channel_names


def _reformat_meta(meta, channel_numbers):
    """Collect the meta data information in a more user friendly format.
    Function looks through the meta data, collecting the channel related information into a
    dataframe and moving it into the _channels_ key.

    Credit: https://github.com/eyurtsev/fcsparser/blob/master/fcsparser/api.py
    """
    channel_properties = []

    for key, value in meta.items():
        if key[:3] == "$P1":
            if key[3] not in string.digits:
                channel_properties.append(key[3:])

    # Capture all the channel information in a list of lists -- used to create
    # a data frame
    channel_matrix = [
        [meta.get("$P{0}{1}".format(ch, p)) for p in channel_properties]
        for ch in channel_numbers
    ]

    # Remove this information from the dictionary
    for ch in channel_numbers:
        for p in channel_properties:
            key = "$P{0}{1}".format(ch, p)
            if key in meta:
                meta.pop(key)

    num_channels = meta["$PAR"]
    column_names = ["$Pn{0}".format(p) for p in channel_properties]

    df = pd.DataFrame(
        channel_matrix, columns=column_names, index=(1 + np.arange(num_channels))
    )

    if "$PnE" in column_names:
        df["$PnE"] = df["$PnE"].apply(lambda x: x.split(","))
    if "$PnB" in column_names:
        df["$PnB"] = df["$PnB"].apply(lambda x: int(x))

    df.index.name = "Channel Number"
    return df


def _read_fcs_header(filename):
    meta = dict()
    with open(filename, "rb") as handle:
        # Parse HEADER
        header = handle.read(58)
        meta["__header__"] = dict()
        meta["__header__"]["FCS format"] = header[0:6].strip()
        meta["__header__"]["text start"] = int(header[10:18].strip())
        meta["__header__"]["text end"] = int(header[18:26].strip())
        meta["__header__"]["data start"] = int(header[26:34].strip())
        meta["__header__"]["data end"] = int(header[34:42].strip())
        meta["__header__"]["analysis start"] = int(header[42:50].strip())
        meta["__header__"]["analysis end"] = int(header[50:58].strip())

        # Parsing TEXT segment
        # read TEXT portion
        handle.seek(meta["__header__"]["text start"])
        # First byte of the text portion defines the delimeter
        delimeter = handle.read(1)
        text = handle.read(
            meta["__header__"]["text end"] - meta["__header__"]["text start"] + 1
        )

        # Variables in TEXT poriton are stored "key/value/key/value/key/value"
        keyvalarray = text.split(delimeter)
        # Iterate over every 2 consecutive elements of the array
        for k, v in zip(keyvalarray[::2], keyvalarray[1::2]):
            meta[k.decode()] = v.decode()
    return meta


def _parse_fcs_header(meta):
    if meta["__header__"]["data start"] == 0 and meta["__header__"]["data end"] == 0:
        meta["$DATASTART"] = int(meta["$DATASTART"])
        meta["$DATAEND"] = int(meta["$DATAEND"])
    else:
        meta["$DATASTART"] = meta["__header__"]["data start"]
        meta["$DATAEND"] = meta["__header__"]["data end"]

    meta["$PAR"] = int(meta["$PAR"])
    meta["$TOT"] = int(meta["$TOT"])

    # Determine data format
    meta["$DATATYPE"] = meta["$DATATYPE"].lower()
    if meta["$DATATYPE"] not in ["f", "d"]:
        raise ValueError(
            "Expected $DATATYPE in ['F', 'D']. " "Got '{}'".format(meta["$DATATYPE"])
        )

    # Determine endianess
    endian = meta["$BYTEORD"]
    if endian == "4,3,2,1":
        # Big endian data format
        meta["$ENDIAN"] = ">"
    elif endian == "1,2,3,4":
        # Little endian data format
        meta["$ENDIAN"] = "<"
    else:
        raise ValueError(
            "Expected $BYTEORD in ['1,2,3,4', '4,3,2,1']. " "Got '{}'".format(endian)
        )
    return meta


def _fcsextract(filename, channel_naming="$PnS", reformat_meta=True):
    """Experimental FCS parser

    Some files fail to load with `fcsparser.parse`. For these, we provide an
    alternative parser. It is not guaranteed to work in all cases.

    Code copied from https://github.com/pontikos/fcstools/blob/master/fcs.extract.py

    Paramseters
    -----------
    channel_naming: '$PnS' | '$PnN'
        Determines which meta data field is used for naming the channels.
        The default should be $PnS (even though it is not guaranteed to be unique)
        $PnN stands for the short name (guaranteed to be unique).
            Will look like 'FL1-H'
        $PnS stands for the actual name (not guaranteed to be unique).
            Will look like 'FSC-H' (Forward scatter)
        The chosen field will be used to population self.channels
        Note: These names are not flipped in the implementation.
        It looks like they were swapped for some reason in the official FCS specification.
    reformat_meta: bool
        If true, the meta data is reformatted with the channel information organized
        into a DataFrame and moved into the '_channels_' key
    """
    meta = _read_fcs_header(filename)
    meta = _parse_fcs_header(meta)
    with open(filename, "rb") as handle:
        # Read DATA portion
        handle.seek(meta["$DATASTART"])

        data = handle.read(meta["$DATAEND"] - meta["$DATASTART"] + 1)
        # Put data in StringIO so we can read bytes like a file
        data = BytesIO(data)

        # Parsing DATA segment
        # Create format string based on endianeness and the specified data type
        fmt = meta["$ENDIAN"] + str(meta["$PAR"]) + meta["$DATATYPE"]
        datasize = struct.calcsize(fmt)
        events = []
        # Read and unpack all the events from the data
        for e in range(meta["$TOT"]):
            event = struct.unpack(fmt, data.read(datasize))
            events.append(event)

    # Number the channels

    pars = meta["$PAR"]
    # Checking whether channel number count starts from 0 or from 1
    if "$P0B" in meta:
        # Channel number count starts from 0
        channel_numbers = range(0, pars)
    else:
        # Channel numbers start from 1
        channel_numbers = range(1, pars + 1)

    channel_names = _get_channel_names(meta, channel_numbers, channel_naming)

    events = pd.DataFrame(
        np.array(events), columns=channel_names, index=np.arange(len(events))
    )

    if reformat_meta:
        try:
            meta["_channels_"] = _reformat_meta(meta, channel_numbers)
        except Exception as e:
            warnings.warn("Metadata reformatting failed: {}".format(str(e)))
        meta["_channel_names_"] = channel_names
    return meta, events


@utils._with_pkg(pkg="fcsparser")
def load_fcs(
    filename,
    gene_names=True,
    cell_names=True,
    sparse=None,
    metadata_channels=[
        "Time",
        "Event_length",
        "DNA1",
        "DNA2",
        "Cisplatin",
        "beadDist",
        "bead1",
    ],
    channel_naming="$PnS",
    reformat_meta=True,
    override=False,
    **kwargs
):
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
        If True, loads the data as a pd.DataFrame[SparseArray]. This uses less memory
        but more CPU.
    metadata_channels : list-like, optional, shape=[n_meta] (default: ['Time', 'Event_length', 'DNA1', 'DNA2', 'Cisplatin', 'beadDist', 'bead1'])
        Channels to be excluded from the data
    channel_naming: '$PnS' | '$PnN'
        Determines which meta data field is used for naming the channels.
        The default should be $PnS (even though it is not guaranteed to be unique)
        $PnN stands for the short name (guaranteed to be unique). Will look like 'FL1-H'
        $PnS stands for the actual name (not guaranteed to be unique). Will look like 'FSC-H' (Forward scatter)
        The chosen field will be used to population self.channels
        Note: These names are not flipped in the implementation.
        It looks like they were swapped for some reason in the official FCS specification.
    reformat_meta : bool, optional (default: True)
        If true, the meta data is reformatted with the channel information
        organized into a DataFrame and moved into the '_channels_' key
    override : bool, optional (default: False)
        If true, uses an experimental override of fcsparser. Should only be
        used in cases where fcsparser fails to load the file, likely due to
        a malformed header. Credit to https://github.com/pontikos/fcstools
    **kwargs : optional arguments for `fcsparser.parse`.

    Returns
    -------
    channel_metadata : dict
        FCS metadata
    cell_metadata : array-like, shape=[n_samples, n_meta]
        Values from metadata channels
    data : array-like, shape=[n_samples, n_features]
        If either gene or cell names are given, data will be a pd.DataFrame or
        pd.DataFrame[SparseArray]. If no names are given, data will be a np.ndarray
        or scipy.sparse.spmatrix
    """
    if cell_names is True:
        cell_names = None
    if gene_names is True:
        gene_names = None
    # Parse the fcs file
    if override:
        channel_metadata, data = _fcsextract(
            filename,
            reformat_meta=reformat_meta,
            channel_naming=channel_naming,
            **kwargs,
        )
    else:
        try:
            channel_metadata, data = fcsparser.api.parse(
                filename, reformat_meta=reformat_meta, **kwargs
            )
        except (fcsparser.api.ParserFeatureNotImplementedError, ValueError):
            raise RuntimeError(
                "fcsparser failed to load {}, likely due to a "
                "malformed header. You can try using "
                "`override=True` to use scprep's built-in "
                "experimental FCS parser.".format(filename)
            )
    metadata_channels = data.columns.intersection(metadata_channels)
    data_channels = data.columns.difference(metadata_channels)
    cell_metadata = data[metadata_channels]
    data = data[data_channels]
    data = _matrix_to_data_frame(
        data, gene_names=gene_names, cell_names=cell_names, sparse=sparse
    )
    return channel_metadata, cell_metadata, data
