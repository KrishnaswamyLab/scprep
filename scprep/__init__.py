# author: Scott Gigante <scott.gigante@yale.edu>
# (C) 2018 Krishnaswamy Lab GPLv2

from .version import __version__
import scprep.io
import scprep.io.hdf5
import scprep.select
import scprep.filter
import scprep.normalize
import scprep.transform
import scprep.measure
import scprep.plot
import scprep.sanitize
import scprep.stats
import scprep.reduce
import scprep.run

import pandas as pd

if int(pd.__version__.split(".")[1]) < 26:

    def fill_value(self):
        # Used in reindex_indexer
        try:
            return self.values.dtype.fill_value
        except AttributeError:
            return self.values.dtype.na_value

    from pandas.core.internals.blocks import ExtensionBlock

    setattr(ExtensionBlock, "fill_value", property(fill_value))
