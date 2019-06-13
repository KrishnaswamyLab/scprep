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

import pandas as _pd
if int(_pd.__version__.split(".")[1]) < 24:
    import numpy as _np

    def __rmatmul__(self, other):
        """ Matrix multiplication using binary `@` operator in Python>=3.5 """
        return self.dot(_np.transpose(other))
    _pd.core.series.Series.__rmatmul__ = __rmatmul__
