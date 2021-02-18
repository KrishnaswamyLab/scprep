import packaging.version
import pandas as pd


def patch_fill_value():
    if packaging.version.parse(pd.__version__) < packaging.version.parse("2.0"):

        def _fill_value(self):
            # Used in reindex_indexer
            try:
                return self.values.dtype.fill_value
            except AttributeError:
                return self.values.dtype.na_value

        from pandas.core.internals.blocks import ExtensionBlock

        setattr(ExtensionBlock, "fill_value", property(_fill_value))
