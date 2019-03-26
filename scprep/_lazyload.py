import importlib
import sys

# Key:
# { module : [{submodule1:[subsubmodule1, subsubmodule2]}, submodule2] }
# each module loads submodules on initialization but is only imported
# and loads methods/classes when these are accessed
importspec = {
    'matplotlib': ['colors', 'pyplot', 'animation', 'cm',
                   'axes', 'lines', 'ticker', 'transforms'],
    'mpl_toolkits': ['mplot3d'],
    'fcsparser': ['api'],
    'h5py': [],
    'tables': []
}


class AliasModule(object):

    def __init__(self, name, members=None):
        if members is None:
            members = []
        # easy access to AliasModule members to avoid recursionerror
        self.__module_name__ = name
        self.__module_members__ = members
        # always import these members if they exist
        self.__implicit_members__ = [
            '__version__', '__warning_registry__', '__file__',
            '__loader__', '__path__', '__doc__', '__package__']
        self.__loaded__ = False
        # create submodules
        submodules = []
        for member in members:
            if isinstance(member, dict):
                for submodule, submembers in member.items():
                    setattr(self, submodule, AliasModule(
                        "{}.{}".format(name, submodule), submembers))
                    submodules.append(submodule)
            else:
                setattr(self, member, AliasModule(
                        "{}.{}".format(name, member)))
                submodules.append(member)
        setattr(self, "__submodules__", submodules)

    def __getattribute__(self, attr):
        # easy access to AliasModule members to avoid recursionerror
        super_getattr = super().__getattribute__
        if attr in super_getattr("__submodules__"):
            # accessing a submodule, return directly
            return super_getattr(attr)
        else:
            # accessing an unknown member
            if not super_getattr("__loaded__"):
                # module hasn't been imported yet
                importlib.import_module(super_getattr("__module_name__"))
            # access lazy loaded member from loaded module
            return getattr(sys.modules[super_getattr("__module_name__")], attr)


# load required aliases into global namespace
# these can be imported as
# from scprep._lazyload import matplotlib
# plt = matplotlib.pyplot
for module, members in importspec.items():
    globals()[module] = AliasModule(module, members)
