import importlib
import sys

"""Key:

{ module : [{submodule1:[subsubmodule1, subsubmodule2]}, submodule2] }

Each module loads submodules on initialization but is only imported
and loads methods/classes when these are accessed.
"""
_importspec = {
    "matplotlib": [
        "colors",
        "pyplot",
        "animation",
        "cm",
        "axes",
        "lines",
        "ticker",
        "transforms",
    ],
    "mpl_toolkits": ["mplot3d"],
    "fcsparser": ["api"],
    "rpy2": [
        {"robjects": ["numpy2ri", "pandas2ri", "packages", "vectors", "conversion"]},
        "rinterface",
        {"rinterface_lib": ["callbacks", "embedded"]},
    ],
    "h5py": [],
    "tables": [],
    "requests": [],
    "anndata2ri": [],
}


class AliasModule(object):
    """Wrapper around Python module to allow lazy loading."""

    def __init__(self, name, members=None):
        """Initialize a module without loading it.

        Parameters
        ----------
        name : str
            Module name
        members : list[str, dict]
            List of submodules to be loaded as AliasModules. If a dict, the submodule
            is loaded with subsubmodules corresponding to the dictionary values;
            if a string, the submodule has no subsubmodules.
        """
        # easy access to AliasModule members to avoid recursionerror
        super_setattr = super().__setattr__
        if members is None:
            members = []
        builtin_members = ["__class__", "__doc__"]
        super_setattr("__module_name__", name)
        # create submodules
        submodules = []
        for member in members:
            if isinstance(member, dict):
                for submodule, submembers in member.items():
                    super_setattr(
                        submodule,
                        AliasModule("{}.{}".format(name, submodule), submembers),
                    )
                    submodules.append(submodule)
            else:
                super_setattr(member, AliasModule("{}.{}".format(name, member)))
                submodules.append(member)
        super_setattr("__submodules__", submodules)
        super_setattr("__builtin_members__", builtin_members)

    @property
    def __loaded_module__(self):
        """Load the module, or retrieve it if already loaded."""
        # easy access to AliasModule members to avoid recursionerror
        super_getattr = super().__getattribute__
        name = super_getattr("__module_name__")
        try:
            return sys.modules[name]
        except KeyError:
            # module hasn't been imported yet
            importlib.import_module(name)
            return sys.modules[name]

    def __getattribute__(self, attr):
        """Access AliasModule members."""
        # easy access to AliasModule members to avoid recursionerror
        super_getattr = super().__getattribute__
        if attr in super_getattr("__submodules__"):
            # accessing a submodule, return directly
            return super_getattr(attr)
        elif attr in super_getattr("__builtin_members__"):
            if super_getattr("__module_name__") in sys.modules:
                # already loaded, return the attribute on the module
                return getattr(super_getattr("__loaded_module__"), attr)
            else:
                # not loaded, return the attribute from this class
                return super_getattr(attr)
        else:
            # accessing an unknown member
            # access lazy loaded member from loaded module
            return getattr(super_getattr("__loaded_module__"), attr)

    def __setattr__(self, name, value):
        """Allow monkey-patching.

        Gives easy access to AliasModule members to avoid recursionerror.
        """
        super_getattr = super().__getattribute__
        return setattr(super_getattr("__loaded_module__"), name, value)


# load required aliases into global namespace
# these can be imported as
# from scprep._lazyload import matplotlib
# plt = matplotlib.pyplot
for module, members in _importspec.items():
    globals()[module] = AliasModule(module, members)
