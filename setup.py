from setuptools import find_packages
from setuptools import setup

import os

install_requires = [
    "numpy>=1.12.0",
    "scipy>=0.18.1",
    "scikit-learn>=0.19.1",
    "decorator>=4.3.0",
    "pandas>=0.25,<1.4",
    "packaging",
]

optional_requires = ["fcsparser", "tables", "h5py", "anndata", "anndata2ri>=1.0.6"]

test_requires = [
    "nose",
    "nose2",
    "coverage",
    "coveralls",
    "parameterized",
    "requests",
    "packaging",
    "mock",
    "h5py",
    "matplotlib>=3.0",
    "rpy2>=3.0",
    "black",
]

doc_requires = [
    "sphinx>=2.2,<2.4",
    "sphinxcontrib-napoleon",
    "ipykernel",
    "nbsphinx",
    "autodocsumm",
]

version_py = os.path.join(os.path.dirname(__file__), "scprep", "version.py")
version = open(version_py).read().strip().split("=")[-1].replace('"', "").strip()

readme = open("README.rst").read()

setup(
    name="scprep",
    version=version,
    description="scprep",
    author="Scott Gigante, Daniel Burkhardt and Jay Stanley, Yale University",
    author_email="krishnaswamylab@gmail.com",
    packages=find_packages(),
    license="GNU General Public License Version 3",
    install_requires=install_requires,
    python_requires=">=3.6",
    extras_require={
        "test": test_requires + optional_requires,
        "doc": doc_requires,
        "optional": optional_requires,
    },
    test_suite="nose2.collector.collector",
    long_description=readme,
    url="https://github.com/KrishnaswamyLab/scprep",
    download_url="https://github.com/KrishnaswamyLab/scprep/archive/v{}.tar.gz".format(
        version
    ),
    keywords=[
        "big-data",
        "computational-biology",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Framework :: Jupyter",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
