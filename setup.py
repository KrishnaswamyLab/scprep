import os
import sys
from setuptools import setup, find_packages

install_requires = [
    "numpy>=1.12.0",
    "scipy>=0.18.1",
    "scikit-learn>=0.19.1",
    "decorator>=4.3.0",
    "pandas>=0.25",
]

test_requires = [
    "nose",
    "nose2",
    "fcsparser",
    "tables",
    "h5py",
    "coverage",
    "coveralls",
    "parameterized",
    "requests",
    "packaging",
]

doc_requires = [
    "sphinx>=2.2,<2.4",
    "sphinxcontrib-napoleon",
    "ipykernel",
    "nbsphinx",
]

if sys.version_info[:2] < (3, 6):
    test_requires += ["matplotlib>=3.0,<3.1", "rpy2>=3.0,<3.1"]
    doc_requires += ["autodocsumm!=0.2.0"]
else:
    test_requires += ["matplotlib>=3.0", "rpy2>=3.0", "black"]
    doc_requires += ["autodocsumm"]

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
    license="GNU General Public License Version 2",
    install_requires=install_requires,
    python_requires=">=3.5",
    extras_require={"test": test_requires, "doc": doc_requires},
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
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
