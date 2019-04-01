lib.path <- "./.Rpackages"
dir.create(lib.path)
.libPaths(c(.libPaths(), lib.path))
chooseCRANmirror(ind=1)
install.packages("BiocManager", quietly=TRUE)
BiocManager::install("splatter", quietly=TRUE)
