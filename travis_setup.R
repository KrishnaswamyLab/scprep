lib.path <- "~/R/Library"
dir.create(lib.path)
.libPaths(c(lib.path, .libPaths()))
chooseCRANmirror(ind=1)
install.packages("BiocManager", quietly=TRUE)
BiocManager::install("splatter", quietly=TRUE)
