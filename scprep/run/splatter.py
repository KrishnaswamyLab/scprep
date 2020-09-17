import numpy as np
import numbers

from . import r_function


def _sum_to_one(x):
    x = x / np.sum(x)  # fix numerical error
    x = x.round(5)
    if not isinstance(x, numbers.Number):
        x[0] += 1 - np.sum(x)
    x = x.round(5)
    return x


def install(site_repository=None, update=False, version=None, verbose=True):
    """Install the required R packages to run Splatter

    Parameters
    ----------
    site_repository : string, optional (default: None)
        additional repository in which to look for packages to install.
        This repository will be prepended to the default repositories
    update : boolean, optional (default: False)
        When False, don't attempt to update old packages.
        When True, update old packages automatically.
    version : string, optional (default: None)
        Bioconductor version to install, e.g., version = "3.8".
        The special symbol version = "devel" installs the current 'development' version.
        If None, installs from the current version.
    verbose : boolean, optional (default: True)
        Install script verbosity.
    """
    r_function.install_bioconductor(
        "splatter",
        site_repository=site_repository,
        update=update,
        version=version,
        verbose=verbose,
    )


_SplatSimulate = r_function.RFunction(
    setup="""
        library(splatter)
    """,
    args="""
        method='paths',
        batch_cells=100, n_genes=10000,
        batch_fac_loc=0.1, batch_fac_scale=0.1,
        mean_rate=0.3, mean_shape=0.6,
        lib_loc=11, lib_scale=0.2, lib_norm=False,
        out_prob=0.05,
        out_fac_loc=4, out_fac_scale=0.5,
        de_prob=0.1, de_down_prob=0.1,
        de_fac_loc=0.1, de_fac_scale=0.4,
        bcv_common=0.1, bcv_df=60,
        dropout_type='none',
        dropout_mid=0, dropout_shape=-1,
        group_prob=1,
        path_from=0, path_length=100, path_skew=0.5,
        path_nonlinear_prob=0.1, path_sigma_fac=0.8,
        seed=0
    """,
    body="""
        batch_cells <- as.numeric(batch_cells)
        group_prob <- as.numeric(group_prob)
        path_from <- as.numeric(path_from)
        path_length <- as.numeric(path_length)
        path_skew <- as.numeric(path_skew)
        dropout_mid <- as.numeric(dropout_mid)
        dropout_shape <- as.numeric(dropout_shape)
        de_fac_loc <- as.numeric(de_fac_loc)
        de_down_prob <- as.numeric(de_down_prob)
        de_fac_scale <- as.numeric(de_fac_scale)

        sim <- splatSimulate(
            method=method,
            group.prob=group_prob,
            batchCells=batch_cells, nGenes=n_genes,
            batch.facLoc=batch_fac_loc, batch.facScale=batch_fac_scale,
            mean.rate=mean_rate, mean.shape=mean_shape,
            lib.loc=lib_loc, lib.scale=lib_scale, lib.norm=lib_norm,
            out.prob=out_prob,
            out.facLoc=out_fac_loc, out.facScale=out_fac_scale,
            de.prob=de_prob, de.downProb=de_down_prob,
            de.facLoc=de_fac_loc, de.facScale=de_fac_scale,
            bcv.common=bcv_common, bcv.df=bcv_df,
            dropout.type=dropout_type, dropout.mid=dropout_mid,
            dropout.shape=dropout_shape,
            path.from=path_from, path.length=path_length, path.skew=path_skew,
            path.nonlinearProb=path_nonlinear_prob, path.sigmaFac=path_sigma_fac,
            seed=seed
        )

        result <- list(counts=t(counts(sim)),
            group=colData(sim)$Group, step=colData(sim)$Step,
            batch=colData(sim)$Batch,
            exp_lib_size=colData(sim)$ExpLibSize,
            base_gene_mean=rowData(sim)$BaseGeneMean,
            outlier_factor=rowData(sim)$OutlierFactor,
            batch_cell_means=t(assays(sim)$BatchCellMeans),
            base_cell_means=t(assays(sim)$BaseCellMeans),
            bcv=t(assays(sim)$BCV), cell_means=t(assays(sim)$CellMeans),
            true_counts=t(assays(sim)$TrueCounts),
            dropout=if (is.null(assays(sim)$Dropout)) NULL else t(assays(sim)$Dropout))
        row_data <- as.data.frame(rowData(sim))
        if (any(startsWith(names(row_data), "BatchFac"))) {
            batch_fac <- as.data.frame(row_data[,startsWith(names(row_data), "BatchFac")])
            names(batch_fac) <- paste("batch_fac", 1:ncol(batch_fac), sep="_")
            batch_fac <- as.list(batch_fac)
            result <- c(result, batch_fac)
        }
        if (any(startsWith(names(row_data), "DEFac"))) {
            de_fac <- as.data.frame(row_data[,startsWith(names(row_data), "DEFac")])
            names(de_fac) <- paste("de_fac", 1:ncol(de_fac), sep="_")
            de_fac <- as.list(de_fac)
            result <- c(result, de_fac)
        }
        if (any(startsWith(names(row_data), "SigmaFac"))) {
            sigma_fac <- as.data.frame(row_data[,startsWith(names(row_data), "SigmaFac")])
            colnames(sigma_fac) <- paste("sigma_fac", 1:ncol(sigma_fac), sep="_")
            sigma_fac <- as.list(sigma_fac)
            result <- c(result, sigma_fac)
        }
        result
    """,
)


def SplatSimulate(
    method="paths",
    batch_cells=100,
    n_genes=10000,
    batch_fac_loc=0.1,
    batch_fac_scale=0.1,
    mean_rate=0.3,
    mean_shape=0.6,
    lib_loc=11,
    lib_scale=0.2,
    lib_norm=False,
    out_prob=0.05,
    out_fac_loc=4,
    out_fac_scale=0.5,
    de_prob=0.1,
    de_down_prob=0.1,
    de_fac_loc=0.1,
    de_fac_scale=0.4,
    bcv_common=0.1,
    bcv_df=60,
    dropout_type="none",
    dropout_prob=0.5,
    dropout_mid=0,
    dropout_shape=-1,
    group_prob=1,
    path_from=0,
    path_length=100,
    path_skew=0.5,
    path_nonlinear_prob=0.1,
    path_sigma_fac=0.8,
    seed=None,
    verbose=1,
):
    """Simulate count data from a fictional single-cell RNA-seq experiment using the Splat method.

    SplatSimulate is a Python wrapper for the R package Splatter. For more
    details, read about Splatter on GitHub_ and Bioconductor_.

    .. _GitHub: https://github.com/Oshlack/splatter
    .. _Bioconductor: https://bioconductor.org/packages/release/bioc/html/splatter.html

    Parameters
    ----------
    batch_cells : list-like or int, optional (default: 100)
        The number of cells in each batch.
    n_genes : int, optional (default:10000)
        The number of genes to simulate.
    batch_fac_loc : float, optional (default: 0.1)
        Location (meanlog) parameter for the batch effects factor
        log-normal distribution.
    batch_fac_scale : float, optional (default: 0.1)
        Scale (sdlog) parameter for the batch effects factor
        log-normal distribution.
    mean_shape : float, optional (default: 0.3)
        Shape parameter for the mean gamma distribution.
    mean_rate : float, optional (default: 0.6)
        Rate parameter for the mean gamma distribution.
    lib_loc : float, optional (default: 11)
        Location (meanlog) parameter for the library size
        log-normal distribution, or mean for the normal distribution.
    lib_scale : float, optional (default: 0.2)
        Scale (sdlog) parameter for the library size log-normal distribution,
        or sd for the normal distribution.
    lib_norm : bool, optional (default: False)
        Whether to use a normal distribution instead of the usual
        log-normal distribution.
    out_prob : float, optional (default: 0.05)
        Probability that a gene is an expression outlier.
    out_fac_loc : float, optional (default: 4)
        Location (meanlog) parameter for the expression outlier factor
        log-normal distribution.
    out_fac_scale : float, optional (default: 0.5)
        Scale (sdlog) parameter for the expression outlier factor
        log-normal distribution.
    de_prob : float, optional (default: 0.1)
        Probability that a gene is differentially expressed in each
        group or path.
    de_down_prob : float, optional (default: 0.1)
        Probability that a differentially expressed gene is down-regulated.
    de_fac_loc : float, optional (default: 0.1)
        Location (meanlog) parameter for the differential expression factor
        log-normal distribution.
    de_fac_scale : float, optional (default: 0.4)
        Scale (sdlog) parameter for the differential expression factor
        log-normal distribution.
    bcv_common : float, optional (default: 0.1)
        Underlying common dispersion across all genes.
    bcv_df float, optional (default: 60)
        Degrees of Freedom for the BCV inverse chi-squared distribution.
    dropout_type : {'none', 'experiment', 'batch', 'group', 'cell', 'binomial'}, optional (default: 'none')
        The type of dropout to simulate. "none" indicates no dropout,
        "experiment" is global dropout using the same parameters for every
        cell, "batch" uses the same parameters for every cell in each batch,
        "group" uses the same parameters for every cell in each groups,
        "cell" uses a different set of parameters for each cell, and
        "binomial" performs post-hoc binomial undersampling.
    dropout_mid : list-like or float, optional (default: 0)
        Midpoint parameter for the dropout logistic function.
    dropout_shape : list-like or float, optional (default: -1)
        Shape parameter for the dropout logistic function.
    dropout_prob : float, optional (default: 0.5)
        Probability for binomial undersampling dropout.
    group_prob : list-like or int, optional (default: 1, shape=[n_groups])
        The probabilities that cells come from particular groups.
    path_from : list-like, optional (default: 0, shape=[n_groups])
        Vector giving the originating point of each path.
    path_length : list-like, optional (default: 100, shape=[n_groups])
        Vector giving the number of steps to simulate along each path.
    path_skew : list-like, optional (default: 0.5, shape=[n_groups])
        Vector giving the skew of each path.
    path_nonlinear_prob : float, optional (default: 0.1)
        Probability that a gene changes expression in a non-linear way along
        the differentiation path.
    path_sigma_fac : float, optional (default: 0.8)
        Sigma factor for non-linear gene paths.
    seed : int or None, optional (default: None)
        Seed to use for generating random numbers.
    verbose : int, optional (default: 1)
        Logging verbosity between 0 and 2.

    Returns
    -------
    sim : dict
        counts : Simulated expression counts.
        group : The group or path the cell belongs to.
        batch : The batch the cell was sampled from.
        exp_lib_size : The expected library size for that cell.
        step (paths only) : how far along the path each cell is.
        base_gene_mean : The base expression level for that gene.
        outlier_factor : Expression outlier factor for that gene. Values of 1 indicate the gene is not an expression outlier.
        gene_mean : Expression level after applying outlier factors.
        batch_fac_[batch] : The batch effects factor for each gene for a particular batch.
        de_fac_[group] : The differential expression factor for each gene in a particular group. Values of 1 indicate the gene is not differentially expressed.
        sigma_fac_[path] : Factor applied to genes that have non-linear changes in expression along a path.
        batch_cell_means : The mean expression of genes in each cell after adding batch effects.
        base_cell_means : The mean expression of genes in each cell after any differential expression and adjusted for expected library size.
        bcv : The Biological Coefficient of Variation for each gene in each cell.
        cell_means : The mean expression level of genes in each cell adjusted for BCV.
        true_counts : The simulated counts before dropout.
        dropout : Logical matrix showing which values have been dropped in which cells.
    """
    if seed is None:
        seed = np.random.randint(2 ** 16 - 1)
    if dropout_type == "binomial":
        dropout_type = "none"
    else:
        dropout_prob = None
    np.random.seed(seed)

    group_prob = _sum_to_one(group_prob)

    sim = _SplatSimulate(
        method=method,
        batch_cells=batch_cells,
        n_genes=n_genes,
        batch_fac_loc=batch_fac_loc,
        batch_fac_scale=batch_fac_scale,
        mean_rate=mean_rate,
        mean_shape=mean_shape,
        lib_loc=lib_loc,
        lib_scale=lib_scale,
        lib_norm=lib_norm,
        out_prob=out_prob,
        out_fac_loc=out_fac_loc,
        out_fac_scale=out_fac_scale,
        de_prob=de_prob,
        de_down_prob=de_down_prob,
        de_fac_loc=de_fac_loc,
        de_fac_scale=de_fac_scale,
        bcv_common=bcv_common,
        bcv_df=bcv_df,
        dropout_type=dropout_type,
        dropout_mid=dropout_mid,
        dropout_shape=dropout_shape,
        group_prob=group_prob,
        path_from=path_from,
        path_length=path_length,
        path_skew=path_skew,
        path_nonlinear_prob=path_nonlinear_prob,
        path_sigma_fac=path_sigma_fac,
        seed=seed,
        rpy_verbose=verbose,
    )
    if dropout_prob is not None:
        sim["counts"] = np.random.binomial(
            n=sim["counts"], p=1 - dropout_prob, size=sim["counts"].shape
        )
    return sim
