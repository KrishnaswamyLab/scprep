from rpy2.robjects.packages import STAP
import rpy2.robjects.numpy2ri
import numpy as np
import pandas as pd

rpy2.robjects.numpy2ri.activate()

def _rpy2_function(r_script):
    r_script_STAP = STAP(r_script, "process_data")
    def py_function(data, **kwargs):
        r_object = r_script_STAP.process_data(data.T, **kwargs)
        return rpy2.robjects.numpy2ri.ri2py(r_object)
    return py_function

## TODO: Figure out how to check if the nescessary package is already installed
## TODO: Make rpy2 not a depedency
## TODO: Is keeping these scripts in this file the best way to maintain code?
## TODO: Make geneson optional for MAST, adding option for releveling the condition factor

_MAST_r_script = """
suppressPackageStartupMessages({
        library(MAST)
        library(data.table)
        })

process_data <- function(data, gene_names, condition_labels) {
    FCTHRESHOLD <- log2(1.5) # provided from https://bit.ly/2QB5D6D
    fdat <- data.frame(primerid = factor(gene_names))

    sca <- FromMatrix(data, fData = fdat)

    # calculate number of genes detected per cell, called DetRate
    cdr2 <- colSums(assay(sca)>0)
    colData(sca)$cngeneson <- scale(cdr2, scale = FALSE) # zscore

    colData(sca)$condition <- factor(condition_labels)
    zlmCond <- zlm(~condition + cngeneson, sca)
    contr <- paste('condition', tail(levels(factor(condition_labels)), n=1), sep='')

    summaryCond <- summary(zlmCond, doLRT=contr)

    summaryDt <- summaryCond$datatable
    fcHurdle <- merge(summaryDt[contrast==contr & component=='H',.(primerid, `Pr(>Chisq)`)], #hurdle P values
                    summaryDt[contrast==contr & component=='logFC', .(primerid, coef, ci.hi, ci.lo)], by='primerid') #logFC coefficients

    fcHurdle[,fdr:=p.adjust(`Pr(>Chisq)`, 'fdr')]
    fcHurdleSig <- merge(fcHurdle[fdr<.05 & abs(coef)>FCTHRESHOLD], as.data.table(mcols(sca)), by='primerid')
    setorder(fcHurdleSig, fdr)
    return(fcHurdleSig) }

"""

def run_MAST(data, gene_names, condition_labels):
    '''
    Takes a data matrix and performs pairwise differential expression analysis
    using a Hurdle model as implemented in [MAST](https://github.com/RGLab/MAST/).
    The current implementation uses the Cell Detection Rate (# non-zero genes per cell)
    as a factor in the analysis because this was found to be higher performing in
    a comparison of differential expression tools (https://www.ncbi.nlm.nih.gov/pubmed/29481549).

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data from both samples
    gene_names : array-like, shape=[n_features]
        Names of each gene - must be unique
    condition_labels : array-like, shape=[n_samples]
        A vector of condition labels associated with each sample.
        E.g. `['Expt', 'Expt', ... , 'Ctrl']`.
        Note that `len(np.unique(condition_labels))` must be `2`.

    Returns
    -------
    results : pandas.DataFrame
        A table of results with columns `Pr(>Chisq)`, `coef`, `ci.hi`, `ci.lo`, and	`fdr`.
        `Pr(>Chisq)`: the p-value associated with the gene
        `coef`: the estimated log-Fold-Change (logFC) associated with the Hurdle model
        `ci.hi`: upper bound of the confidence interval for logFC
        `ci.lo`: lower bound of the confidence interval for logFC
        `fdr`: false-discovery rate
        The number of genes in the table is the number of significant genes at a
        false discovery rate of 0.05.

    Examples
    --------
    >>> import scprep
    >>> data = scprep.io.load_csv("my_data.csv")
    >>> data_ln = scprep.normalize.library_size_normalize(data)
    >>> cond = np.hstack([np.tile('cond1', ncells_in_cond1), np.tile('cond2', ncells_in_cond2)])
    >>> results = scprep.run.run_MAST(np.log2(data_ln + 1), gene_names = data.columns, condition = cond)
    """
    '''
    _run_MAST = _rpy2_function(_MAST_r_script)
    results = pd.DataFrame.from_records(_run_MAST(data, gene_names=gene_names, condition_labels=condition_labels), index='primerid')
    results.index.names = ['gene_name']
    return results
