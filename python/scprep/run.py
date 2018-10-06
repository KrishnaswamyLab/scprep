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
    _run_MAST = _rpy2_function(_MAST_r_script)
    return pd.DataFrame.from_records(_run_MAST(data, gene_names=gene_names, condition_labels=condition_labels))
