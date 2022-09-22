from . import r_function

import pandas as pd

_install_dyngen = r_function.RFunction(
    args="""lib=.libPaths()[1], dependencies=NA,
            repos='http://cran.rstudio.com', verbose=TRUE""",
    body="""
        install.packages(
            c("dynwrap", "dyngen"),
            lib=lib,
            repos=repos,
            dependencies=dependencies
        )
    """,
)

_get_backbones = r_function.RFunction(
    setup="""
        library(dyngen)
    """,
    body="""
        names(list_backbones())
    """,
)

_DyngenSimulate = r_function.RFunction(
    args="""
        backbone_name=character(), num_cells=500, num_tfs=100, num_targets=50,
        num_hks=25,simulation_census_interval=10, compute_cellwise_grn=FALSE,
        compute_rna_velocity=FALSE, n_jobs=7, random_state=NA, verbose=TRUE
    """,
    setup="""
        library(dyngen)
    """,
    body="""
        if (!is.na(random_state)) {
            set.seed(random_state)
        }

        backbones <- list('bifurcating'=backbone_bifurcating(),
                  'bifurcating_converging'=backbone_bifurcating_converging(),
                  'bifurcating_cycle'=backbone_bifurcating_cycle(),
                  'bifurcating_loop'=backbone_bifurcating_loop(),
                  'binary_tree'=backbone_binary_tree(),
                  'branching'=backbone_branching(),
                  'consecutive_bifurcating'=backbone_consecutive_bifurcating(),
                  'converging'=backbone_converging(),
                  'cycle'=backbone_cycle(),
                  'cycle_simple'=backbone_cycle_simple(),
                  'disconnected'=backbone_disconnected(),
                  'linear'=backbone_linear(),
                  'linear_simple'=backbone_linear_simple(),
                  'trifurcating'=backbone_trifurcating()
                 )

        backbone <- backbones[[backbone_name]]
        # silent default behavior of dyngen
        if (num_tfs < nrow(backbone$module_info)) {
            if (verbose) {
            cat("If input num_tfs is less than backbone default,",
                "Dyngen uses backbone default.\n")
            }
            num_tfs <- nrow(backbone$module_info)
        }
        if (verbose) {
            cat('Run Parameters:')
            cat('\n\tBackbone:', backbone_name)
            cat('\n\tNumber of Cells:', num_cells)
            cat('\n\tNumber of TFs:', num_tfs)
            cat('\n\tNumber of Targets:', num_targets)
            cat('\n\tNumber of HKs:', num_hks, '\n')
        }

        init <- initialise_model(
          backbone=backbone,
          num_cells=num_cells,
          num_tfs=num_tfs,
          num_targets=num_targets,
          num_hks=num_hks,
          simulation_params=simulation_default(
            census_interval=as.double(simulation_census_interval),
            kinetics_noise_function = kinetics_noise_simple(mean=1, sd=0.005),
            ssa_algorithm = ssa_etl(tau=300/3600),
            compute_cellwise_grn=compute_cellwise_grn,
            compute_rna_velocity=compute_rna_velocity),
          num_cores = n_jobs,
          download_cache_dir=NULL,
          verbose=verbose
        )
        out <- generate_dataset(init)
        data <- list(cell_info = as.data.frame(out$dataset$cell_info),
             expression = as.data.frame(as.matrix(out$dataset$expression)))

        if (compute_cellwise_grn) {
            data[['bulk_grn']] <- as.data.frame(out$dataset$regulatory_network)
            data[['cellwise_grn']] <- as.data.frame(out$dataset$regulatory_network_sc)
        }
        if (compute_rna_velocity) {
            data[['rna_velocity']] <- as.data.frame(as.matrix(out$dataset$rna_velocity))
        }

        data
    """,
)


def install(
    lib=None,
    dependencies=None,
    repos="http://cran.us.r-project.org",
    verbose=True,
):
    """Install Dyngen from CRAN.

    Parameters
    ----------
    lib: string
        Directory to install the package.
        If missing, defaults to the first element of .libPaths().
    dependencies: boolean, optional (default: None/NA)
        When True, installs all packages specified under "Depends", "Imports",
        "LinkingTo" and "Suggests".
        When False, installs no dependencies.
        When None/NA, installs all packages specified under "Depends", "Imports"
        and "LinkingTo".
    repos: string, optional (default: "http://cran.us.r-project.org"):
        R package repository.
    verbose: boolean, optional (default: True)
        Install script verbosity.
    """

    kwargs = {}
    if lib is not None:
        kwargs["lib"] = lib
    if dependencies is not None:
        kwargs["dependencies"] = dependencies

    _install_dyngen(
        repos=repos,
        verbose=verbose,
        **kwargs,
    )


def get_backbones():
    """Output full list of cell trajectory backbones.

    Returns
    -------
    backbones: array of backbone names
    """
    return _get_backbones()


def DyngenSimulate(
    backbone,
    num_cells=500,
    num_tfs=100,
    num_targets=50,
    num_hks=25,
    simulation_census_interval=10,
    compute_cellwise_grn=False,
    compute_rna_velocity=False,
    n_jobs=7,
    random_state=None,
    verbose=True,
    force_num_cells=False,
):
    """Simulate dataset with cellular backbone.

    The backbone determines the overall dynamic process during a simulation.
    It consists of a set of gene modules, which regulate each other such that
    expression of certain genes change over time in a specific manner.

    DyngenSimulate is a Python wrapper for the R package Dyngen.
    Default values obtained from Github vignettes.
    For more details, read about Dyngen on Github_.

    .. _Github: https://github.com/dynverse/dyngen

    Parameters
    ----------
    backbone: string
           Backbone name from dyngen list of backbones.
           Get list with get_backbones()).
    num_cells: int, optional (default: 500)
           Number of cells.
    num_tfs: int, optional (default: 100)
           Number of transcription factors.
           The TFs are the main drivers of the molecular changes in the simulation.
           A TF can only be regulated by other TFs or itself.

           NOTE: If num_tfs input is less than nrow(backbone$module_info),
           Dyngen will default to nrow(backbone$module_info).
           This quantity varies between backbones and with each run (without seed).
           It is generally less than 75.
           It is recommended to input num_tfs >= 100 to stabilize the output.
    num_targets: int, optional (default: 50)
           Number of target genes.
           Target genes are regulated by a TF or another target gene,
           but are always downstream of at least one TF.
    num_hks: int, optional (default: 25)
           Number of housekeeping genees.
           Housekeeping genes are completely separate from any TFs or target genes.
    simulation_census_interval: int, optional (default: 10)
           Stores the abundance levels only after a specific interval has passed.
           The lower the interval, the higher detail of simulation trajectory retained,
           though many timepoints will contain similar information.
    compute_cellwise_grn: boolean, optional (default: False)
           If True, computes the ground truth cellwise gene regulatory networks.
           Also outputs ground truth bulk (entire dataset) regulatory network.
           NOTE: Increases compute time significantly.
    compute_rna_velocity: boolean, optional (default: False)
           If true, computes the ground truth propensity ratios after simulation.
           NOTE: Increases compute time significantly.
    n_jobs: int, optional (default: 8)
           Number of cores to use.
    random_state: int, optional (default: None)
           Fixes seed for simulation generator.
    verbose: boolean, optional (default: True)
           Data generation verbosity.
    force_num_cells: boolean, optional (default: False)
           Dyngen occassionally produces fewer cells than specified.
           Set this flag to True to rerun Dyngen until correct cell count is reached.

    Returns
    -------
    Dictionary data of pd.DataFrames:
    data['cell_info']: pd.DataFrame, shape (n_cells, 4)
           Columns: cell_id, step_ix, simulation_i, sim_time
           sim_time is the simulated timepoint for a given cell.

    data['expression']: pd.DataFrame, shape (n_cells, n_genes)
           Log-transformed counts with dropout.

    If compute_cellwise_grn is True,
    data['bulk_grn']: pd.DataFrame, shape (n_tf_target_interactions, 4)
           Columns: regulator, target, strength, effect.
           Strength is positive and unbounded.
           Effect is either +1 (for activation) or -1 (for inhibition).

    data['cellwise_grn']: pd.DataFrame, shape (n_tf_target_interactions_per_cell, 4)
           Columns: cell_id, regulator, target, strength.
           The output does not include all edges per cell.
           The regulatory effect lies between [−1, 1], where -1 is complete inhibition
           of target by TF, +1 is maximal activation of target by TF,
           and 0 is inactivity of the regulatory interaction between R and T.

    If compute_rna_velocity is True,
    data['rna_velocity']: pd.DataFrame, shape (n_cells, n_genes)
           Propensity ratios for each cell.

    Example
    --------
    >>> import scprep
    >>> scprep.run.dyngen.install()
    >>> backbones = scprep.run.dyngen.get_backbones()
    >>> data = scprep.run.DyngenSimulate(backbone=backbones[0])
    """
    if backbone not in get_backbones():
        raise ValueError(
            (
                "Input not in default backbone list. "
                "Choose backbone from get_backbones()"
            )
        )

    kwargs = {}
    if random_state is not None:
        kwargs["random_state"] = random_state

    rdata = _DyngenSimulate(
        backbone_name=backbone,
        num_cells=num_cells,
        num_tfs=num_tfs,
        num_targets=num_targets,
        num_hks=num_hks,
        simulation_census_interval=simulation_census_interval,
        compute_cellwise_grn=compute_cellwise_grn,
        compute_rna_velocity=compute_rna_velocity,
        n_jobs=n_jobs,
        verbose=verbose,
        rpy_verbose=verbose,
        **kwargs,
    )

    if force_num_cells:
        if random_state is None:
            random_state = -1

        if pd.DataFrame(rdata["cell_info"]).shape[0] != num_cells:
            random_state += 1
            rdata = DyngenSimulate(
                backbone=backbone,
                num_cells=num_cells,
                num_tfs=num_tfs,
                num_targets=num_targets,
                num_hks=num_hks,
                simulation_census_interval=simulation_census_interval,
                compute_cellwise_grn=compute_cellwise_grn,
                compute_rna_velocity=compute_rna_velocity,
                n_jobs=n_jobs,
                verbose=verbose,
                random_state=random_state,
                force_num_cells=force_num_cells,
            )

    data = {}
    data["cell_info"] = pd.DataFrame(rdata["cell_info"])
    data["expression"] = pd.DataFrame(rdata["expression"])
    if compute_cellwise_grn:
        data["cellwise_grn"] = pd.DataFrame(rdata["cellwise_grn"])
        data["bulk_grn"] = pd.DataFrame(rdata["bulk_grn"])
    if compute_rna_velocity:
        data["rna_velocity"] = pd.DataFrame(rdata["rna_velocity"])

    return data
