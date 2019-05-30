from tools import utils, matrix, data
import numpy as np
import scprep
import scprep.run.r_function
import unittest
import rpy2.rinterface_lib.callbacks

builtin_warning = rpy2.rinterface_lib.callbacks.consolewrite_warnerror


def test_verbose():
    fun = scprep.run.RFunction(
        setup="message('This should not print')",
        body="message('Verbose test\n\n'); list(1,2,3)", verbose=True)
    assert np.all(fun() == np.array([[1], [2], [3]]))


class TestRFunctions(unittest.TestCase):

    def test_splatter_default(self):
        sim = scprep.run.SplatSimulate(
            batch_cells=10, n_genes=200, verbose=0)
        assert sim['counts'].shape == (10, 200)
        assert np.all(sim['batch'] == 'Batch1')
        assert sim['batch_cell_means'].shape == (10, 200)
        assert sim['base_cell_means'].shape == (10, 200)
        assert sim['bcv'].shape == (10, 200)
        assert sim['cell_means'].shape == (10, 200)
        assert sim['true_counts'].shape == (10, 200)
        assert sim['dropout'] is None
        assert sim['step'].shape == (10,)
        assert sim['group'].shape == (10,)
        assert sim['exp_lib_size'].shape == (10,)
        assert sim['base_gene_mean'].shape == (200,)
        assert sim['outlier_factor'].shape == (200,)
        assert sum(['batch_fac' in k for k in sim.keys()]) == 0
        assert sum(['de_fac' in k for k in sim.keys()]) == 1
        assert sim['de_fac_1'].shape == (200,)
        assert sum(['sigma_fac' in k for k in sim.keys()]) == 1
        assert sim['sigma_fac_1'].shape == (200,)

    def test_splatter_batch(self):
        sim = scprep.run.SplatSimulate(
            batch_cells=[5, 5], n_genes=200, verbose=0)
        assert sim['counts'].shape == (10, 200)
        assert np.all(sim['batch'][:5] == 'Batch1')
        assert np.all(sim['batch'][5:] == 'Batch2')
        assert sim['batch_cell_means'].shape == (10, 200)
        assert sim['base_cell_means'].shape == (10, 200)
        assert sim['bcv'].shape == (10, 200)
        assert sim['cell_means'].shape == (10, 200)
        assert sim['true_counts'].shape == (10, 200)
        assert sim['dropout'] is None
        assert sim['step'].shape == (10,)
        assert sim['group'].shape == (10,)
        assert sim['exp_lib_size'].shape == (10,)
        assert sim['base_gene_mean'].shape == (200,)
        assert sim['outlier_factor'].shape == (200,)
        assert sum(['batch_fac' in k for k in sim.keys()]) == 2
        assert sim['batch_fac_1'].shape == (200,)
        assert sim['batch_fac_2'].shape == (200,)
        assert sum(['de_fac' in k for k in sim.keys()]) == 1
        assert sim['de_fac_1'].shape == (200,)
        assert sum(['sigma_fac' in k for k in sim.keys()]) == 1
        assert sim['sigma_fac_1'].shape == (200,)

    def test_splatter_groups(self):
        sim = scprep.run.SplatSimulate(method='groups', batch_cells=10,
                                       group_prob=[0.5, 0.5], n_genes=200,
                                       de_fac_loc=[0.1, 0.5], verbose=0)
        assert sim['counts'].shape == (10, 200)
        assert np.all(sim['batch'] == 'Batch1')
        assert sim['batch_cell_means'].shape == (10, 200)
        assert sim['base_cell_means'].shape == (10, 200)
        assert sim['bcv'].shape == (10, 200)
        assert sim['cell_means'].shape == (10, 200)
        assert sim['true_counts'].shape == (10, 200)
        assert sim['dropout'] is None
        assert sim['step'] is None
        assert sim['group'].shape == (10,)
        assert sim['exp_lib_size'].shape == (10,)
        assert sim['base_gene_mean'].shape == (200,)
        assert sim['outlier_factor'].shape == (200,)
        assert sum(['batch_fac' in k for k in sim.keys()]) == 0
        assert sum(['de_fac' in k for k in sim.keys()]) == 2
        assert sim['de_fac_1'].shape == (200,)
        assert sim['de_fac_2'].shape == (200,)
        assert sum(['sigma_fac' in k for k in sim.keys()]) == 0

    def test_splatter_paths(self):
        sim = scprep.run.SplatSimulate(method='paths', batch_cells=10, n_genes=200,
                                       group_prob=[0.5, 0.5], path_from=[0, 0],
                                       path_length=[100, 200], path_skew=[0.4, 0.6],
                                       de_fac_loc=[0.1, 0.5], verbose=0)
        assert sim['counts'].shape == (10, 200)
        assert np.all(sim['batch'] == 'Batch1')
        assert sim['batch_cell_means'].shape == (10, 200)
        assert sim['base_cell_means'].shape == (10, 200)
        assert sim['bcv'].shape == (10, 200)
        assert sim['cell_means'].shape == (10, 200)
        assert sim['true_counts'].shape == (10, 200)
        assert sim['dropout'] is None
        assert sim['step'].shape == (10,)
        assert sim['group'].shape == (10,)
        assert sim['exp_lib_size'].shape == (10,)
        assert sim['base_gene_mean'].shape == (200,)
        assert sim['outlier_factor'].shape == (200,)
        assert sum(['batch_fac' in k for k in sim.keys()]) == 0
        assert sum(['de_fac' in k for k in sim.keys()]) == 2
        assert sim['de_fac_1'].shape == (200,)
        assert sim['de_fac_2'].shape == (200,)
        assert sum(['sigma_fac' in k for k in sim.keys()]) == 2
        assert sim['sigma_fac_1'].shape == (200,)
        assert sim['sigma_fac_2'].shape == (200,)

    def test_splatter_dropout(self):
        sim = scprep.run.SplatSimulate(batch_cells=10, n_genes=200,
                                       dropout_type='experiment',
                                       verbose=0)
        assert sim['counts'].shape == (10, 200)
        assert np.all(sim['batch'] == 'Batch1')
        assert sim['batch_cell_means'].shape == (10, 200)
        assert sim['base_cell_means'].shape == (10, 200)
        assert sim['bcv'].shape == (10, 200)
        assert sim['cell_means'].shape == (10, 200)
        assert sim['true_counts'].shape == (10, 200)
        assert sim['dropout'].shape == (10, 200)
        assert sim['step'].shape == (10,)
        assert sim['group'].shape == (10,)
        assert sim['exp_lib_size'].shape == (10,)
        assert sim['base_gene_mean'].shape == (200,)
        assert sim['outlier_factor'].shape == (200,)
        assert sum(['batch_fac' in k for k in sim.keys()]) == 0
        assert sum(['de_fac' in k for k in sim.keys()]) == 1
        assert sim['de_fac_1'].shape == (200,)
        assert sum(['sigma_fac' in k for k in sim.keys()]) == 1
        assert sim['sigma_fac_1'].shape == (200,)

    def test_splatter_dropout_binomial(self):
        sim = scprep.run.SplatSimulate(batch_cells=10, n_genes=200,
                                       dropout_type='binomial',
                                       dropout_prob=0.5, verbose=False)
        assert sim['counts'].shape == (10, 200)
        assert np.all(sim['batch'] == 'Batch1')
        assert sim['batch_cell_means'].shape == (10, 200)
        assert sim['base_cell_means'].shape == (10, 200)
        assert sim['bcv'].shape == (10, 200)
        assert sim['cell_means'].shape == (10, 200)
        assert sim['true_counts'].shape == (10, 200)
        dropout_proportion = np.mean(
            sim['counts'][np.where(sim['true_counts'] > 0)] /
            sim['true_counts'][np.where(sim['true_counts'] > 0)])
        assert dropout_proportion < 0.55
        assert dropout_proportion > 0.45
        assert sim['dropout'] is None
        assert sim['step'].shape == (10,)
        assert sim['group'].shape == (10,)
        assert sim['exp_lib_size'].shape == (10,)
        assert sim['base_gene_mean'].shape == (200,)
        assert sim['outlier_factor'].shape == (200,)
        assert sum(['batch_fac' in k for k in sim.keys()]) == 0
        assert sum(['de_fac' in k for k in sim.keys()]) == 1
        assert sim['de_fac_1'].shape == (200,)
        assert sum(['sigma_fac' in k for k in sim.keys()]) == 1
        assert sim['sigma_fac_1'].shape == (200,)

    def test_splatter_warning(self):
        assert rpy2.rinterface_lib.callbacks.consolewrite_warnerror is \
            builtin_warning
        scprep.run.r_function._ConsoleWarning.set_debug()
        assert rpy2.rinterface_lib.callbacks.consolewrite_warnerror is \
            scprep.run.r_function._ConsoleWarning.debug
        scprep.run.r_function._ConsoleWarning.set_warning()
        assert rpy2.rinterface_lib.callbacks.consolewrite_warnerror is \
            scprep.run.r_function._ConsoleWarning.warning
        scprep.run.r_function._ConsoleWarning.set_builtin()
        assert rpy2.rinterface_lib.callbacks.consolewrite_warnerror is \
            builtin_warning
