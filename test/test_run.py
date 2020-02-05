import sys

if int(sys.version.split(".")[1]) < 6:
    # python 3.5
    pass
else:
    from tools import utils, matrix, data
    import numpy as np
    import pandas as pd
    import scprep
    import scprep.run.r_function
    import unittest
    import sklearn.cluster
    import rpy2.rinterface_lib.callbacks
    import rpy2.rinterface_lib.embedded

    builtin_warning = rpy2.rinterface_lib.callbacks.consolewrite_warnerror

    def test_verbose():
        fun = scprep.run.RFunction(
            setup="message('This should not print')",
            body="message('Verbose test\n\n'); list(1,2,3)",
            verbose=True,
        )
        assert np.all(fun() == np.array([[1], [2], [3]]))

    def test_install_bioc():
        utils.assert_raises_message(
            rpy2.rinterface_lib.embedded.RRuntimeError,
            "Error: Bioconductor version '3.1' requires R version '3.2'; see",
            scprep.run.install_bioconductor,
            version="3.1",
            site_repository="https://bioconductor.org/packages/3.1/bioc",
            verbose=False,
        )

    class TestSplatter(unittest.TestCase):
        @classmethod
        def setUpClass(self):
            scprep.run.splatter.install(verbose=False)

        def test_splatter_default(self):
            sim = scprep.run.SplatSimulate(batch_cells=10, n_genes=200, verbose=0)
            assert sim["counts"].shape == (10, 200)
            assert np.all(sim["batch"] == "Batch1")
            assert sim["batch_cell_means"].shape == (10, 200)
            assert sim["base_cell_means"].shape == (10, 200)
            assert sim["bcv"].shape == (10, 200)
            assert sim["cell_means"].shape == (10, 200)
            assert sim["true_counts"].shape == (10, 200)
            assert sim["dropout"] is None
            assert sim["step"].shape == (10,)
            assert sim["group"].shape == (10,)
            assert sim["exp_lib_size"].shape == (10,)
            assert sim["base_gene_mean"].shape == (200,)
            assert sim["outlier_factor"].shape == (200,)
            assert sum(["batch_fac" in k for k in sim.keys()]) == 0
            assert sum(["de_fac" in k for k in sim.keys()]) == 1
            assert sim["de_fac_1"].shape == (200,)
            assert sum(["sigma_fac" in k for k in sim.keys()]) == 1
            assert sim["sigma_fac_1"].shape == (200,)

        def test_splatter_batch(self):
            sim = scprep.run.SplatSimulate(batch_cells=[5, 5], n_genes=200, verbose=0)
            assert sim["counts"].shape == (10, 200)
            assert np.all(sim["batch"][:5] == "Batch1")
            assert np.all(sim["batch"][5:] == "Batch2")
            assert sim["batch_cell_means"].shape == (10, 200)
            assert sim["base_cell_means"].shape == (10, 200)
            assert sim["bcv"].shape == (10, 200)
            assert sim["cell_means"].shape == (10, 200)
            assert sim["true_counts"].shape == (10, 200)
            assert sim["dropout"] is None
            assert sim["step"].shape == (10,)
            assert sim["group"].shape == (10,)
            assert sim["exp_lib_size"].shape == (10,)
            assert sim["base_gene_mean"].shape == (200,)
            assert sim["outlier_factor"].shape == (200,)
            assert sum(["batch_fac" in k for k in sim.keys()]) == 2
            assert sim["batch_fac_1"].shape == (200,)
            assert sim["batch_fac_2"].shape == (200,)
            assert sum(["de_fac" in k for k in sim.keys()]) == 1
            assert sim["de_fac_1"].shape == (200,)
            assert sum(["sigma_fac" in k for k in sim.keys()]) == 1
            assert sim["sigma_fac_1"].shape == (200,)

        def test_splatter_groups(self):
            sim = scprep.run.SplatSimulate(
                method="groups",
                batch_cells=10,
                group_prob=[0.5, 0.5],
                n_genes=200,
                de_fac_loc=[0.1, 0.5],
                verbose=0,
            )
            assert sim["counts"].shape == (10, 200)
            assert np.all(sim["batch"] == "Batch1")
            assert sim["batch_cell_means"].shape == (10, 200)
            assert sim["base_cell_means"].shape == (10, 200)
            assert sim["bcv"].shape == (10, 200)
            assert sim["cell_means"].shape == (10, 200)
            assert sim["true_counts"].shape == (10, 200)
            assert sim["dropout"] is None
            assert sim["step"] is None
            assert sim["group"].shape == (10,)
            assert sim["exp_lib_size"].shape == (10,)
            assert sim["base_gene_mean"].shape == (200,)
            assert sim["outlier_factor"].shape == (200,)
            assert sum(["batch_fac" in k for k in sim.keys()]) == 0
            assert sum(["de_fac" in k for k in sim.keys()]) == 2
            assert sim["de_fac_1"].shape == (200,)
            assert sim["de_fac_2"].shape == (200,)
            assert sum(["sigma_fac" in k for k in sim.keys()]) == 0

        def test_splatter_paths(self):
            sim = scprep.run.SplatSimulate(
                method="paths",
                batch_cells=10,
                n_genes=200,
                group_prob=[0.5, 0.5],
                path_from=[0, 0],
                path_length=[100, 200],
                path_skew=[0.4, 0.6],
                de_fac_loc=[0.1, 0.5],
                verbose=0,
            )
            assert sim["counts"].shape == (10, 200)
            assert np.all(sim["batch"] == "Batch1")
            assert sim["batch_cell_means"].shape == (10, 200)
            assert sim["base_cell_means"].shape == (10, 200)
            assert sim["bcv"].shape == (10, 200)
            assert sim["cell_means"].shape == (10, 200)
            assert sim["true_counts"].shape == (10, 200)
            assert sim["dropout"] is None
            assert sim["step"].shape == (10,)
            assert sim["group"].shape == (10,)
            assert sim["exp_lib_size"].shape == (10,)
            assert sim["base_gene_mean"].shape == (200,)
            assert sim["outlier_factor"].shape == (200,)
            assert sum(["batch_fac" in k for k in sim.keys()]) == 0
            assert sum(["de_fac" in k for k in sim.keys()]) == 2
            assert sim["de_fac_1"].shape == (200,)
            assert sim["de_fac_2"].shape == (200,)
            assert sum(["sigma_fac" in k for k in sim.keys()]) == 2
            assert sim["sigma_fac_1"].shape == (200,)
            assert sim["sigma_fac_2"].shape == (200,)

        def test_splatter_dropout(self):
            sim = scprep.run.SplatSimulate(
                batch_cells=10, n_genes=200, dropout_type="experiment", verbose=0
            )
            assert sim["counts"].shape == (10, 200)
            assert np.all(sim["batch"] == "Batch1")
            assert sim["batch_cell_means"].shape == (10, 200)
            assert sim["base_cell_means"].shape == (10, 200)
            assert sim["bcv"].shape == (10, 200)
            assert sim["cell_means"].shape == (10, 200)
            assert sim["true_counts"].shape == (10, 200)
            assert sim["dropout"].shape == (10, 200)
            assert sim["step"].shape == (10,)
            assert sim["group"].shape == (10,)
            assert sim["exp_lib_size"].shape == (10,)
            assert sim["base_gene_mean"].shape == (200,)
            assert sim["outlier_factor"].shape == (200,)
            assert sum(["batch_fac" in k for k in sim.keys()]) == 0
            assert sum(["de_fac" in k for k in sim.keys()]) == 1
            assert sim["de_fac_1"].shape == (200,)
            assert sum(["sigma_fac" in k for k in sim.keys()]) == 1
            assert sim["sigma_fac_1"].shape == (200,)

        def test_splatter_dropout_binomial(self):
            sim = scprep.run.SplatSimulate(
                batch_cells=10,
                n_genes=200,
                dropout_type="binomial",
                dropout_prob=0.5,
                verbose=False,
            )
            assert sim["counts"].shape == (10, 200)
            assert np.all(sim["batch"] == "Batch1")
            assert sim["batch_cell_means"].shape == (10, 200)
            assert sim["base_cell_means"].shape == (10, 200)
            assert sim["bcv"].shape == (10, 200)
            assert sim["cell_means"].shape == (10, 200)
            assert sim["true_counts"].shape == (10, 200)
            dropout_proportion = np.mean(
                sim["counts"][np.where(sim["true_counts"] > 0)]
                / sim["true_counts"][np.where(sim["true_counts"] > 0)]
            )
            assert dropout_proportion < 0.55
            assert dropout_proportion > 0.45
            assert sim["dropout"] is None
            assert sim["step"].shape == (10,)
            assert sim["group"].shape == (10,)
            assert sim["exp_lib_size"].shape == (10,)
            assert sim["base_gene_mean"].shape == (200,)
            assert sim["outlier_factor"].shape == (200,)
            assert sum(["batch_fac" in k for k in sim.keys()]) == 0
            assert sum(["de_fac" in k for k in sim.keys()]) == 1
            assert sim["de_fac_1"].shape == (200,)
            assert sum(["sigma_fac" in k for k in sim.keys()]) == 1
            assert sim["sigma_fac_1"].shape == (200,)

        def test_splatter_warning(self):
            assert (
                rpy2.rinterface_lib.callbacks.consolewrite_warnerror is builtin_warning
            )
            scprep.run.r_function._ConsoleWarning.set_debug()
            assert (
                rpy2.rinterface_lib.callbacks.consolewrite_warnerror
                is scprep.run.r_function._ConsoleWarning.debug
            )
            scprep.run.r_function._ConsoleWarning.set_warning()
            assert (
                rpy2.rinterface_lib.callbacks.consolewrite_warnerror
                is scprep.run.r_function._ConsoleWarning.warning
            )
            scprep.run.r_function._ConsoleWarning.set_builtin()
            assert (
                rpy2.rinterface_lib.callbacks.consolewrite_warnerror is builtin_warning
            )

    class TestSlingshot(unittest.TestCase):
        @classmethod
        def setUpClass(self):
            scprep.run.slingshot.install(verbose=False)
            self.X = data.load_10X()
            self.X_pca = scprep.reduce.pca(self.X)
            self.clusters = sklearn.cluster.KMeans(6).fit_predict(self.X_pca)

        def test_slingshot(self):
            slingshot = scprep.run.Slingshot(
                self.X_pca[:, :2], self.clusters, verbose=False
            )
            pseudotime, branch, curves = (
                slingshot["pseudotime"],
                slingshot["branch"],
                slingshot["curves"],
            )
            assert pseudotime.shape[0] == self.X_pca.shape[0]
            assert pseudotime.shape[1] == curves.shape[0]
            assert branch.shape[0] == self.X_pca.shape[0]
            current_pseudotime = -1
            for i in np.unique(branch):
                branch_membership = np.isnan(pseudotime[branch == i])
                assert np.all(branch_membership == branch_membership[0])
                new_pseudotime = np.nanmean(pseudotime[branch == i])
                assert new_pseudotime > current_pseudotime
                current_pseudotime = new_pseudotime
            assert curves.shape[1] == self.X_pca.shape[0]
            assert curves.shape[2] == 2
            assert np.all(np.any(~np.isnan(pseudotime), axis=1))

        def test_slingshot_pandas(self):
            slingshot = scprep.run.Slingshot(
                pd.DataFrame(self.X_pca[:, :2], index=self.X.index),
                self.clusters,
                verbose=False,
            )
            pseudotime, branch, curves = (
                slingshot["pseudotime"],
                slingshot["branch"],
                slingshot["curves"],
            )
            assert np.all(pseudotime.index == self.X.index)
            assert np.all(branch.index == self.X.index)
            assert branch.name == "branch"
            assert pseudotime.shape[0] == self.X_pca.shape[0]
            assert pseudotime.shape[1] == curves.shape[0]
            assert branch.shape[0] == self.X_pca.shape[0]
            current_pseudotime = -1
            for i in np.unique(branch):
                branch_membership = np.isnan(pseudotime.loc[branch == i])
                assert np.all(branch_membership == branch_membership.iloc[0])
                new_pseudotime = np.nanmean(np.nanmean(pseudotime.loc[branch == i]))
                assert new_pseudotime > current_pseudotime
                current_pseudotime = new_pseudotime
            assert curves.shape[1] == self.X_pca.shape[0]
            assert curves.shape[2] == 2
            assert np.all(np.any(~np.isnan(pseudotime), axis=1))

        def test_slingshot_distance(self):
            utils.assert_raises_message(
                NotImplementedError,
                "distance argument not currently implemented",
                scprep.run.Slingshot,
                self.X_pca,
                self.clusters,
                distance=lambda X, Y: np.sum(X - Y),
            )

        def test_slingshot_optional_args(self):
            slingshot = scprep.run.Slingshot(
                self.X_pca[:, :2],
                self.clusters,
                start_cluster=4,
                omega=0.1,
                verbose=False,
            )
            pseudotime, branch, curves = (
                slingshot["pseudotime"],
                slingshot["branch"],
                slingshot["curves"],
            )
            assert pseudotime.shape[0] == self.X_pca.shape[0]
            assert pseudotime.shape[1] == curves.shape[0]
            assert branch.shape[0] == self.X_pca.shape[0]
            current_pseudotime = -1
            for i in np.unique(branch):
                branch_membership = np.isnan(pseudotime[branch == i])
                assert np.all(branch_membership == branch_membership[0])
                if np.all(np.isnan(pseudotime[branch == i])):
                    assert i == -1
                else:
                    new_pseudotime = np.nanmean(pseudotime[branch == i])
                    assert new_pseudotime > current_pseudotime
                    current_pseudotime = new_pseudotime
            assert curves.shape[1] == self.X_pca.shape[0]
            assert curves.shape[2] == 2
            slingshot = scprep.run.Slingshot(
                self.X_pca[:, :2], self.clusters, end_cluster=0, verbose=False
            )
            pseudotime, branch, curves = (
                slingshot["pseudotime"],
                slingshot["branch"],
                slingshot["curves"],
            )
            assert pseudotime.shape[0] == self.X_pca.shape[0]
            assert pseudotime.shape[1] == curves.shape[0]
            assert branch.shape[0] == self.X_pca.shape[0]
            current_pseudotime = -1
            for i in np.unique(branch):
                branch_membership = np.isnan(pseudotime[branch == i])
                assert np.all(branch_membership == branch_membership[0])
                new_pseudotime = np.nanmean(pseudotime[branch == i])
                assert new_pseudotime > current_pseudotime
                current_pseudotime = new_pseudotime
            assert curves.shape[1] == self.X_pca.shape[0]
            assert curves.shape[2] == 2
            assert np.all(np.any(~np.isnan(pseudotime), axis=1))

        def test_slingshot_errors(self):
            utils.assert_warns_message(
                UserWarning,
                "Expected data to be low-dimensional. " "Got data.shape[1] = 4",
                scprep.run.Slingshot,
                self.X_pca[:, :4],
                self.clusters,
                verbose=False,
            )
            utils.assert_raises_message(
                ValueError,
                "Expected len(cluster_labels) ({}) to equal "
                "data.shape[0] ({})".format(self.X.shape[0] // 2, self.X.shape[0]),
                scprep.run.Slingshot,
                self.X_pca[:, :2],
                self.clusters[: self.X.shape[0] // 2],
                verbose=False,
            )
