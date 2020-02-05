from tools import utils, matrix, data
import scprep
import pandas as pd
import numpy as np

from scipy import sparse
from functools import partial
import unittest


class Test10X(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.X_dense = data.load_10X(sparse=False)
        self.X_sparse = data.load_10X(sparse=True)
        self.libsize = scprep.measure.library_size(self.X_dense.to_numpy())

    def test_filter_empty_cells(self):
        X_filtered = scprep.filter.filter_empty_cells(self.X_dense)
        assert X_filtered.shape[1] == self.X_dense.shape[1]
        assert not np.any(X_filtered.sum(1) == 0)
        matrix.test_all_matrix_types(
            self.X_dense,
            utils.assert_transform_equals,
            Y=X_filtered,
            transform=scprep.filter.filter_empty_cells,
        )
        sample_labels = np.arange(self.X_dense.shape[0])
        sample_labels_filt = sample_labels[self.X_dense.sum(1) > 0]
        X_filtered_2, sample_labels = scprep.filter.filter_empty_cells(
            self.X_dense, sample_labels
        )
        assert X_filtered_2.shape[0] == len(sample_labels)
        assert np.all(sample_labels == sample_labels_filt)
        assert np.all(X_filtered_2 == X_filtered)

    def test_filter_duplicates(self):
        unique_idx = np.sort(np.unique(self.X_dense, axis=0, return_index=True)[1])
        X_filtered = np.array(self.X_dense)[unique_idx]
        matrix.test_all_matrix_types(
            self.X_dense,
            utils.assert_transform_equals,
            Y=X_filtered,
            transform=scprep.filter.filter_duplicates,
        )
        sample_labels = np.arange(self.X_dense.shape[0])
        sample_labels_filt = sample_labels[unique_idx]
        X_filtered_2, sample_labels = scprep.filter.filter_duplicates(
            self.X_dense, sample_labels
        )
        assert X_filtered_2.shape[0] == len(sample_labels)
        assert np.all(sample_labels == sample_labels_filt)
        assert np.all(X_filtered_2 == X_filtered)

    def test_filter_empty_cells_sample_label(self):
        sample_labels = np.arange(self.X_dense.shape[0])
        sample_labels_filt = sample_labels[self.X_dense.sum(1) > 0]
        X_filtered, sample_labels = scprep.filter.filter_empty_cells(
            self.X_dense, sample_labels
        )
        assert X_filtered.shape[0] == len(sample_labels)
        assert np.all(sample_labels == sample_labels_filt)

    def test_filter_empty_cells_sparse(self):
        X_filtered = scprep.filter.filter_empty_cells(self.X_sparse)
        assert X_filtered.shape[1] == self.X_sparse.shape[1]
        assert not np.any(X_filtered.sum(1) == 0)
        matrix.test_all_matrix_types(
            self.X_sparse,
            utils.assert_transform_equals,
            Y=X_filtered,
            transform=scprep.filter.filter_empty_cells,
        )

    def test_filter_empty_genes(self):
        X_filtered = scprep.filter.filter_empty_genes(self.X_dense)
        assert X_filtered.shape[0] == self.X_dense.shape[0]
        assert not np.any(X_filtered.sum(0) == 0)
        matrix.test_all_matrix_types(
            self.X_dense,
            utils.assert_transform_equals,
            Y=X_filtered,
            transform=scprep.filter.filter_empty_genes,
        )

    def test_filter_empty_genes_sparse(self):
        X_filtered = scprep.filter.filter_empty_genes(self.X_sparse)
        assert X_filtered.shape[0] == self.X_sparse.shape[0]
        assert not np.any(X_filtered.sum(0) == 0)
        matrix.test_all_matrix_types(
            self.X_sparse,
            utils.assert_transform_equals,
            Y=X_filtered,
            transform=scprep.filter.filter_empty_genes,
        )

    def test_filter_rare_genes(self):
        X_filtered = scprep.filter.filter_rare_genes(self.X_dense)
        assert X_filtered.shape[0] == self.X_dense.shape[0]
        assert not np.any(X_filtered.sum(0) < 5)
        matrix.test_all_matrix_types(
            self.X_dense,
            utils.assert_transform_equals,
            Y=X_filtered,
            transform=scprep.filter.filter_rare_genes,
        )

    def test_library_size_filter(self):
        X_filtered = scprep.filter.filter_library_size(self.X_sparse, cutoff=100)
        assert X_filtered.shape[1] == self.X_sparse.shape[1]
        assert not np.any(X_filtered.sum(1) <= 100)
        X_filtered, libsize = scprep.filter.filter_library_size(
            self.X_sparse, cutoff=100, return_library_size=True
        )
        assert np.all(scprep.measure.library_size(X_filtered) == libsize)
        matrix.test_all_matrix_types(
            self.X_sparse,
            utils.assert_transform_equals,
            Y=X_filtered,
            transform=partial(scprep.filter.filter_library_size, cutoff=100),
        )
        X_filtered = scprep.filter.filter_library_size(
            self.X_sparse, cutoff=100, keep_cells="below"
        )
        assert X_filtered.shape[1] == self.X_sparse.shape[1]
        assert not np.any(X_filtered.sum(1) >= 100)

    def test_library_size_filter_below(self):
        X_filtered = scprep.filter.filter_library_size(
            self.X_sparse, cutoff=100, keep_cells="below"
        )
        assert X_filtered.shape[1] == self.X_sparse.shape[1]
        assert not np.any(X_filtered.sum(1) >= 100)

    def test_library_size_filter_between(self):
        X_filtered = scprep.filter.filter_library_size(self.X_sparse, cutoff=(50, 100))
        assert X_filtered.shape[1] == self.X_sparse.shape[1]
        assert not np.any(X_filtered.sum(1) >= 100)
        assert not np.any(X_filtered.sum(1) <= 50)
        X_filtered = scprep.filter.filter_library_size(
            self.X_sparse, percentile=(20, 80)
        )
        assert X_filtered.shape[1] == self.X_sparse.shape[1]
        assert not np.any(X_filtered.sum(1) >= np.percentile(self.libsize, 80))
        assert not np.any(X_filtered.sum(1) <= np.percentile(self.libsize, 20))

    def test_library_size_filter_error(self):
        utils.assert_raises_message(
            ValueError,
            "Expected `keep_cells` in ['above', 'below', 'between']. Got invalid",
            scprep.filter.filter_library_size,
            self.X_sparse,
            cutoff=100,
            keep_cells="invalid",
        )
        utils.assert_raises_message(
            ValueError,
            "Expected cutoff of length 2 with keep_cells='between'. Got 100",
            scprep.filter.filter_library_size,
            self.X_sparse,
            cutoff=100,
            keep_cells="between",
        )
        utils.assert_raises_message(
            ValueError,
            "Expected a single cutoff with keep_cells='above'. Got (50, 100)",
            scprep.filter.filter_library_size,
            self.X_sparse,
            cutoff=(50, 100),
            keep_cells="above",
        )
        utils.assert_raises_message(
            ValueError,
            "Expected a single cutoff with keep_cells='below'. Got (50, 100)",
            scprep.filter.filter_library_size,
            self.X_sparse,
            cutoff=(50, 100),
            keep_cells="below",
        )

    def test_library_size_filter_sample_label(self):
        sample_labels = pd.Series(
            np.random.choice([0, 1], self.X_dense.shape[0]), index=self.X_dense.index
        )
        sample_labels_filt = sample_labels.loc[self.X_dense.sum(1) > 100]
        X_filtered, sample_labels_filt2 = scprep.filter.filter_library_size(
            self.X_dense, sample_labels, cutoff=100
        )
        assert X_filtered.shape[0] == len(sample_labels_filt2)
        assert np.all(np.all(sample_labels_filt2 == sample_labels_filt))

    def test_gene_expression_filter_below(self):
        genes = np.arange(10)
        X_filtered = scprep.filter.filter_gene_set_expression(
            self.X_sparse, genes=genes, percentile=90, library_size_normalize=False
        )
        gene_cols = np.array(self.X_sparse.columns)[genes]
        assert X_filtered.shape[1] == self.X_sparse.shape[1]
        assert np.max(np.sum(self.X_sparse[gene_cols], axis=1)) > np.max(
            np.sum(X_filtered[gene_cols], axis=1)
        )
        matrix.test_all_matrix_types(
            self.X_sparse,
            utils.assert_transform_equals,
            Y=X_filtered,
            transform=partial(
                scprep.filter.filter_gene_set_expression,
                genes=genes,
                percentile=90,
                keep_cells="below",
                library_size_normalize=False,
            ),
        )

    def test_gene_expression_filter_above(self):
        genes = np.arange(10)
        gene_cols = np.array(self.X_sparse.columns)[genes]
        X_filtered = scprep.filter.filter_gene_set_expression(
            self.X_sparse,
            genes=genes,
            percentile=10,
            keep_cells="above",
            library_size_normalize=False,
        )
        assert X_filtered.shape[1] == self.X_sparse.shape[1]
        assert np.min(np.sum(self.X_sparse[gene_cols], axis=1)) < np.min(
            np.sum(X_filtered[gene_cols], axis=1)
        )
        matrix.test_all_matrix_types(
            self.X_sparse,
            utils.assert_transform_equals,
            Y=X_filtered,
            transform=partial(
                scprep.filter.filter_gene_set_expression,
                genes=genes,
                percentile=10,
                keep_cells="above",
                library_size_normalize=False,
            ),
        )

    def test_gene_expression_libsize(self):
        genes = np.arange(10)
        X_filtered = scprep.filter.filter_gene_set_expression(
            self.X_sparse,
            genes=genes,
            percentile=10,
            keep_cells="above",
            library_size_normalize=True,
        )
        X_libsize = scprep.normalize.library_size_normalize(self.X_sparse)
        Y = scprep.filter.filter_gene_set_expression(
            X_libsize,
            genes=genes,
            percentile=10,
            keep_cells="above",
            library_size_normalize=False,
        )
        assert X_filtered.shape == Y.shape
        assert np.all(X_filtered.index == Y.index)

    def test_gene_expression_filter_sample_label(self):
        genes = np.arange(10)
        sample_labels = pd.DataFrame(
            np.arange(self.X_dense.shape[0]), index=self.X_dense.index
        )
        X_filtered, sample_labels = scprep.filter.filter_gene_set_expression(
            self.X_dense, sample_labels, genes=genes, percentile=90
        )
        assert X_filtered.shape[0] == len(sample_labels)

    def test_gene_expression_filter_warning(self):
        genes = np.arange(10)
        no_genes = "not_a_gene"
        utils.assert_warns_message(
            UserWarning,
            "`percentile` expects values between 0 and 100."
            "Got 0.9. Did you mean 90.0?",
            scprep.filter.filter_gene_set_expression,
            self.X_sparse,
            genes=genes,
            percentile=0.90,
            keep_cells="below",
        )
        utils.assert_raises_message(
            ValueError,
            "Only one of `cutoff` and `percentile` should be given.",
            scprep.filter.filter_gene_set_expression,
            self.X_sparse,
            genes=genes,
            percentile=0.90,
            cutoff=50,
        )
        utils.assert_raises_message(
            ValueError,
            "Expected `keep_cells` in ['above', 'below', 'between']. " "Got neither",
            scprep.filter.filter_gene_set_expression,
            self.X_sparse,
            genes=genes,
            percentile=90.0,
            keep_cells="neither",
        )
        utils.assert_warns_message(
            UserWarning,
            "`percentile` expects values between 0 and 100."
            "Got 0.9. Did you mean 90.0?",
            scprep.filter.filter_gene_set_expression,
            self.X_sparse,
            genes=genes,
            percentile=0.90,
            keep_cells="below",
        )
        utils.assert_raises_message(
            ValueError,
            "One of either `cutoff` or `percentile` must be given.",
            scprep.filter.filter_gene_set_expression,
            self.X_sparse,
            genes=genes,
            cutoff=None,
            percentile=None,
        )
        utils.assert_raises_message(
            KeyError,
            "not_a_gene",
            scprep.filter.filter_gene_set_expression,
            self.X_sparse,
            genes=no_genes,
            percentile=90.0,
            keep_cells="below",
        )

    def filter_series(self):
        libsize = scprep.measure.library_size(self.X_sparse)
        libsize_filt = scprep.filter.filter_values(libsize, libsize, cutoff=100)
        assert np.all(libsize_filt > 100)

    def test_deprecated_remove(self):
        utils.assert_warns_message(
            DeprecationWarning,
            "`scprep.filter.remove_empty_genes` is deprecated. Use "
            "`scprep.filter.filter_empty_genes` instead.",
            scprep.filter.remove_empty_genes,
            self.X_dense,
        )
        utils.assert_warns_message(
            DeprecationWarning,
            "`scprep.filter.remove_rare_genes` is deprecated. Use "
            "`scprep.filter.filter_rare_genes` instead.",
            scprep.filter.remove_rare_genes,
            self.X_dense,
        )
        utils.assert_warns_message(
            DeprecationWarning,
            "`scprep.filter.remove_empty_cells` is deprecated. Use "
            "`scprep.filter.filter_empty_cells` instead.",
            scprep.filter.remove_empty_cells,
            self.X_dense,
        )
        utils.assert_warns_message(
            DeprecationWarning,
            "`scprep.filter.remove_duplicates` is deprecated. Use "
            "`scprep.filter.filter_duplicates` instead.",
            scprep.filter.remove_duplicates,
            self.X_dense,
        )

    def test_deprecated_sample_labels(self):
        sample_labels = np.arange(self.X_dense.shape[0])
        utils.assert_warns_message(
            DeprecationWarning,
            "`sample_labels` is deprecated. "
            "Passing `sample_labels` as `extra_data`.",
            scprep.filter.filter_empty_cells,
            self.X_dense,
            sample_labels=sample_labels,
        )
        utils.assert_warns_message(
            DeprecationWarning,
            "`sample_labels` is deprecated. "
            "Passing `sample_labels` as `extra_data`.",
            scprep.filter.filter_duplicates,
            self.X_dense,
            sample_labels=sample_labels,
        )
        utils.assert_warns_message(
            DeprecationWarning,
            "`sample_labels` is deprecated. "
            "Passing `sample_labels` as `extra_data`.",
            scprep.filter.filter_library_size,
            self.X_dense,
            cutoff=10,
            sample_labels=sample_labels,
        )
        utils.assert_warns_message(
            DeprecationWarning,
            "`filter_per_sample` is deprecated. " "Filtering as a single sample.",
            scprep.filter.filter_library_size,
            self.X_dense,
            cutoff=10,
            filter_per_sample=True,
        )


def test_large_sparse_dataframe_library_size():
    if matrix._pandas_0:
        matrix._ignore_pandas_sparse_warning()
        X = pd.SparseDataFrame(
            sparse.coo_matrix((10 ** 7, 2 * 10 ** 4)), default_fill_value=0.0
        )
        cell_sums = scprep.measure.library_size(X)
        assert cell_sums.shape[0] == X.shape[0]
        matrix._reset_warnings()
    X = matrix.SparseDataFrame(
        sparse.coo_matrix((10 ** 7, 2 * 10 ** 4)), default_fill_value=0.0
    )
    cell_sums = scprep.measure.library_size(X)
    assert cell_sums.shape[0] == X.shape[0]
