from tools import data, matrix, utils
import scprep

import numpy as np
import pandas as pd
import unittest
from scipy import sparse


class Test10X(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.X = data.load_10X(sparse=False)
        self.X_sparse = data.load_10X(sparse=True)
        self.libsize = scprep.measure.library_size(self.X)

    def test_get_gene_set_starts_with(self):
        gene_idx = np.argwhere([g.startswith("D") for g in self.X.columns]).flatten()
        gene_names = self.X.columns[gene_idx]
        assert np.all(scprep.select.get_gene_set(self.X, starts_with="D") == gene_names)
        assert np.all(scprep.select.get_gene_set(self.X, regex="^D") == gene_names)
        assert np.all(
            scprep.select.get_gene_set(self.X.columns, regex="^D") == gene_names
        )

    def test_get_gene_set_ends_with(self):
        gene_idx = np.argwhere([g.endswith("8") for g in self.X.columns]).flatten()
        gene_names = self.X.columns[gene_idx]
        assert np.all(scprep.select.get_gene_set(self.X, ends_with="8") == gene_names)
        assert np.all(scprep.select.get_gene_set(self.X, regex="8$") == gene_names)

    def test_get_gene_set_ndarray(self):
        utils.assert_raises_message(
            TypeError,
            "data must be a list of gene names or a pandas " "DataFrame. Got ndarray",
            scprep.select.get_gene_set,
            data=self.X.to_numpy(),
            regex="8$",
        )

    def test_get_gene_set_no_condition(self):
        utils.assert_warns_message(
            UserWarning,
            "No selection conditions provided. Returning all genes.",
            scprep.select.get_gene_set,
            self.X,
        )

    def test_get_cell_set_starts_with(self):
        cell_idx = np.argwhere([g.startswith("A") for g in self.X.index]).flatten()
        cell_names = self.X.index[cell_idx]
        assert np.all(scprep.select.get_cell_set(self.X, starts_with="A") == cell_names)
        assert np.all(scprep.select.get_cell_set(self.X, regex="^A") == cell_names)
        assert np.all(
            scprep.select.get_cell_set(self.X.index, regex="^A") == cell_names
        )

    def test_get_cell_set_ends_with(self):
        cell_idx = np.argwhere([g.endswith("G-1") for g in self.X.index]).flatten()
        cell_names = self.X.index[cell_idx]
        assert np.all(scprep.select.get_cell_set(self.X, ends_with="G-1") == cell_names)
        assert np.all(scprep.select.get_cell_set(self.X, regex="G\\-1$") == cell_names)

    def test_get_cell_set_ndarray(self):
        utils.assert_raises_message(
            TypeError,
            "data must be a list of cell names or a pandas " "DataFrame. Got ndarray",
            scprep.select.get_cell_set,
            data=self.X.to_numpy(),
            regex="G\\-1$",
        )

    def test_get_cell_set_no_condition(self):
        utils.assert_warns_message(
            UserWarning,
            "No selection conditions provided. Returning all cells.",
            scprep.select.get_cell_set,
            self.X,
        )

    def test_select_rows_boolean_index(self):
        matrix.test_all_matrix_types(
            self.X,
            scprep.select.select_rows,
            idx=np.random.choice([True, False], [self.X.shape[0]]),
        )

    def test_select_rows_integer_array_index(self):
        matrix.test_all_matrix_types(
            self.X,
            scprep.select.select_rows,
            idx=np.random.choice(self.X.shape[0], self.X.shape[0] // 2),
        )

    def test_select_rows_integer_list_index(self):
        matrix.test_all_matrix_types(
            self.X,
            scprep.select.select_rows,
            idx=np.random.choice(self.X.shape[0], self.X.shape[0] // 2).tolist(),
        )

    def test_select_rows_integer_index(self):
        matrix.test_all_matrix_types(
            self.X, scprep.select.select_rows, idx=np.random.choice(self.X.shape[0])
        )

    def test_select_rows_string_array_index(self):
        matrix.test_pandas_matrix_types(
            self.X,
            scprep.select.select_rows,
            idx=np.random.choice(self.X.index.to_numpy(), self.X.shape[0] // 2),
        )

    def test_select_rows_pandas_index_index(self):
        matrix.test_pandas_matrix_types(
            self.X,
            scprep.select.select_rows,
            idx=self.X.index[np.random.choice([True, False], [self.X.shape[0]])],
        )

    def test_select_rows_series_index(self):
        matrix.test_pandas_matrix_types(
            self.X,
            scprep.select.select_rows,
            idx=pd.Series(
                self.X.index[np.random.choice([True, False], [self.X.shape[0]])]
            ),
        )

    def test_select_rows_dataframe_index(self):
        matrix.test_all_matrix_types(
            self.X,
            scprep.select.select_rows,
            idx=pd.DataFrame(
                np.random.choice([True, False], [self.X.shape[0], 1]),
                index=self.X.index,
            ),
        )

    def test_select_rows_series_data_boolean_index(self):
        scprep.select.select_rows(
            self.X,
            self.X.iloc[:, 0],
            idx=np.random.choice([True, False], [self.X.shape[0]]),
        )

    def test_select_rows_sparse_series_data_boolean_index(self):
        scprep.select.select_rows(
            self.X,
            self.X_sparse.iloc[:, 0],
            idx=np.random.choice([True, False], [self.X.shape[0]]),
        )

    def test_select_rows_series_data_integer_index(self):
        scprep.select.select_rows(
            self.X,
            self.X.iloc[:, 0],
            idx=np.random.choice(self.X.shape[1], self.X.shape[0] // 2),
        )

    def test_select_rows_sparse_series_data_integer_index(self):
        scprep.select.select_rows(
            self.X,
            self.X_sparse.iloc[:, 0],
            idx=np.random.choice(self.X.shape[1], self.X.shape[0] // 2),
        )

    def test_select_rows_1d_array_data(self):
        scprep.select.select_rows(
            self.X,
            self.X.to_numpy()[:, 0],
            idx=np.random.choice([True, False], [self.X.shape[0]]),
        )

    def test_select_rows_list_data(self):
        scprep.select.select_rows(
            self.X,
            self.X.to_numpy()[:, 0].tolist(),
            idx=np.random.choice([True, False], [self.X.shape[1]]),
        )

    def test_select_rows_get_cell_set(self):
        matrix.test_pandas_matrix_types(
            self.X, scprep.select.select_rows, self.X.iloc[:, 0], starts_with="A"
        )

    def test_select_rows_zero_rows(self):
        utils.assert_warns_message(
            UserWarning,
            "Selecting 0 rows",
            scprep.select.select_rows,
            self.X,
            idx=(self.X.sum(axis=1) < 0),
        )

    def test_select_rows_no_condition(self):
        utils.assert_warns_message(
            UserWarning,
            "No selection conditions provided. Returning all rows.",
            scprep.select.select_rows,
            self.X,
        )

    def test_select_cols_boolean_index(self):
        matrix.test_all_matrix_types(
            self.X,
            scprep.select.select_cols,
            idx=np.random.choice([True, False], [self.X.shape[1]]),
        )

    def test_select_cols_integer_array_index(self):
        matrix.test_all_matrix_types(
            self.X,
            scprep.select.select_cols,
            idx=np.random.choice(self.X.shape[1], self.X.shape[1] // 2),
        )

    def test_select_cols_integer_list_index(self):
        matrix.test_all_matrix_types(
            self.X,
            scprep.select.select_cols,
            idx=np.random.choice(self.X.shape[1], self.X.shape[1] // 2).tolist(),
        )

    def test_select_cols_integer_index(self):
        matrix.test_all_matrix_types(
            self.X, scprep.select.select_cols, idx=np.random.choice(self.X.shape[1])
        )

    def test_select_cols_string_array_index(self):
        matrix.test_pandas_matrix_types(
            self.X,
            scprep.select.select_cols,
            idx=np.random.choice(self.X.columns.to_numpy(), self.X.shape[1] // 2),
        )

    def test_select_cols_pandas_index_index(self):
        matrix.test_pandas_matrix_types(
            self.X,
            scprep.select.select_cols,
            idx=self.X.columns[np.random.choice([True, False], [self.X.shape[1]])],
        )

    def test_select_cols_series_index(self):
        matrix.test_pandas_matrix_types(
            self.X,
            scprep.select.select_cols,
            idx=pd.Series(
                self.X.columns[np.random.choice([True, False], [self.X.shape[1]])]
            ),
        )

    def test_select_cols_dataframe_index(self):
        matrix.test_all_matrix_types(
            self.X,
            scprep.select.select_cols,
            idx=pd.DataFrame(
                np.random.choice([True, False], [1, self.X.shape[1]]),
                index=[1],
                columns=self.X.columns,
            ),
        )

    def test_select_cols_sparse_index(self):
        matrix.test_all_matrix_types(
            self.X,
            scprep.select.select_cols,
            idx=sparse.coo_matrix(
                np.random.choice([True, False], [1, self.X.shape[1]])
            ),
        )
        matrix.test_all_matrix_types(
            self.X,
            scprep.select.select_cols,
            idx=sparse.coo_matrix(
                np.random.choice([True, False], [self.X.shape[1], 1])
            ),
        )

    def test_select_rows_sparse_index(self):
        matrix.test_all_matrix_types(
            self.X,
            scprep.select.select_rows,
            idx=sparse.coo_matrix(
                np.random.choice([True, False], [1, self.X.shape[0]])
            ),
        )
        matrix.test_all_matrix_types(
            self.X,
            scprep.select.select_rows,
            idx=sparse.coo_matrix(
                np.random.choice([True, False], [self.X.shape[0], 1])
            ),
        )

    def test_select_cols_series_data_boolean_index(self):
        scprep.select.select_cols(
            self.X,
            self.X.iloc[0, :],
            idx=np.random.choice([True, False], [self.X.shape[1]]),
        )

    def test_select_cols_sparse_series_data_boolean_index(self):
        scprep.select.select_cols(
            self.X,
            self.X_sparse.iloc[0, :],
            idx=np.random.choice([True, False], [self.X.shape[1]]),
        )

    def test_select_cols_series_data_integer_index(self):
        scprep.select.select_cols(
            self.X,
            self.X.iloc[0, :],
            idx=np.random.choice(self.X.shape[1], self.X.shape[1] // 2),
        )

    def test_select_cols_sparse_series_data_integer_index(self):
        scprep.select.select_cols(
            self.X,
            self.X_sparse.iloc[0, :],
            idx=np.random.choice(self.X.shape[1], self.X.shape[1] // 2),
        )

    def test_select_cols_1d_array_data(self):
        scprep.select.select_cols(
            self.X,
            self.X.to_numpy()[0, :],
            idx=np.random.choice([True, False], [self.X.shape[1]]),
        )

    def test_select_cols_list_data(self):
        scprep.select.select_cols(
            self.X,
            self.X.to_numpy()[0, :].tolist(),
            idx=np.random.choice([True, False], [self.X.shape[1]]),
        )

    def test_select_cols_get_gene_set(self):
        matrix.test_pandas_matrix_types(
            self.X, scprep.select.select_cols, self.X.iloc[0, :], starts_with="D"
        )

    def test_select_cols_zero_columns(self):
        utils.assert_warns_message(
            UserWarning,
            "Selecting 0 columns",
            scprep.select.select_cols,
            self.X,
            idx=(self.X.sum(axis=0) < 0),
        )

    def test_select_cols_no_condition(self):
        utils.assert_warns_message(
            UserWarning,
            "No selection conditions provided. Returning all columns.",
            scprep.select.select_cols,
            self.X,
        )

    def test_select_rows_invalid_index(self):
        utils.assert_raises_message(
            KeyError,
            "'not_a_cell'",
            scprep.select.select_rows,
            self.X,
            idx="not_a_cell",
        )

    def test_select_cols_invalid_index(self):
        utils.assert_raises_message(
            KeyError,
            "'not_a_gene'",
            scprep.select.select_cols,
            self.X,
            idx="not_a_gene",
        )

    def test_select_rows_2d_dataframe_index(self):
        utils.assert_raises_message(
            ValueError,
            "Expected idx to be 1D. " "Got shape (2, {})".format(self.X.shape[0]),
            scprep.select.select_rows,
            self.X,
            idx=pd.DataFrame([self.X.index, self.X.index]),
        )

    def test_select_rows_2d_list_index(self):
        utils.assert_raises_message(
            ValueError,
            "Expected idx to be 1D. " "Got shape (2, {})".format(self.X.shape[0]),
            scprep.select.select_rows,
            self.X,
            idx=[self.X.index, self.X.index],
        )

    def test_select_cols_2d_dataframe_index(self):
        utils.assert_raises_message(
            ValueError,
            "Expected idx to be 1D. " "Got shape (2, {})".format(self.X.shape[1]),
            scprep.select.select_cols,
            self.X,
            idx=pd.DataFrame([self.X.columns, self.X.columns]),
        )

    def test_select_cols_2d_list_index(self):
        utils.assert_raises_message(
            ValueError,
            "Expected idx to be 1D. " "Got shape (2, {})".format(self.X.shape[1]),
            scprep.select.select_cols,
            self.X,
            idx=[self.X.columns, self.X.columns],
        )

    def test_select_cols_unequal_columns(self):
        utils.assert_raises_message(
            ValueError,
            "Expected `data` and `extra_data` to have the same number of "
            "columns. Got [100, 50]",
            scprep.select.select_cols,
            self.X,
            self.X.to_numpy()[:, :50],
        )

    def test_select_cols_return_series(self):
        assert isinstance(scprep.select.select_cols(self.X, idx=0), pd.Series)

    def test_select_cols_return_dataframe(self):
        assert isinstance(scprep.select.select_cols(self.X, idx=[0, 1]), pd.DataFrame)

    def test_select_rows_unequal_rows(self):
        utils.assert_raises_message(
            ValueError,
            "Expected `data` and `extra_data` to have the same number of "
            "rows. Got [100, 50]",
            scprep.select.select_rows,
            self.X,
            self.X.to_numpy()[:50, :],
        )

    def test_select_cols_conflicting_data(self):
        utils.assert_raises_message(
            ValueError,
            "Expected `data` and `extra_data` pandas inputs to have the same "
            "column names. Fix with "
            "`scprep.select.select_cols(*extra_data, idx=data.columns)`",
            scprep.select.select_cols,
            self.X,
            self.X.iloc[:, ::-1],
        )

    def test_select_rows_conflicting_data(self):
        utils.assert_raises_message(
            ValueError,
            "Expected `data` and `extra_data` pandas inputs to have the same "
            "index. Fix with "
            "`scprep.select.select_rows(*extra_data, idx=data.index)`",
            scprep.select.select_rows,
            self.X,
            self.X.iloc[::-1],
        )

    def test_select_cols_get_gene_set_ndarray_data(self):
        utils.assert_raises_message(
            ValueError,
            "Can only select based on column names with DataFrame input. "
            "Please set `idx` to select specific columns.",
            scprep.select.select_cols,
            self.X.to_numpy(),
            starts_with="A",
        )

    def test_select_rows_get_cell_set_ndarray_data(self):
        utils.assert_raises_message(
            ValueError,
            "Can only select based on row names with DataFrame input. "
            "Please set `idx` to select specific rows.",
            scprep.select.select_rows,
            self.X.to_numpy(),
            starts_with="A",
        )

    def test_select_rows_return_series(self):
        assert isinstance(scprep.select.select_rows(self.X, idx=0), pd.Series)

    def test_select_rows_return_dataframe(self):
        assert isinstance(scprep.select.select_rows(self.X, idx=[0, 1]), pd.DataFrame)

    def test_subsample(self):
        self.X = data.generate_positive_sparse_matrix(shape=(50, 100))
        Y = scprep.select.subsample(self.X, n=20, seed=42)
        matrix.test_all_matrix_types(
            self.X,
            utils.assert_transform_equals,
            Y=Y,
            transform=scprep.select.subsample,
            check=utils.assert_all_equal,
            n=20,
            seed=42,
        )

    def test_subsample_multiple(self):
        Y, libsize_sub = scprep.select.subsample(self.X, self.libsize, n=20, seed=42)

        def test_fun(X, **kwargs):
            libsize = scprep.measure.library_size(X)
            return scprep.select.subsample(X, libsize, **kwargs)[0]

        matrix.test_all_matrix_types(
            self.X,
            utils.assert_transform_equals,
            Y=Y,
            transform=test_fun,
            check=utils.assert_all_equal,
            n=20,
            seed=42,
        )

        def test_fun(X, **kwargs):
            libsize = scprep.measure.library_size(X)
            return scprep.select.subsample(X, libsize, **kwargs)[1]

        matrix.test_all_matrix_types(
            self.X,
            utils.assert_transform_equals,
            Y=libsize_sub,
            transform=test_fun,
            check=utils.assert_all_close,
            n=20,
            seed=42,
        )

    def test_subsample_mismatch_size(self):
        libsize = self.libsize[:25]
        utils.assert_raises_message(
            ValueError,
            "Expected `data` and `extra_data` to have the same number of "
            "rows. Got [100, 25]",
            scprep.select.subsample,
            self.X,
            libsize,
            n=20,
        )

    def test_subsample_n_too_large(self):
        utils.assert_raises_message(
            ValueError,
            "Expected n (101) <= n_samples (100)",
            scprep.select.subsample,
            self.X,
            n=self.X.shape[0] + 1,
        )

    def test_sparse_dataframe_fill_value(self):
        def test_fun(X):
            Y = scprep.select.select_rows(X, idx=np.arange(X.shape[0] // 2))
            for col in Y.columns:
                assert X[col].dtype == Y[col].dtype, (X[col].dtype, Y[col].dtype)
            Y = scprep.select.select_cols(X, idx=np.arange(X.shape[1] // 2))
            for col in Y.columns:
                assert X[col].dtype == Y[col].dtype, (X[col].dtype, Y[col].dtype)

        matrix.test_matrix_types(
            self.X.astype(float), test_fun, matrix._pandas_sparse_matrix_types
        )

    def test_select_variable_genes(self):
        X = scprep.filter.filter_rare_genes(self.X, cutoff=5)
        X_filtered = scprep.select.highly_variable_genes(X, percentile=90)
        assert X_filtered.shape[0] == X.shape[0]
        assert X_filtered.shape[1] == int(np.round(X.shape[1] / 10)), (
            X.shape[1],
            X_filtered.shape[1],
        )
        assert X.columns[np.argmax(X.values.std(axis=0))] in X_filtered.columns
        matrix.test_all_matrix_types(
            X,
            utils.assert_transform_equals,
            Y=X_filtered,
            transform=scprep.select.highly_variable_genes,
            percentile=90,
        )


def test_string_subset_exact_word():
    np.testing.assert_array_equal(
        scprep.select._get_string_subset_mask(["hello", "world"], exact_word="hello"),
        [True, False],
    )
    np.testing.assert_array_equal(
        scprep.select._get_string_subset_mask([" hello ", "world"], exact_word="hello"),
        [True, False],
    )
    np.testing.assert_array_equal(
        scprep.select._get_string_subset_mask(["(hello)", "world"], exact_word="hello"),
        [True, False],
    )
    np.testing.assert_array_equal(
        scprep.select._get_string_subset_mask(["[hello]", "world"], exact_word="hello"),
        [True, False],
    )
    np.testing.assert_array_equal(
        scprep.select._get_string_subset_mask(
            ["hello...?", "world"], exact_word="hello"
        ),
        [True, False],
    )
    np.testing.assert_array_equal(
        scprep.select._get_string_subset_mask(
            ["hello world", "world"], exact_word="hello"
        ),
        [True, False],
    )
    np.testing.assert_array_equal(
        scprep.select._get_string_subset_mask(
            ["(hello) world", "world"], exact_word="hello"
        ),
        [True, False],
    )
    np.testing.assert_array_equal(
        scprep.select._get_string_subset_mask(
            ["World, hello!", "world"], exact_word="hello"
        ),
        [True, False],
    )
    np.testing.assert_array_equal(
        scprep.select._get_string_subset_mask(
            ["helloooo!", "world"], exact_word="hello"
        ),
        [False, False],
    )
    np.testing.assert_array_equal(
        scprep.select._get_string_subset_mask(
            ["(hello) world", "world"], exact_word="(hello) world"
        ),
        [True, False],
    )


def test_string_subset_list():
    np.testing.assert_array_equal(
        scprep.select._get_string_subset_mask(
            ["hello", "world"], exact_word=["hello", "world"]
        ),
        [True, True],
    )
    np.testing.assert_array_equal(
        scprep.select._get_string_subset_mask(
            ["hello", "world"], exact_word=["hello", "earth"]
        ),
        [True, False],
    )
    np.testing.assert_array_equal(
        scprep.select._get_string_subset_mask(
            ["hello", "world"], starts_with=["hell", "w"]
        ),
        [True, True],
    )
    np.testing.assert_array_equal(
        scprep.select._get_string_subset_mask(
            ["hello", "world"], starts_with=["hell", "e"]
        ),
        [True, False],
    )
    np.testing.assert_array_equal(
        scprep.select._get_string_subset_mask(
            ["hello", "world"], ends_with=["ello", "ld"]
        ),
        [True, True],
    )
    np.testing.assert_array_equal(
        scprep.select._get_string_subset_mask(
            ["hello", "world"], ends_with=["ello", "h"]
        ),
        [True, False],
    )
    np.testing.assert_array_equal(
        scprep.select._get_string_subset_mask(
            ["hello", "world"], regex=["^hell.", "^.or.*"]
        ),
        [True, True],
    )
    np.testing.assert_array_equal(
        scprep.select._get_string_subset_mask(
            ["hello", "world"], regex=["^hell", "^earth"]
        ),
        [True, False],
    )
