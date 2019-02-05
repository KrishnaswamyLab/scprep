from tools import data, matrix, utils
import scprep
from sklearn.utils.testing import assert_raise_message, assert_warns_message
import numpy as np
import pandas as pd
import unittest


class Test10X(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.X = data.load_10X(sparse=False)
        self.X_sparse = data.load_10X(sparse=True)
        self.libsize = scprep.measure.library_size(self.X)

    def test_get_gene_set_starts_with(self):
        gene_idx = np.argwhere([g.startswith("D")
                                for g in self.X.columns]).flatten()
        gene_names = self.X.columns[gene_idx]
        assert np.all(scprep.select.get_gene_set(
            self.X, starts_with="D") == gene_names)
        assert np.all(scprep.select.get_gene_set(
            self.X, regex="^D") == gene_names)
        assert np.all(scprep.select.get_gene_set(
            self.X.columns, regex="^D") == gene_names)

    def test_get_gene_set_ends_with(self):
        gene_idx = np.argwhere([g.endswith("8")
                                for g in self.X.columns]).flatten()
        gene_names = self.X.columns[gene_idx]
        assert np.all(scprep.select.get_gene_set(
            self.X, ends_with="8") == gene_names)
        assert np.all(scprep.select.get_gene_set(
            self.X, regex="8$") == gene_names)

    def test_get_gene_set_ndarray(self):
        assert_raise_message(
            TypeError,
            "data must be a list of gene names or a pandas "
            "DataFrame. Got ndarray",
            scprep.select.get_gene_set,
            data=self.X.values, regex="8$")

    def test_get_gene_set_no_condition(self):
        assert_warns_message(
            UserWarning,
            "No selection conditions provided. Returning all genes.",
            scprep.select.get_gene_set, self.X)

    def test_get_cell_set_starts_with(self):
        cell_idx = np.argwhere([g.startswith("A")
                                for g in self.X.index]).flatten()
        cell_names = self.X.index[cell_idx]
        assert np.all(scprep.select.get_cell_set(
            self.X, starts_with="A") == cell_names)
        assert np.all(scprep.select.get_cell_set(
            self.X, regex="^A") == cell_names)
        assert np.all(scprep.select.get_cell_set(
            self.X.index, regex="^A") == cell_names)

    def test_get_cell_set_ends_with(self):
        cell_idx = np.argwhere([g.endswith("G-1")
                                for g in self.X.index]).flatten()
        cell_names = self.X.index[cell_idx]
        assert np.all(scprep.select.get_cell_set(
            self.X, ends_with="G-1") == cell_names)
        assert np.all(scprep.select.get_cell_set(
            self.X, regex="G\\-1$") == cell_names)

    def test_get_cell_set_ndarray(self):
        assert_raise_message(
            TypeError,
            "data must be a list of cell names or a pandas "
            "DataFrame. Got ndarray",
            scprep.select.get_cell_set,
            data=self.X.values, regex="G\\-1$")

    def test_get_cell_set_no_condition(self):
        assert_warns_message(
            UserWarning,
            "No selection conditions provided. Returning all cells.",
            scprep.select.get_cell_set, self.X)

    def test_select_rows_boolean_index(self):
        matrix.test_all_matrix_types(
            self.X, scprep.select.select_rows,
            idx=np.random.choice([True, False], [self.X.shape[0]]))

    def test_select_rows_integer_array_index(self):
        matrix.test_all_matrix_types(
            self.X, scprep.select.select_rows,
            idx=np.random.choice(self.X.shape[0], self.X.shape[0] // 2))

    def test_select_rows_integer_list_index(self):
        matrix.test_all_matrix_types(
            self.X, scprep.select.select_rows,
            idx=np.random.choice(self.X.shape[0], self.X.shape[0] // 2).tolist())

    def test_select_rows_integer_index(self):
        matrix.test_all_matrix_types(
            self.X, scprep.select.select_rows,
            idx=np.random.choice(self.X.shape[0]))

    def test_select_rows_string_array_index(self):
        matrix.test_pandas_matrix_types(
            self.X, scprep.select.select_rows,
            idx=np.random.choice(self.X.index.values, self.X.shape[0] // 2))

    def test_select_rows_pandas_index_index(self):
        matrix.test_pandas_matrix_types(
            self.X, scprep.select.select_rows,
            idx=self.X.index[np.random.choice([True, False], [self.X.shape[0]])])

    def test_select_rows_series_index(self):
        matrix.test_pandas_matrix_types(
            self.X, scprep.select.select_rows,
            idx=pd.Series(self.X.index[np.random.choice([True, False], [self.X.shape[0]])]))

    def test_select_rows_dataframe_index(self):
        matrix.test_all_matrix_types(
            self.X, scprep.select.select_rows,
            idx=pd.DataFrame(np.random.choice([True, False], [self.X.shape[0], 1]),
                             index=self.X.index))

    def test_select_rows_series_data_boolean_index(self):
        scprep.select.select_rows(
            self.X, self.X.iloc[:, 0], idx=np.random.choice([True, False], [self.X.shape[0]]))

    def test_select_rows_sparse_series_data_boolean_index(self):
        scprep.select.select_rows(
            self.X, self.X_sparse.iloc[:, 0], idx=np.random.choice([True, False], [self.X.shape[0]]))

    def test_select_rows_series_data_integer_index(self):
        scprep.select.select_rows(
            self.X, self.X.iloc[:, 0], idx=np.random.choice(self.X.shape[1], self.X.shape[0] // 2))

    def test_select_rows_sparse_series_data_integer_index(self):
        scprep.select.select_rows(
            self.X, self.X_sparse.iloc[:, 0], idx=np.random.choice(self.X.shape[1], self.X.shape[0] // 2))

    def test_select_rows_1d_array_data(self):
        scprep.select.select_rows(
            self.X, self.X.values[:, 0], idx=np.random.choice([True, False], [self.X.shape[0]]))

    def test_select_rows_list_data(self):
        scprep.select.select_rows(
            self.X, self.X.values[:, 0].tolist(), idx=np.random.choice([True, False], [self.X.shape[1]]))

    def test_select_rows_get_cell_set(self):
        matrix.test_pandas_matrix_types(
            self.X, scprep.select.select_rows, self.X.iloc[:, 0],
            starts_with="A")

    def test_select_rows_zero_rows(self):
        assert_warns_message(
            UserWarning,
            "Selecting 0 rows",
            scprep.select.select_rows, self.X,
            idx=(self.X.sum(axis=1) < 0))

    def test_select_rows_no_condition(self):
        assert_warns_message(
            UserWarning,
            "No selection conditions provided. Returning all rows.",
            scprep.select.select_rows, self.X)

    def test_select_cols_boolean_index(self):
        matrix.test_all_matrix_types(
            self.X, scprep.select.select_cols,
            idx=np.random.choice([True, False], [self.X.shape[1]]))

    def test_select_cols_integer_array_index(self):
        matrix.test_all_matrix_types(
            self.X, scprep.select.select_cols,
            idx=np.random.choice(self.X.shape[1], self.X.shape[1] // 2))

    def test_select_cols_integer_list_index(self):
        matrix.test_all_matrix_types(
            self.X, scprep.select.select_cols,
            idx=np.random.choice(self.X.shape[1], self.X.shape[1] // 2).tolist())

    def test_select_cols_integer_index(self):
        matrix.test_all_matrix_types(
            self.X, scprep.select.select_cols,
            idx=np.random.choice(self.X.shape[1]))

    def test_select_cols_string_array_index(self):
        matrix.test_pandas_matrix_types(
            self.X, scprep.select.select_cols,
            idx=np.random.choice(self.X.columns.values, self.X.shape[1] // 2))

    def test_select_cols_pandas_index_index(self):
        matrix.test_pandas_matrix_types(
            self.X, scprep.select.select_cols,
            idx=self.X.columns[np.random.choice([True, False], [self.X.shape[1]])])

    def test_select_cols_series_index(self):
        matrix.test_pandas_matrix_types(
            self.X, scprep.select.select_cols,
            idx=pd.Series(self.X.columns[np.random.choice([True, False], [self.X.shape[1]])]))

    def test_select_cols_dataframe_index(self):
        matrix.test_all_matrix_types(
            self.X, scprep.select.select_cols,
            idx=pd.DataFrame(np.random.choice([True, False], [1, self.X.shape[1]]),
                             index=[1], columns=self.X.columns))

    def test_select_cols_series_data_boolean_index(self):
        scprep.select.select_cols(
            self.X, self.X.iloc[0, :], idx=np.random.choice([True, False], [self.X.shape[1]]))

    def test_select_cols_sparse_series_data_boolean_index(self):
        scprep.select.select_cols(
            self.X, self.X_sparse.iloc[0, :], idx=np.random.choice([True, False], [self.X.shape[1]]))

    def test_select_cols_series_data_integer_index(self):
        scprep.select.select_cols(
            self.X, self.X.iloc[0, :], idx=np.random.choice(self.X.shape[1], self.X.shape[1] // 2))

    def test_select_cols_sparse_series_data_integer_index(self):
        scprep.select.select_cols(
            self.X, self.X_sparse.iloc[0, :], idx=np.random.choice(self.X.shape[1], self.X.shape[1] // 2))

    def test_select_cols_1d_array_data(self):
        scprep.select.select_cols(
            self.X, self.X.values[0, :], idx=np.random.choice([True, False], [self.X.shape[1]]))

    def test_select_cols_list_data(self):
        scprep.select.select_cols(
            self.X, self.X.values[0, :].tolist(), idx=np.random.choice([True, False], [self.X.shape[1]]))

    def test_select_cols_get_gene_set(self):
        matrix.test_pandas_matrix_types(
            self.X, scprep.select.select_cols, self.X.iloc[0, :],
            starts_with="D")

    def test_select_cols_zero_columns(self):
        assert_warns_message(
            UserWarning,
            "Selecting 0 columns",
            scprep.select.select_cols, self.X,
            idx=(self.X.sum(axis=0) < 0))

    def test_select_cols_no_condition(self):
        assert_warns_message(
            UserWarning,
            "No selection conditions provided. Returning all columns.",
            scprep.select.select_cols, self.X)

    def test_select_rows_invalid_index(self):
        assert_raise_message(KeyError,
                             "the label [not_a_cell] is not in the [index]",
                             scprep.select.select_rows,
                             self.X,
                             idx='not_a_cell')

    def test_select_cols_invalid_index(self):
        assert_raise_message(KeyError,
                             "the label [not_a_gene] is not in the [columns]",
                             scprep.select.select_cols,
                             self.X,
                             idx='not_a_gene')

    def test_select_rows_2d_index(self):
        assert_raise_message(ValueError,
                             "Expected idx to be 1D. Got shape ",
                             scprep.select.select_rows,
                             self.X,
                             idx=pd.DataFrame([self.X.index, self.X.index]))

    def test_select_cols_2d_index(self):
        assert_raise_message(ValueError,
                             "Expected idx to be 1D. Got shape ",
                             scprep.select.select_cols,
                             self.X,
                             idx=pd.DataFrame([self.X.index, self.X.index]))

    def test_select_cols_unequal_columns(self):
        assert_raise_message(
            ValueError,
            "Expected all data to have the same number of "
            "columns. Got [100, 50]",
            scprep.select.select_cols,
            self.X,
            self.X.values[:, :50])

    def test_select_rows_unequal_rows(self):
        assert_raise_message(
            ValueError,
            "Expected all data to have the same number of "
            "rows. Got [100, 50]",
            scprep.select.select_rows,
            self.X,
            self.X.values[:50, :])

    def test_select_cols_conflicting_data(self):
        assert_raise_message(
            ValueError,
            "Expected all pandas inputs to have the same columns. "
            "Fix with "
            "`scprep.select.select_cols(extra_data, data.columns)`",
            scprep.select.select_cols,
            self.X,
            scprep.select.subsample(self.X.T, n=self.X.shape[0]).T)

    def test_select_rows_conflicting_data(self):
        assert_raise_message(
            ValueError,
            "Expected all pandas inputs to have the same index. "
            "Fix with "
            "`scprep.select.select_rows(extra_data, data.index)`",
            scprep.select.select_rows,
            self.X,
            scprep.select.subsample(self.X, n=self.X.shape[0]))

    def test_select_cols_get_gene_set_ndarray_data(self):
        assert_raise_message(
            ValueError,
            "Can only select based on column names with DataFrame input. "
            "Please set `idx` to select specific columns.",
            scprep.select.select_cols, self.X.values, starts_with="A"
        )

    def test_select_rows_get_cell_set_ndarray_data(self):
        assert_raise_message(
            ValueError,
            "Can only select based on row names with DataFrame input. "
            "Please set `idx` to select specific rows.",
            scprep.select.select_rows, self.X.values, starts_with="A"
        )

    def test_subsample(self):
        self.X = data.generate_positive_sparse_matrix(shape=(50, 100))
        Y = scprep.select.subsample(self.X, n=20, seed=42)
        matrix.test_all_matrix_types(
            self.X, utils.assert_transform_equals, Y=Y,
            transform=scprep.select.subsample,
            check=utils.assert_all_equal, n=20, seed=42)

    def test_subsample_multiple(self):
        Y, libsize_sub = scprep.select.subsample(
            self.X, self.libsize, n=20, seed=42)

        def test_fun(X, **kwargs):
            libsize = scprep.measure.library_size(X)
            return scprep.select.subsample(X, libsize, **kwargs)[0]
        matrix.test_all_matrix_types(
            self.X, utils.assert_transform_equals, Y=Y,
            transform=test_fun,
            check=utils.assert_all_equal, n=20, seed=42)

        def test_fun(X, **kwargs):
            libsize = scprep.measure.library_size(X)
            return scprep.select.subsample(X, libsize, **kwargs)[1]
        matrix.test_all_matrix_types(
            self.X, utils.assert_transform_equals, Y=libsize_sub,
            transform=test_fun,
            check=utils.assert_all_close, n=20, seed=42)

    def test_subsample_mismatch_size(self):
        libsize = self.libsize[:25]
        assert_raise_message(
            ValueError,
            "Expected all data to have the same number of rows. "
            "Got [100, 25]",
            scprep.select.subsample, self.X, libsize, n=20)

    def test_subsample_n_too_large(self):
        assert_raise_message(
            ValueError,
            "Expected n (101) <= n_samples (100)",
            scprep.select.subsample, self.X, n=self.X.shape[0] + 1)
