import numpy as np
import pandas as pd

import numbers
import warnings
import importlib
import re

from scipy import sparse
from decorator import decorator

from . import select

try:
    ModuleNotFoundError
except NameError:
    # python 3.5
    ModuleNotFoundError = ImportError

__imported_pkgs = set()


def _try_import(pkg):
    try:
        return importlib.import_module(pkg)
    except ModuleNotFoundError:
        return None


def _version_check(version, min_version=None):
    if min_version is None:
        # no requirement
        return True
    min_version = str(min_version)
    min_version_split = re.split(r"[^0-9]+", min_version)
    version_split = re.split(r"[^0-9]+", version)
    version_major = int(version_split[0])
    min_major = int(min_version_split[0])
    if min_major > version_major:
        # failed major version requirement
        return False
    elif min_major < version_major:
        # exceeded major version requirement
        return True
    elif len(min_version_split) == 1:
        # no minor version requirement
        return True
    else:
        version_minor = int(version_split[1])
        min_minor = int(min_version_split[1])
        if min_minor > version_minor:
            # failed minor version requirement
            return False
        else:
            # met minor version requirement
            return True


def check_version(pkg, min_version=None):
    try:
        module = importlib.import_module(pkg)
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "{0} not found. "
            "Please install it with e.g. `pip install --user {0}`".format(pkg)
        )
    if not _version_check(module.__version__, min_version):
        raise ImportError(
            "{0}>={1} is required (installed: {2}). "
            "Please upgrade it with e.g."
            " `pip install --user --upgrade {0}`".format(
                pkg, min_version, module.__version__
            )
        )


@decorator
def _with_pkg(fun, pkg=None, min_version=None, *args, **kwargs):
    global __imported_pkgs
    if (pkg, min_version) not in __imported_pkgs:
        check_version(pkg, min_version=min_version)
        __imported_pkgs.add((pkg, min_version))
    return fun(*args, **kwargs)


def _get_percentile_cutoff(data, cutoff=None, percentile=None, required=False):
    """Get a cutoff for a dataset

    Parameters
    ----------
    data : array-like
    cutoff : float or None, optional (default: None)
        Absolute cutoff value. Only one of cutoff and percentile may be given
    percentile : float or None, optional (default: None)
        Percentile cutoff value between 0 and 100.
        Only one of cutoff and percentile may be given
    required : bool, optional (default: False)
        If True, one of cutoff and percentile must be given.

    Returns
    -------
    cutoff : float or None
        Absolute cutoff value. Can only be None if required is False and
        cutoff and percentile are both None.
    """
    if percentile is not None:
        if cutoff is not None:
            raise ValueError(
                "Only one of `cutoff` and `percentile` should be given."
                "Got cutoff={}, percentile={}".format(cutoff, percentile)
            )
        if not isinstance(percentile, numbers.Number):
            return [_get_percentile_cutoff(data, percentile=p) for p in percentile]
        if percentile < 1:
            warnings.warn(
                "`percentile` expects values between 0 and 100."
                "Got {}. Did you mean {}?".format(percentile, percentile * 100),
                UserWarning,
            )
        cutoff = np.percentile(np.array(data).reshape(-1), percentile)
    elif cutoff is None and required:
        raise ValueError("One of either `cutoff` or `percentile` must be given.")
    return cutoff


def _get_filter_idx(values, cutoff, percentile, keep_cells):
    """Return a boolean array to index cells based on a filter

    Parameters
    ----------
    values : list-like, shape=[n_samples]
        Value upon which to filter
    cutoff : float or tuple of floats, optional (default: None)
        Value above or below which to retain cells. Only one of `cutoff`
        and `percentile` should be specified.
    percentile : int or tuple of ints, optional (Default: None)
        Percentile above or below which to retain cells.
        Must be an integer between 0 and 100. Only one of `cutoff`
        and `percentile` should be specified.
    keep_cells : {'above', 'below', 'between'} or None, optional (default: None)
        Keep cells above, below or between the cutoff.
        If None, defaults to 'above' when a single cutoff is given and
        'between' when two cutoffs are given.

    Returns
    -------
    keep_cells_idx : list-like
        Boolean retention array
    """
    cutoff = _get_percentile_cutoff(values, cutoff, percentile, required=True)
    if keep_cells is None:
        if isinstance(cutoff, numbers.Number):
            keep_cells = "above"
        else:
            keep_cells = "between"
    if keep_cells == "above":
        if not isinstance(cutoff, numbers.Number):
            raise ValueError(
                "Expected a single cutoff with keep_cells='above'."
                " Got {}".format(cutoff)
            )
        keep_cells_idx = values > cutoff
    elif keep_cells == "below":
        if not isinstance(cutoff, numbers.Number):
            raise ValueError(
                "Expected a single cutoff with keep_cells='below'."
                " Got {}".format(cutoff)
            )
        keep_cells_idx = values < cutoff
    elif keep_cells == "between":
        if isinstance(cutoff, numbers.Number) or len(cutoff) != 2:
            raise ValueError(
                "Expected cutoff of length 2 with keep_cells='between'."
                " Got {}".format(cutoff)
            )
        keep_cells_idx = np.logical_and(
            values > np.min(cutoff), values < np.max(cutoff)
        )
    else:
        raise ValueError(
            "Expected `keep_cells` in ['above', 'below', 'between']. "
            "Got {}".format(keep_cells)
        )
    return keep_cells_idx


def toarray(x):
    """Convert an array-like to a np.ndarray

    Parameters
    ----------
    x : array-like
        Array-like to be converted

    Returns
    -------
    x : np.ndarray
    """
    if is_SparseDataFrame(x):
        x = x.to_coo().toarray()
    elif is_SparseSeries(x):
        x = x.to_dense().to_numpy()
    elif isinstance(x, (pd.DataFrame, pd.Series, pd.Index)):
        x = x.to_numpy()
    elif isinstance(x, sparse.spmatrix):
        x = x.toarray()
    elif isinstance(x, np.matrix):
        x = x.A
    elif isinstance(x, list):
        x_out = []
        for xi in x:
            try:
                xi = toarray(xi)
            except TypeError:
                # recursed too far
                pass
            x_out.append(xi)
        try:
            x = np.array(x_out)
        except ValueError as e:
            if str(e) == "setting an array element with a sequence":
                x = np.array(x_out, dtype=object)
            else:
                raise
    elif isinstance(x, (np.ndarray, numbers.Number)):
        pass
    else:
        raise TypeError("Expected array-like. Got {}".format(type(x)))
    return x


def to_array_or_spmatrix(x):
    """Convert an array-like to a np.ndarray or scipy.sparse.spmatrix

    Parameters
    ----------
    x : array-like
        Array-like to be converted

    Returns
    -------
    x : np.ndarray or scipy.sparse.spmatrix
    """
    if is_SparseDataFrame(x):
        x = x.to_coo()
    elif is_sparse_dataframe(x) or is_sparse_series(x):
        x = x.sparse.to_coo()
    elif isinstance(
        x, (sparse.spmatrix, np.ndarray, numbers.Number)
    ) and not isinstance(x, np.matrix):
        pass
    elif isinstance(x, list):
        x_out = []
        for xi in x:
            try:
                xi = to_array_or_spmatrix(xi)
            except TypeError:
                # recursed too far
                pass
            x_out.append(xi)
        x = np.array(x_out)
    else:
        x = toarray(x)
    return x


def is_SparseSeries(X):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            "The SparseSeries class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version",
            FutureWarning,
        )
        try:
            return isinstance(X, pd.SparseSeries)
        except AttributeError:
            return False


def is_SparseDataFrame(X):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            "The SparseDataFrame class is removed from pandas. Accessing it from the top-level namespace will also be removed in the next version",
            FutureWarning,
        )
        try:
            return isinstance(X, pd.SparseDataFrame)
        except AttributeError:
            return False


def is_sparse_dataframe(x):
    if isinstance(x, pd.DataFrame) and not is_SparseDataFrame(x):
        try:
            x.sparse
            return True
        except AttributeError:
            pass
    return False


def is_sparse_series(x):
    if isinstance(x, pd.Series) and not is_SparseSeries(x):
        try:
            x.sparse
            return True
        except AttributeError:
            pass
    return False


def dataframe_to_sparse(x, fill_value=0.0):
    return x.astype(pd.SparseDtype(float, fill_value=fill_value))


def SparseDataFrame(X, columns=None, index=None, default_fill_value=0.0):
    if sparse.issparse(X):
        X = pd.DataFrame.sparse.from_spmatrix(X)
        X.sparse.fill_value = default_fill_value
    else:
        if is_SparseDataFrame(X) or not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        X = dataframe_to_sparse(X, fill_value=default_fill_value)
    if columns is not None:
        X.columns = columns
    if index is not None:
        X.index = index
    return X


def matrix_transform(data, fun, *args, **kwargs):
    """Perform a numerical transformation to data

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    fun : callable
        Numerical transformation function, `np.ufunc` or similar.
    args, kwargs : additional arguments, optional
        arguments for `fun`. `data` is always passed as the first argument

    Returns
    -------
    data : array-like, shape=[n_samples, n_features]
        Transformed output data
    """
    if is_sparse_dataframe(data) or is_SparseDataFrame(data):
        data = data.copy()
        for col in data.columns:
            data[col] = fun(data[col], *args, **kwargs)
    elif sparse.issparse(data):
        if isinstance(data, (sparse.lil_matrix, sparse.dok_matrix)):
            data = data.tocsr()
        else:
            # avoid modifying in place
            data = data.copy()
        data.data = fun(data.data, *args, **kwargs)
    else:
        data = fun(data, *args, **kwargs)
    return data


def matrix_sum(data, axis=None):
    """Get the column-wise, row-wise, or total sum of values in a matrix

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    axis : int or None, optional (default: None)
        Axis across which to sum. axis=0 gives column sums,
        axis=1 gives row sums. None gives the total sum.

    Returns
    -------
    sums : array-like or float
        Sums along desired axis.
    """
    if axis not in [0, 1, None]:
        raise ValueError("Expected axis in [0, 1, None]. Got {}".format(axis))
    if isinstance(data, pd.DataFrame):
        if is_SparseDataFrame(data):
            if axis is None:
                sums = data.to_coo().sum()
            else:
                index = data.index if axis == 1 else data.columns
                sums = pd.Series(
                    np.array(data.to_coo().sum(axis)).flatten(), index=index
                )
        elif is_sparse_dataframe(data):
            if axis is None:
                sums = data.sparse.to_coo().sum()
            else:
                index = data.index if axis == 1 else data.columns
                sums = pd.Series(
                    np.array(data.sparse.to_coo().sum(axis)).flatten(), index=index
                )
        elif axis is None:
            sums = data.to_numpy().sum()
        else:
            sums = data.sum(axis)
    else:
        sums = np.sum(data, axis=axis)
        if isinstance(sums, np.matrix):
            sums = np.array(sums).flatten()
    return sums


def matrix_std(data, axis=None):
    """Get the column-wise, row-wise, or total standard deviation of a matrix

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    axis : int or None, optional (default: None)
        Axis across which to calculate standard deviation.
        axis=0 gives column standard deviation,
        axis=1 gives row standard deviation.
        None gives the total standard deviation.

    Returns
    -------
    std : array-like or float
        Standard deviation along desired axis.
    """
    if axis not in [0, 1, None]:
        raise ValueError("Expected axis in [0, 1, None]. Got {}".format(axis))
    index = None
    if isinstance(data, pd.DataFrame) and axis is not None:
        if axis == 1:
            index = data.index
        elif axis == 0:
            index = data.columns
    data = to_array_or_spmatrix(data)
    if sparse.issparse(data):
        if axis is None:
            if isinstance(data, (sparse.lil_matrix, sparse.dok_matrix)):
                data = data.tocoo()
            data_sq = data.copy()
            data_sq.data = data_sq.data ** 2
            variance = data_sq.mean() - data.mean() ** 2
            std = np.sqrt(variance)
        else:
            if axis == 0:
                data = data.tocsc()
                next_fn = data.getcol
                N = data.shape[1]
            elif axis == 1:
                data = data.tocsr()
                next_fn = data.getrow
                N = data.shape[0]
            std = []
            for i in range(N):
                col = next_fn(i)
                col_sq = col.copy()
                col_sq.data = col_sq.data ** 2
                variance = col_sq.mean() - col.mean() ** 2
                std.append(np.sqrt(variance))
            std = np.array(std)
    else:
        std = np.std(data, axis=axis)
    if index is not None:
        std = pd.Series(std, index=index, name="std")
    return std


def matrix_vector_elementwise_multiply(data, multiplier, axis=None):
    """Elementwise multiply a matrix by a vector

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    multiplier : array-like, shape=[n_samples, 1] or [1, n_features]
        Vector by which to multiply `data`
    axis : int or None, optional (default: None)
        Axis across which to sum. axis=0 multiplies each column,
        axis=1 multiplies each row. None guesses based on dimensions

    Returns
    -------
    product : array-like
        Multiplied matrix
    """
    if axis not in [0, 1, None]:
        raise ValueError("Expected axis in [0, 1, None]. Got {}".format(axis))

    if axis is None:
        if data.shape[0] == data.shape[1]:
            raise RuntimeError(
                "`data` is square, cannot guess axis from input. "
                "Please provide `axis=0` to multiply along rows or "
                "`axis=1` to multiply along columns."
            )
        elif np.prod(multiplier.shape) == data.shape[0]:
            axis = 0
        elif np.prod(multiplier.shape) == data.shape[1]:
            axis = 1
        else:
            raise ValueError(
                "Expected `multiplier` to be a vector of length "
                "`data.shape[0]` ({}) or `data.shape[1]` ({}). Got {}".format(
                    data.shape[0], data.shape[1], multiplier.shape
                )
            )
    multiplier = toarray(multiplier)
    if axis == 0:
        if not np.prod(multiplier.shape) == data.shape[0]:
            raise ValueError(
                "Expected `multiplier` to be a vector of length "
                "`data.shape[0]` ({}). Got {}".format(data.shape[0], multiplier.shape)
            )
        multiplier = multiplier.reshape(-1, 1)
    else:
        if not np.prod(multiplier.shape) == data.shape[1]:
            raise ValueError(
                "Expected `multiplier` to be a vector of length "
                "`data.shape[1]` ({}). Got {}".format(data.shape[1], multiplier.shape)
            )
        multiplier = multiplier.reshape(1, -1)

    if is_SparseDataFrame(data) or is_sparse_dataframe(data):
        data = data.copy()
        multiplier = multiplier.flatten()
        if axis == 0:
            for col in data.columns:
                try:
                    mult_indices = data[col].values.sp_index.indices
                except AttributeError:
                    mult_indices = data[col].values.sp_index.to_int_index().indices
                new_data = data[col].values.sp_values * multiplier[mult_indices]
                data[col].values.sp_values.put(
                    np.arange(data[col].sparse.npoints), new_data
                )
        else:
            for col, mult in zip(data.columns, multiplier):
                data[col] = data[col] * mult
    elif isinstance(data, pd.DataFrame):
        data = data.mul(multiplier.flatten(), axis=axis)
    elif sparse.issparse(data):
        if isinstance(
            data,
            (
                sparse.lil_matrix,
                sparse.dok_matrix,
                sparse.coo_matrix,
                sparse.bsr_matrix,
                sparse.dia_matrix,
            ),
        ):
            data = data.tocsr()
        data = data.multiply(multiplier)
    else:
        data = data * multiplier

    return data


def sparse_series_min(data):
    """Get the minimum value from a pandas sparse series

    Pandas SparseDataFrame does not handle np.min.

    Parameters
    ----------
    data : pd.Series[SparseArray]
        Input data

    Returns
    -------
    minimum : float
        Minimum entry in `data`.
    """
    return np.concatenate([data.sparse.sp_values, [data.sparse.fill_value]]).min()


def matrix_min(data):
    """Get the minimum value from a data matrix.

    Pandas SparseDataFrame does not handle np.min.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data

    Returns
    -------
    minimum : float
        Minimum entry in `data`.
    """
    if is_SparseDataFrame(data):
        data = [np.min(data[col]) for col in data.columns]
    elif is_sparse_dataframe(data):
        data = [sparse_series_min(data[col]) for col in data.columns]
    elif isinstance(data, pd.DataFrame):
        data = np.min(data)
    elif isinstance(data, sparse.lil_matrix):
        data = [np.min(d) for d in data.data] + [0]
    elif isinstance(data, sparse.dok_matrix):
        data = list(data.values()) + [0]
    elif isinstance(data, sparse.dia_matrix):
        data = [np.min(data.data), 0]
    return np.min(data)


def matrix_non_negative(data, allow_equal=True):
    """Check if all values in a matrix are non-negative

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    allow_equal : bool, optional (default: True)
        If True, min(data) can be equal to 0

    Returns
    -------
    is_non_negative : bool
    """
    return matrix_min(data) >= 0 if allow_equal else matrix_min(data) > 0


def matrix_any(condition):
    """Check if a condition is true anywhere in a data matrix

    np.any doesn't handle matrices of type pd.DataFrame

    Parameters
    ----------
    condition : array-like
        Boolean matrix

    Returns
    -------
    any : bool
        True if condition contains any True values, False otherwise
    """
    return np.sum(np.sum(condition)) > 0


def check_consistent_columns(data, common_columns_only=True):
    """Ensure that a set of data matrices have consistent columns

    Parameters
    ----------
    data : list of array-likes
        List of matrices to be checked
    common_columns_only : bool, optional (default: True)
        With pandas inputs, drop any columns that are not common to
        all matrices

    Returns
    -------
    data : list of array-likes
        List of matrices with consistent columns, subsetted if necessary

    Raises
    ------
    ValueError
        Raised if data has inconsistent number of columns and does not
        have column names for subsetting
    """
    matrix_type = type(data[0])
    matrix_shape = data[0].shape[1]
    if issubclass(matrix_type, pd.DataFrame):
        if not (
            np.all([d.shape[1] == matrix_shape for d in data[1:]])
            and np.all([data[0].columns == d.columns for d in data])
        ):
            if common_columns_only:
                common_genes = data[0].columns.values
                for d in data[1:]:
                    common_genes = common_genes[np.isin(common_genes, d.columns.values)]
                warnings.warn(
                    "Input data has inconsistent column names. "
                    "Subsetting to {} common columns. "
                    "To retain all columns, use "
                    "`common_columns_only=False`.".format(len(common_genes)),
                    UserWarning,
                )
                for i in range(len(data)):
                    data[i] = data[i][common_genes]
            else:
                columns = [d.columns.values for d in data]
                all_columns = np.unique(np.concatenate(columns))
                warnings.warn(
                    "Input data has inconsistent column names. "
                    "Padding with zeros to {} total columns.".format(len(all_columns)),
                    UserWarning,
                )
    else:
        for d in data[1:]:
            if not d.shape[1] == matrix_shape:
                shapes = ", ".join([str(d.shape[1]) for d in data])
                raise ValueError(
                    "Expected data all with the same number of "
                    "columns. Got {}".format(shapes)
                )
    return data


def combine_batches(
    data, batch_labels, append_to_cell_names=None, common_columns_only=True
):
    """Combine data matrices from multiple batches and store a batch label

    Parameters
    ----------
    data : list of array-like, shape=[n_batch]
        All matrices must be of the same format and have the same number of
        columns (or genes.)
    batch_labels : list of `str`, shape=[n_batch]
        List of names assigned to each batch
    append_to_cell_names : bool, optional (default: None)
        If input is a pandas dataframe, add the batch label corresponding to
        each cell to its existing index (or cell name / barcode.)
        Default behavior is `True` for dataframes and `False` otherwise.
    common_columns_only : bool, optional (default: True)
        With pandas inputs, drop any columns that are not common to
        all data matrices

    Returns
    -------
    data : data matrix, shape=[n_samples, n_features]
        Number of samples is the sum of numbers of samples of all batches.
        Number of features is the same as each of the batches.
    sample_labels : list-like, shape=[n_samples]
        Batch labels corresponding to each sample
    """
    if not len(data) == len(batch_labels):
        raise ValueError(
            "Expected data ({}) and batch_labels ({}) to be the "
            "same length.".format(len(data), len(batch_labels))
        )

    # check consistent type
    matrix_type = type(data[0])
    if is_SparseDataFrame(data[0]):
        matrix_type = pd.DataFrame
    if not issubclass(matrix_type, (np.ndarray, pd.DataFrame, sparse.spmatrix)):
        raise ValueError(
            "Expected data to contain pandas DataFrames, "
            "scipy sparse matrices or numpy arrays. "
            "Got {}".format(matrix_type.__name__)
        )
    for d in data[1:]:
        if not isinstance(d, matrix_type):
            types = ", ".join([type(d).__name__ for d in data])
            raise TypeError(
                "Expected data all of the same class. " "Got {}".format(types)
            )

    data = check_consistent_columns(data, common_columns_only=common_columns_only)

    # check append_to_cell_names
    if append_to_cell_names and not issubclass(matrix_type, pd.DataFrame):
        warnings.warn(
            "append_to_cell_names only valid for pd.DataFrame input."
            " Got {}".format(matrix_type.__name__),
            UserWarning,
        )
    elif append_to_cell_names is None:
        if issubclass(matrix_type, pd.DataFrame):
            if all([isinstance(d.index, pd.RangeIndex) for d in data]):
                # rangeindex should still be a rangeindex
                append_to_cell_names = False
            else:
                append_to_cell_names = True
        else:
            append_to_cell_names = False

    # concatenate labels
    sample_labels = np.concatenate(
        [np.repeat(batch_labels[i], d.shape[0]) for i, d in enumerate(data)]
    )

    # conatenate data
    if issubclass(matrix_type, pd.DataFrame):
        data_combined = pd.concat(data, axis=0, sort=True, join="outer").fillna(0)
        if append_to_cell_names:
            index = np.concatenate(
                [
                    np.core.defchararray.add(
                        np.array(d.index, dtype=str), "_" + str(batch_labels[i])
                    )
                    for i, d in enumerate(data)
                ]
            )
            data_combined.index = index
        elif all([isinstance(d.index, pd.RangeIndex) for d in data]):
            # rangeindex should still be a rangeindex
            data_combined = data_combined.reset_index(drop=True)
        sample_labels = pd.Series(
            sample_labels, index=data_combined.index, name="sample_labels"
        )
    elif issubclass(matrix_type, sparse.spmatrix):
        data_combined = sparse.vstack(data)
    elif issubclass(matrix_type, np.ndarray):
        data_combined = np.vstack(data)

    return data_combined, sample_labels


def select_cols(data, idx):
    raise RuntimeError(
        "`scprep.utils.select_cols` is deprecated. Use "
        "`scprep.select.select_cols` instead."
    )


def select_rows(data, idx):
    raise RuntimeError(
        "`scprep.utils.select_rows` is deprecated. Use "
        "`scprep.select.select_rows` instead."
    )


def get_gene_set(data, starts_with=None, ends_with=None, regex=None):
    raise RuntimeError(
        "`scprep.utils.get_gene_set` is deprecated. Use "
        "`scprep.select.get_gene_set` instead."
    )


def get_cell_set(data, starts_with=None, ends_with=None, regex=None):
    raise RuntimeError(
        "`scprep.utils.get_cell_set` is deprecated. Use "
        "`scprep.select.get_cell_set` instead."
    )


def subsample(*data, n=10000, seed=None):
    raise RuntimeError(
        "`scprep.utils.subsample` is deprecated. Use "
        "`scprep.select.subsample` instead."
    )


def sort_clusters_by_values(clusters, values):
    """Sorts `clusters` in increasing order of `values`.

    Parameters
    ----------
    clusters : array-like
        An array of cluster assignments, like the output of
        a `fit_predict()` call.
    values : type
        An associated value for each index in `clusters` to use
        for sorting the clusters.

    Returns
    -------
    new_clusters : array-likes
        Reordered cluster assignments. `np.mean(values[new_clusters == 0])`
        will be less than `np.mean(values[new_clusters == 1])` which
        will be less than `np.mean(values[new_clusters == 2])`
        and so on.

    """
    clusters = toarray(clusters)
    values = toarray(values)
    if not len(clusters) == len(values):
        raise ValueError(
            "Expected clusters ({}) and values ({}) to be the "
            "same length.".format(len(clusters), len(values))
        )

    uniq_clusters = np.unique(clusters)
    means = np.array([np.mean(values[clusters == cl]) for cl in uniq_clusters])
    new_clust_map = {
        curr_cl: i for i, curr_cl in enumerate(uniq_clusters[np.argsort(means)])
    }

    return np.array([new_clust_map[cl] for cl in clusters])
