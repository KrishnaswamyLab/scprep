from sklearn import decomposition, random_projection
import sklearn.base
import pandas as pd
from scipy import sparse
import numpy as np
import warnings


class SparseFriendlyPCA(sklearn.base.BaseEstimator):
    """Calculate PCA using random projections to handle sparse matrices

    Uses the Johnson-Lindenstrauss Lemma to determine the number of
    dimensions of random projectiosn prior to subtracting the mean.

    Parameters
    ----------
    n_components : int, optional (default: 2)
        Number of components to keep.
    eps : strictly positive float, optional (default=0.1)
        Parameter to control the quality of the embedding according to the
        Johnson-Lindenstrauss lemma when n_components is set to ‘auto’.
        Smaller values lead to better embedding but higher computation and
        memory costs
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by np.random.
    kwargs
        Additional keyword arguments for `sklearn.decomposition.PCA`
    """

    def __init__(self, n_components=2, eps=0.1, random_state=None,
                 **kwargs):
        self.pca_op = decomposition.PCA(n_components=n_components)
        self.rproj_op = random_projection.GaussianRandomProjection(eps=eps)

    @property
    def pseudoinverse(self):
        try:
            return self._pseudoinverse
        except AttributeError:
            self._pseudoinverse = np.linalg.pinv(self.rproj_op.components_.T)
            return self._pseudoinverse

    @property
    def singular_values_(self):
        return self.pca_op.singular_values_

    @property
    def explained_variance_(self):
        return self.pca_op.explained_variance_

    @property
    def explained_variance_ratio_(self):
        return self.pca_op.explained_variance_ratio_

    def fit(self, X):
        X_rproj = self.rproj_op.fit_transform(X)
        self.pca_op.fit(X_rproj)

    def transform(self, X):
        X_rproj = self.rproj_op.transform(X)
        X_pca = self.pca_op.transform(X_rproj)
        return X_pca

    def fit_transform(self, X):
        X_rproj = self.rproj_op.fit_transform(X)
        X_pca = self.pca_op.fit_transform(X_rproj)
        return X_pca

    def inverse_transform(self, X):
        X_rproj = self.pca_op.inverse_transform(X)
        X_ambient = X_rproj.dot(self.pseudoinverse)
        return X_ambient


def pca(data, n_components=100, eps=0.1, seed=None,
        n_pca=None, svd_offset=None, svd_multiples=None):
    """Calculate randomized PCA using SVD first to handle sparse matrices

    Compute `svd_multiples * n_pca` SVD components to compute before
    subtracting the mean when data is sparse.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    n_components : int, optional (default: 100)
        Number of PCs to compute
    eps : strictly positive float, optional (default=0.1)
        Control the quality of the embedding. Smaller values lead
        to better embeddings at the cost of runtime and memory.
    seed : int, RandomState or None, optional (default: None)
        Random state.
    n_pca : Deprecated
    svd_offset : Deprecated
    svd_multiples :Deprecated

    Returns
    -------
    data_pca : array-like, shape=[n_samples, n_pca]
        PCA reduction of `data`
    """
    if n_pca is not None:
        warnings.warn("n_pca is deprecated. Use n_components in future.",
                      FutureWarning)
    if svd_offset is not None:
        warnings.warn("svd_offset is deprecated. Use eps in future.",
                      FutureWarning)
    if svd_multiples is not None:
        warnings.warn("svd_multiples is deprecated. Use eps in future.",
                      FutureWarning)

    if not 0 <= n_components <= min(data.shape):
        raise ValueError("n_components={} must be between 0 and "
                         "min(n_samples, n_features)={}".format(
                             n_components, min(data.shape)))

    # handle dataframes
    if isinstance(data, pd.SparseDataFrame):
        data = data.to_coo()
    elif isinstance(data, pd.DataFrame):
        data = data.values

    # handle sparsity
    if sparse.issparse(data):
        data = SparseFriendlyPCA(
            n_components=n_components, eps=eps,
            random_state=seed).fit_transform(data)
    else:
        data = decomposition.PCA(n_pca, random_state=seed).fit_transform(data)
    return data
