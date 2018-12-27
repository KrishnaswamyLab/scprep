from sklearn import decomposition, random_projection
import sklearn.base
import pandas as pd
from scipy import sparse
import numpy as np
import warnings

from . import utils


class SparseFriendlyPCA(sklearn.base.BaseEstimator):
    """Calculate PCA using random projections to handle sparse matrices

    Uses the Johnson-Lindenstrauss Lemma to determine the number of
    dimensions of random projections prior to subtracting the mean.

    Parameters
    ----------
    n_components : int, optional (default: 2)
        Number of components to keep.
    eps : strictly positive float, optional (default=0.15)
        Parameter to control the quality of the embedding according to the
        Johnson-Lindenstrauss lemma when n_components is set to ‘auto’.
        Smaller values lead to better embedding but higher computation and
        memory costs
    orthogonalize : bool, optional (default: False)
        Orthongalize the random projection matrix. If True, this improves the
        accuracy of the embedding at the cost of runtime (but not memory)
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by np.random.
    kwargs
        Additional keyword arguments for `sklearn.decomposition.PCA`
    """

    def __init__(self, n_components=2, eps=0.15,
                 orthogonalize=False, random_state=None,
                 **kwargs):
        self.pca_op = decomposition.PCA(n_components=n_components,
                                        random_state=random_state)
        self.rproj_op = random_projection.GaussianRandomProjection(
            eps=eps, random_state=random_state)
        self.orthogonalize = orthogonalize

    @property
    def pseudoinverse(self):
        """Pseudoinverse of the random projection

        This inverts the projection operation for any vector in the span of the
        random projection. For small enough `eps`, this should be close to the
        correct inverse.
        """
        try:
            return self._pseudoinverse
        except AttributeError:
            if self.orthogonalize:
                # orthogonal matrix: inverse is just its transpose
                self._pseudoinverse = self.rproj_op.components_
            else:
                self._pseudoinverse = np.linalg.pinv(
                    self.rproj_op.components_.T)
            return self._pseudoinverse

    @property
    def singular_values_(self):
        """Singular values of the PCA decomposition
        """
        return self.pca_op.singular_values_

    @property
    def explained_variance_(self):
        """The amount of variance explained by each of the selected components.
        """
        return self.pca_op.explained_variance_

    @property
    def explained_variance_ratio_(self):
        """Percentage of variance explained by each of the selected components.

        The sum of the ratios is equal to 1.0.
        If n_components is `None` then the number of components grows as`eps`
        gets smaller.
        """
        return self.pca_op.explained_variance_ratio_

    @property
    def components_(self):
        """Principal axes in feature space, representing the
        directions of maximum variance in the data.

        The components are sorted by explained_variance_.
        """
        return self.pca_op.components_.dot(self.pseudoinverse)

    def _fit(self, X):
        self.rproj_op.fit(X)
        if self.orthogonalize:
            Q, _ = np.linalg.qr(self.rproj_op.components_.T)
            self.rproj_op.components_ = Q.T
        X_rproj = self.rproj_op.transform(X)
        self.pca_op.fit(X_rproj)
        return X_rproj

    def fit(self, X):
        """Fit the model with X.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
        """
        self._fit(X)
        return self

    def transform(self, X):
        """Apply dimensionality reduction to X.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)

        Returns
        -------
        X_new : array-like, shape=(n_samples, n_components)
        """
        X_rproj = self.rproj_op.transform(X)
        X_pca = self.pca_op.transform(X_rproj)
        return X_pca

    def fit_transform(self, X):
        """Fit the model with X and apply the dimensionality reduction on X.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)

        Returns
        -------
        X_new : array-like, shape=(n_samples, n_components)
        """
        X_rproj = self._fit(X)
        X_pca = self.pca_op.transform(X_rproj)
        return X_pca

    def inverse_transform(self, X):
        """Transform data back to its original space.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_components)

        Returns
        -------
        X_new : array-like, shape=(n_samples, n_features)
        """
        X_rproj = self.pca_op.inverse_transform(X)
        X_ambient = X_rproj.dot(self.pseudoinverse)
        return X_ambient


def pca(data, n_components=100, eps=0.15,
        orthogonalize=False, seed=None,
        n_pca=None, svd_offset=None, svd_multiples=None):
    """Calculate PCA using random projections to handle sparse matrices

    Uses the Johnson-Lindenstrauss Lemma to determine the number of
    dimensions of random projectiosn prior to subtracting the mean.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    n_components : int, optional (default: 100)
        Number of PCs to compute
    eps : strictly positive float, optional (default=0.15)
        Control the quality of the embedding. Smaller values lead
        to better embeddings at the cost of runtime and memory.
    orthogonalize : bool, optional (default: False)
        Orthongalize the random projection matrix. If True, this improves the
        embedding at the cost of runtime (but not memory)
    seed : int, RandomState or None, optional (default: None)
        Random state.
    n_pca : Deprecated.
    svd_offset : Deprecated.
    svd_multiples :Deprecated.

    Returns
    -------
    data_pca : array-like, shape=[n_samples, n_components]
        PCA reduction of `data`
    """
    if n_pca is not None:
        warnings.warn(
            "n_pca is deprecated. Setting n_components={}.".format(n_pca),
            FutureWarning)
        n_components = n_pca
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
        try:
            data = SparseFriendlyPCA(
                n_components=n_components, eps=eps,
                random_state=seed).fit_transform(data)
        except RuntimeError as e:
            if "which is larger than the original space" in str(e):
                # eps too small - the best we can do is make the data dense
                return pca(utils.toarray(data), n_components=n_components,
                           seed=seed)
    else:
        data = decomposition.PCA(n_components,
                                 random_state=seed).fit_transform(data)
    return data
