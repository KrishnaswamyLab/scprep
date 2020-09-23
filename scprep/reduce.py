from sklearn import decomposition, random_projection
import sklearn.base
import pandas as pd
from scipy import sparse
import numpy as np
import warnings

from . import utils


class InvertibleRandomProjection(random_projection.GaussianRandomProjection):
    """Gaussian random projection with an inverse transform using the pseudoinverse."""

    def __init__(
        self, n_components="auto", eps=0.3, orthogonalize=False, random_state=None
    ):
        self.orthogonalize = orthogonalize
        super().__init__(n_components=n_components, eps=eps, random_state=random_state)

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
                self._pseudoinverse = self.components_
            else:
                self._pseudoinverse = np.linalg.pinv(self.components_.T)
            return self._pseudoinverse

    def fit(self, X):
        super().fit(X)
        if self.orthogonalize:
            Q, _ = np.linalg.qr(self.components_.T)
            self.components_ = Q.T
        return self

    def inverse_transform(self, X):
        return X.dot(self.pseudoinverse)


class AutomaticDimensionSVD(decomposition.TruncatedSVD):
    """Truncated SVD with automatic dimensionality selected by the Johnson-Lindenstrauss lemma."""

    def __init__(
        self,
        n_components="auto",
        eps=0.3,
        algorithm="randomized",
        n_iter=5,
        random_state=None,
        tol=0.0,
    ):
        self.eps = eps
        if n_components == "auto":
            # just pass through -1 - we will change it later
            n_components = -1
        super().__init__(
            n_components=n_components,
            algorithm=algorithm,
            n_iter=n_iter,
            random_state=random_state,
            tol=tol,
        )

    def fit(self, X):
        if self.n_components == -1:
            super().set_params(
                n_components=random_projection.johnson_lindenstrauss_min_dim(
                    n_samples=X.shape[0], eps=self.eps
                )
            )
        try:
            return super().fit(X)
        except ValueError as e:
            if self.n_components >= X.shape[1]:
                raise RuntimeError(
                    "eps={} and n_samples={} lead to a target "
                    "dimension of {} which is larger than the "
                    "original space with n_features={}".format(
                        self.eps, X.shape[0], self.n_components, X.shape[1]
                    )
                )
            else:
                raise


class SparseInputPCA(sklearn.base.BaseEstimator):
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
        Smaller values lead to more accurate embeddings but higher
        computational and memory costs
    method : {'svd', 'orth_rproj', 'rproj'}, optional (default: 'svd')
        Dimensionality reduction method applied prior to mean centering.
        The method choice affects accuracy (`svd` > `orth_rproj` > `rproj`)
        comes with increased computational cost (but not memory.)
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number
        generator; If RandomState instance, random_state is the random number
        generator; If None, the random number generator is the RandomState
        instance used by np.random.
    kwargs
        Additional keyword arguments for `sklearn.decomposition.PCA`
    """

    def __init__(
        self, n_components=2, eps=0.3, random_state=None, method="svd", **kwargs
    ):
        self.pca_op = decomposition.PCA(
            n_components=n_components, random_state=random_state
        )
        if method == "svd":
            self.proj_op = AutomaticDimensionSVD(eps=eps, random_state=random_state)
        elif method == "orth_rproj":
            self.proj_op = InvertibleRandomProjection(
                eps=eps, random_state=random_state, orthogonalize=True
            )
        elif method == "rproj":
            self.proj_op = InvertibleRandomProjection(
                eps=eps, random_state=random_state, orthogonalize=False
            )
        else:
            raise ValueError(
                "Expected `method` in ['svd', 'orth_rproj', 'rproj']. "
                "Got '{}'".format(method)
            )

    @property
    def singular_values_(self):
        """Singular values of the PCA decomposition"""
        return self.pca_op.singular_values_

    @property
    def explained_variance_(self):
        """The amount of variance explained by each of the selected components."""
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

        The components are sorted by explained variance.
        """
        return self.proj_op.inverse_transform(self.pca_op.components_)

    def _fit(self, X):
        self.proj_op.fit(X)
        X_proj = self.proj_op.transform(X)
        self.pca_op.fit(X_proj)
        return X_proj

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
        X_proj = self.proj_op.transform(X)
        X_pca = self.pca_op.transform(X_proj)
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
        X_proj = self._fit(X)
        X_pca = self.pca_op.transform(X_proj)
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
        X_proj = self.pca_op.inverse_transform(X)
        X_ambient = self.proj_op.inverse_transform(X_proj)
        return X_ambient


def pca(
    data,
    n_components=100,
    eps=0.3,
    method="svd",
    seed=None,
    return_singular_values=False,
    n_pca=None,
    svd_offset=None,
    svd_multiples=None,
):
    """Calculate PCA using random projections to handle sparse matrices

    Uses the Johnson-Lindenstrauss Lemma to determine the number of
    dimensions of random projections prior to subtracting the mean.
    Dense matrices are provided to `sklearn.decomposition.PCA` directly.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    n_components : int, optional (default: 100)
        Number of PCs to compute
    eps : strictly positive float, optional (default=0.3)
        Parameter to control the quality of the embedding of sparse input.
        Smaller values lead to more accurate embeddings but higher
        computational and memory costs
    method : {'svd', 'orth_rproj', 'rproj', 'dense'}, optional (default: 'svd')
        Dimensionality reduction method applied prior to mean centering
        of sparse input. The method choice affects accuracy
        (`svd` > `orth_rproj` > `rproj`) and comes with increased
        computational cost (but not memory.) On the other hand,
        `method='dense'` adds a memory cost but is faster.
    seed : int, RandomState or None, optional (default: None)
        Random state.
    return_singular_values : bool, optional (default: False)
        If True, also return the singular values
    n_pca : Deprecated.
    svd_offset : Deprecated.
    svd_multiples :Deprecated.

    Returns
    -------
    data_pca : array-like, shape=[n_samples, n_components]
        PCA reduction of `data`
    singular_values : list-like, shape=[n_components]
        Singular values corresponding to principal components
        returned only if return_values is True
    """
    if n_pca is not None:
        warnings.warn(
            "n_pca is deprecated. Setting n_components={}.".format(n_pca), FutureWarning
        )
        n_components = n_pca
    if svd_offset is not None:
        warnings.warn(
            "svd_offset is deprecated. Please use `eps` instead.", FutureWarning
        )
    if svd_multiples is not None:
        warnings.warn(
            "svd_multiples is deprecated. Please use `eps` instead.", FutureWarning
        )

    if not 0 < n_components <= min(data.shape):
        raise ValueError(
            "n_components={} must be between 0 and "
            "min(n_samples, n_features)={}".format(n_components, min(data.shape))
        )

    # handle dataframes
    if isinstance(data, pd.DataFrame):
        index = data.index
    else:
        index = None
    if method == "dense":
        data = utils.toarray(data)
    else:
        data = utils.to_array_or_spmatrix(data)

    # handle sparsity
    if sparse.issparse(data):
        try:
            pca_op = SparseInputPCA(
                n_components=n_components, eps=eps, method=method, random_state=seed
            )
            data = pca_op.fit_transform(data)
        except RuntimeError as e:
            if "which is larger than the original space" in str(e):
                # eps too small - the best we can do is make the data dense
                return pca(
                    utils.toarray(data),
                    n_components=n_components,
                    seed=seed,
                    return_singular_values=return_singular_values,
                )
    else:
        pca_op = decomposition.PCA(n_components, random_state=seed)
        data = pca_op.fit_transform(data)

    if index is not None:
        data = pd.DataFrame(
            data,
            index=index,
            columns=["PC{}".format(i + 1) for i in range(n_components)],
        )

    if return_singular_values:
        data = (data, pca_op.singular_values_)
    return data
