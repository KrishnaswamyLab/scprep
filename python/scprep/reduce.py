from sklearn import decomposition
import pandas as pd
from scipy import sparse


def pca(data, n_pca=100, svd_offset=100, svd_multiples=3, return_operator=False, seed=None):
    """Calculate randomized PCA using SVD first to handle sparse matrices

    Compute `svd_multiples * n_pca` SVD components to compute before
    subtracting the mean when data is sparse.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data
    n_pca : int, optional (default: 100)
        Number of PCs to compute
    svd_offset : int, optional (default: 100)
        Absolute minimum number of SVD components
    svd_multiples : int, optional (default: 5)
        Multiplies `n_pca` to get number of SVD components
    return_operator : boolean, optional (default: False)
        If true, also returns the operator used for PCA.

    Returns
    -------
    data_pca : array-like, shape=[n_samples, n_pca]
        PCA reduction of `data`
    """
    # handle dataframes
    if isinstance(data, pd.SparseDataFrame):
        data = data.to_coo()
    elif isinstance(data, pd.DataFrame):
        data = data.values

    # handle sparsity
    if sparse.issparse(data):
        n_svd = int(svd_offset + svd_multiples * n_pca)
        n_svd = min(n_svd, data.shape[1] - 1)
        data = decomposition.TruncatedSVD(
            n_svd, random_state=seed).fit_transform(data)
    n_pca = min(n_pca, data.shape[1])
    pca_op = decomposition.PCA(n_pca, random_state=seed)
    data = pca_op.fit_transform(data)
    if return_operator == True:
        return data, pca_op
    else:
        return data
