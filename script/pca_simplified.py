import numpy as np
from scipy.sparse import issparse

def PCA_svd(X, k, scale=False):
    """
    PCA for input matrix with rows = cells, columns = genes.
    Uses SVD directly on the centered data matrix.

    Returns
    -------
    cell_embeddings : (n_cells, k)
    pc_loadings : (n_genes, k)
    explained_variance : (k,)
    explained_variance_ratio : (k,)
    """
    if issparse(X):
        X = X.toarray()
    else:
        X = np.asarray(X, dtype=float)

    # center each gene across cells
    X_centered = X - np.mean(X, axis=0, keepdims=True)

    # optional standardization
    if scale:
        std = np.std(X_centered, axis=0, ddof=1, keepdims=True)
        std[std == 0] = 1.0
        X_centered = X_centered / std

    # SVD: X = U S Vt
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # keep top k
    U_k = U[:, :k]
    S_k = S[:k]
    Vt_k = Vt[:k, :]

    # cell embeddings = U_k * S_k
    cell_embeddings = U_k * S_k

    # gene loadings = columns are principal directions
    pc_loadings = Vt_k.T

    # explained variance from singular values
    n_samples = X_centered.shape[0]
    explained_variance = (S_k ** 2) / (n_samples - 1)

    all_explained_variance = (S ** 2) / (n_samples - 1)
    explained_variance_ratio = explained_variance / np.sum(all_explained_variance)

    return cell_embeddings, pc_loadings, explained_variance, explained_variance_ratio

test_matrix = [
    [5, 4, 3, 2],   # cell1
    [1, 2, 3, 4],   # cell2
    [2, 2, 2, 2]    # cell3
]

#cell_embeddings = PCA(test_matrix, 2)
#print(cell_embeddings)

