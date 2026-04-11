import numpy as np
from scipy.sparse import issparse

def PCA(count_matrix, k, scale=False):
    """
    PCA for a matrix with rows = genes, columns = cells.

    Returns
    -------
    cell_embeddings : ndarray, shape (num_cells, k)
        Coordinates of each cell in PC space.
    pc_loadings : ndarray, shape (num_genes, k)
        Top-k principal directions (eigenvectors).
    explained_variance : ndarray, shape (k,)
        Variance explained by each selected PC.
    explained_variance_ratio : ndarray, shape (k,)
        Fraction of total variance explained by each selected PC.
    """
    X = count_matrix

    if issparse(X):
        X = X.toarray()
    else:
        X = np.array(X, dtype=float)

    # Convert from genes x cells to cells x genes
    X = X.T

    # Center each gene (column) across cells
    X_centered = X - np.mean(X, axis=0, keepdims=True)

    # Optional: standardize each gene
    if scale:
        std = np.std(X_centered, axis=0, ddof=1, keepdims=True)
        std[std == 0] = 1.0
        X_centered = X_centered / std

    # Covariance matrix of genes/features
    # cov_matrix = np.cov(X_centered, rowvar=False) (np.cov center it for us. )
    # S = X_c.T @ X_c / (N-1)
    cov_matrix = X_centered.T @ X_centered / (X_centered.shape[0] - 1)

    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Keep top k
    explained_variance = eigenvalues[:k]
    pc_loadings = eigenvectors[:, :k]

    # Project cells onto top k PCs
    cell_embeddings = X_centered @ pc_loadings

    """
    # Explained variance ratio
    total_variance = np.sum(eigenvalues)
    explained_variance_ratio = explained_variance / total_variance
    """
    return cell_embeddings, pc_loadings