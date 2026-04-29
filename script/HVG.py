import numpy as np

def select_hvg_by_variance(X_train, n_top_genes=3000):
    """
    Select highly variable genes using variance computed on training data only.

    Parameters
    ----------
    X_train : ndarray, shape (n_cells, n_genes)
    n_top_genes : int

    Returns
    -------
    selected_idx : ndarray, shape (n_top_genes,)
        Indices of selected genes.
    gene_variances : ndarray, shape (n_genes,)
        Variance of each gene in training data.
    """
    X_train = np.asarray(X_train, dtype=float)

    gene_variances = np.var(X_train, axis=0, ddof=1)
    selected_idx = np.argsort(gene_variances)[::-1][:n_top_genes]

    return selected_idx, gene_variances


def apply_hvg_selection(X, selected_idx):
    """
    Apply previously selected gene indices to any dataset.
    """
    X = np.asarray(X, dtype=float)
    return X[:, selected_idx]