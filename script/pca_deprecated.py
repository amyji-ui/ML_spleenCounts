import numpy as np

def PCA(count_matrix, k):
    x_centered = CentreFeatures(count_matrix)

    U, S, Vt = svd(x_centered)

    Uk = U[:, 0:k] # gene x k
    Sk = S[0:k] # length k
    Vtk = Vt[0:k, :] # k x cell

    Sigma_k = CreateDiagonalMatrix(Sk)

    cell_scores = Sigma_k @ Vtk
    cell_embeddings = cell_scores.T

    all_variance = []
    num_cells = len(x_centered[0])

    for s in S:
        variance = (s * s) / (num_cells - 1)
        all_variance.append(variance)

    total_variance = sum(all_variance)

    explained_variance = all_variance[:k]
    explained_variance_ratio = []

    for v in explained_variance:
        explained_variance_ratio.append(v / total_variance)

    # Technically only cell embedding is necessary but we may need other output like variance for downstream analysis. 
    return cell_embeddings, Uk, explained_variance, explained_variance_ratio
    
# QuickNote: if the matrix is big, this function will slow down the process. We can replace this with numpy function if necessary.
'''
Numpy version:
def CentreFeatures(X):
    X = np.asarray(X, dtype=float)
    gene_means = np.mean(X, axis=1, keepdims=True)
    X_centered = X - gene_means
    return X_centered, gene_means
'''
'''
Assuming that each row is a gene and each column is a cell. 
For each gene j, compute its mean across all cells.
Center each count according to the mean and update the count matrix.
'''
def CentreFeatures(count_matrix):
    n = len(count_matrix)
    m = len(count_matrix[0])

    gene_means = []
    for i in range(n):
        total = 0
        for j in range(m):
            count = count_matrix[i][j]
            total += count
        mean = total/m
        gene_means.append(mean)
    
    for i in range(n):
        for j in range(m):
            val_c = count_matrix[i][j] - gene_means[i]
            count_matrix[i][j] = val_c
    
    return np.asarray(count_matrix, dtype=float)


def svd(X, tol=1e-10):
    """
    Compute reduced SVD of X using eigendecomposition of X.T @ X.

    U : ndarray, shape (m, r)
        Left singular vectors.
    S : ndarray, shape (r,)
        Singular values in descending order.
    Vt : ndarray, shape (r, n)
        Right singular vectors transposed.
    """
 
    X = np.asarray(X, dtype=float)

    # form X^T X
    XtX = X.T @ X

    # eigendecomposition of X^T X. eigvals.shape = n; V.shape() = n*n
    # Right singular vectors are just the eigenvector of X^TX
    eigvals, V = np.linalg.eigh(XtX) # use eigh because X^TX is symetric.

    # sort eigenvalues/vectors in descending order
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    V = V[:, idx]

    # remove tiny negative values caused by numerical error
    eigvals[eigvals < 0] = 0.0

    # singular values is the square root of eigenvalues.
    S_all = np.sqrt(eigvals)

    # keep only nonzero singular values
    keep = S_all > tol
    S = S_all[keep]
    V = V[:, keep]

    # compute Left singular vectors U from u_i = ( X * v_i ) / sigma_i
    U_cols = []
    for i in range(len(S)):
        vi = V[:, i] # column i of V is v_i
        sigma = S[i] # item i of S is sigma_i
        ui = X @ vi / sigma 
        ui = ui / np.linalg.norm(ui)
        U_cols.append(ui)

    if len(U_cols) == 0:
        U = np.empty((X.shape[0], 0))
        Vt = np.empty((0, X.shape[1]))
    else:
        U = np.column_stack(U_cols)
        Vt = V.T

    return U, S, Vt

def CreateDiagonalMatrix(values):
    k = len(values)
    Sigma = []
    for i in range(k):
        row = []
        for j in range(k):
            if i == j:
                row.append(values[i])
            else:
                row.append(0.0)
        Sigma.append(row)
    return np.asarray(Sigma, dtype=float)

test_matrix = [
    [5, 4, 3, 2],   # gene 1
    [1, 2, 3, 4],   # gene 2
    [2, 2, 2, 2]    # gene 3
]

cell_embeddings, gene_loadings, ev, evr = PCA(test_matrix, 2)

print("Cell embeddings:")
print(cell_embeddings)

print("\nGene loadings:")
print(gene_loadings)

print("\nExplained variance:")
print(ev)

print("\nExplained variance ratio:")
print(evr)