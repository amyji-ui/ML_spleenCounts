import numpy as np

def PCA(count_matrix, k):
    x_centered = CentreFeatures(count_matrix)

    # I tried to make svd from scratch but my brain explodes so for this we use numpy function for now. 
    U, S, Vt = np.linalg.svd(x_centered, full_matrices=False)

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

'''
Create the Diagonal matrix. This is how much we 'scale' the vectors. 
'''
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