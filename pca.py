def PCA(count_matrix, k):
    x_centered = CentreFeatures(count_matrix)

    U, S, Vt = SVD(x_centered)

    Uk = U[:, 0:k]
    Sk = S[0:k]
    Vtk = Vt[0:k, :]

    Sigma_k = DiagonalMatrix(Sk)

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

    return cell_embeddings, Uk, explained_variance, explained_variance_ratio
    


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
    
    return count_matrix

'''
Singular Value Decomposition
'''
def SVD():
    return


'''
Create the Diagonal matrix
'''
def DiagonalMatrix():
    return