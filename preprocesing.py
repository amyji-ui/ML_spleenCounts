import pandas as pd
import numpy as np
import scanpy as sc
from scipy.sparse import csr_matrix, save_npz, load_npz

# Extract spleen annotations from annotations_facs.csv
def extract_spleen_rows(in_path, out_path, tissue="Spleen"):
    df = pd.read_csv(in_path, low_memory=False)
    selected_rows = df[df["tissue"] == tissue]
    selected_rows.to_csv(out_path, index=False)
    return selected_rows

extract_spleen_rows("data/annotations_facs.csv","data/annotations_facs_spleen.csv","Spleen")

# ====== Load spleen count data and convert to a sparse matrix ====
df = pd.read_csv("data/Spleen-counts.csv")
df = df.set_index(df.columns[0])

gene_ids = df.index.to_numpy()
cell_barcodes = df.columns.to_numpy()

X = df.apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
X_csr = csr_matrix(X)

save_npz("data/Spleen-counts_sparse.npz", X_csr)
np.save("data/gene_ids.npy", gene_ids)
np.save("data/cell_barcodes.npy", cell_barcodes)

# load
sparse_gene_by_cell = load_npz("data/Spleen-counts_sparse.npz")
gene_ids2 = np.load("data/gene_ids.npy", allow_pickle=True).astype(str)
cell_barcodes2 = np.load("data/cell_barcodes.npy", allow_pickle=True).astype(str)

#============ Normalization ===============
adata = sc.AnnData(sparse_gene_by_cell.T)
adata.obs_names = cell_barcodes2
adata.var_names = gene_ids2

# Saving count data
adata.layers["counts"] = adata.X.copy()

# Normalizing to median total counts
sc.pp.normalize_total(adata, target_sum=1e6)
# Logarithmize the data
sc.pp.log1p(adata)

#========= Highly Variable Genes (HVG) ===============
sc.pp.highly_variable_genes(adata, flavor = "seurat_v3", n_top_genes= 3000 , layer="counts")

#========= Principle Component Analysis (PCA) =============
