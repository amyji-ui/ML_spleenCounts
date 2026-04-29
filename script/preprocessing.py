import pandas as pd
import numpy as np
import scanpy as sc
from scipy.sparse import csr_matrix, save_npz, load_npz
from sklearn.model_selection import train_test_split


def extract_tissue_rows(in_path, out_path, tissue="Spleen"):
    """
    Extract rows for one tissue from the annotation file and save them.
    Only needs to be run once!
    """
    df = pd.read_csv(in_path, low_memory=False)
    selected_rows = df[df["tissue"] == tissue].copy()
    selected_rows.to_csv(out_path, index=False)
    return selected_rows


def load_count_matrix_as_sparse(input_count, output_sparse, output_gene_id, output_cell_barcode):
    """
    Load gene-by-cell count matrix from CSV, convert to sparse format, and save
    sparse matrix plus gene IDs and cell barcodes.
    """
    df = pd.read_csv(input_count)
    df = df.set_index(df.columns[0])

    gene_ids = df.index.to_numpy().astype(str)
    cell_barcodes = df.columns.to_numpy().astype(str)

    X_raw = df.apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
    X_csr = csr_matrix(X_raw)

    save_npz(output_sparse, X_csr)
    np.save(output_gene_id, gene_ids)
    np.save(output_cell_barcode, cell_barcodes)

    return X_csr, gene_ids, cell_barcodes


def load_sparse_count_data(output_sparse, output_gene_id, output_cell_barcode):
    """
    Reload saved sparse count matrix, gene IDs, and cell barcodes.
    """
    sparse_gene_by_cell = load_npz(output_sparse)
    gene_ids = np.load(output_gene_id, allow_pickle=True).astype(str)
    cell_barcodes = np.load(output_cell_barcode, allow_pickle=True).astype(str)

    return sparse_gene_by_cell, gene_ids, cell_barcodes


def create_normalized_adata(sparse_gene_by_cell, gene_ids, cell_barcodes, target_sum=1e6):
    """
    Create AnnData object with rows=cells, columns=genes, save raw counts,
    and perform normalization + log transform.
    """
    adata = sc.AnnData(sparse_gene_by_cell.T)
    adata.obs_names = cell_barcodes
    adata.var_names = gene_ids

    # Save raw counts before normalization
    adata.layers["counts"] = adata.X.copy()

    # Normalize and log-transform
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)

    return adata


def load_annotation_labels(input_annotation, barcode_col=None, label_col=None):
    """
    Load annotation file and keep only barcode + label columns.
    If column names are not provided, use the first two columns.
    """
    ann = pd.read_csv(input_annotation)

    if barcode_col is None or label_col is None:
        ann = ann.iloc[:, :2].copy()
        ann.columns = ["cellbarcode", "cell_type"]
    else:
        ann = ann[[barcode_col, label_col]].copy()
        ann.columns = ["cellbarcode", "cell_type"]

    ann["cellbarcode"] = ann["cellbarcode"].astype(str)
    ann["cell_type"] = ann["cell_type"].astype(str)
    ann = ann.set_index("cellbarcode")

    return ann


def align_adata_and_annotations(adata, ann):
    """
    Keep only cells that appear in both expression data and annotation file,
    and align them in the same order.
    """
    common_barcodes = adata.obs_names.intersection(ann.index)

    adata_aligned = adata[common_barcodes].copy()
    ann_aligned = ann.loc[common_barcodes].copy()

    return adata_aligned, ann_aligned


def build_features_and_labels(adata, ann):
    """
    Build numeric feature matrix X and label vector y from aligned AnnData and annotation table.
    """
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X, dtype=float)
    y = ann.loc[adata.obs_names, "cell_type"].to_numpy()
    barcodes = adata.obs_names.to_numpy()

    return X, y, barcodes


def split_train_test(X, y, barcodes=None, test_size=0.2, random_state=42, stratify=True):
    """
    Split data into train/test sets. Optionally keep barcodes with the split.
    """
    stratify_labels = y if stratify else None

    if barcodes is None:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_labels
        )
        return X_train, X_test, y_train, y_test

    X_train, X_test, y_train, y_test, bc_train, bc_test = train_test_split(
        X, y, barcodes, test_size=test_size, random_state=random_state, stratify=stratify_labels
    )
    return X_train, X_test, y_train, y_test, bc_train, bc_test


def print_basic_checks(X, y, barcodes):
    """
    Print basic sanity checks.
    """
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("First 5 cell barcodes:", barcodes[:5].tolist())
    print("First 5 labels:", y[:5])


def run_preprocessing_pipeline(
    annotation_input_full,
    annotation_output_tissue,
    count_input,
    sparse_output,
    gene_id_output,
    cell_barcode_output,
    tissue="Spleen",
    barcode_col=None,
    label_col=None,
    test_size=0.2,
    random_state=42,
):
    """
    Full preprocessing pipeline up to train/test split.
    """
    # 1. Extract tissue-specific annotations
    extract_tissue_rows(annotation_input_full, annotation_output_tissue, tissue=tissue)

    # 2. Load count matrix and save sparse outputs
    load_count_matrix_as_sparse(
        count_input,
        sparse_output,
        gene_id_output,
        cell_barcode_output
    )

    # 3. Reload sparse data
    sparse_gene_by_cell, gene_ids, cell_barcodes = load_sparse_count_data(
        sparse_output,
        gene_id_output,
        cell_barcode_output
    )

    # 4. Create and normalize AnnData
    adata = create_normalized_adata(sparse_gene_by_cell, gene_ids, cell_barcodes)

    # 5. Load annotation labels
    ann = load_annotation_labels(
        annotation_output_tissue,
        barcode_col=barcode_col,
        label_col=label_col
    )

    # 6. Align data and annotations
    adata, ann = align_adata_and_annotations(adata, ann)

    # 7. Build X and y
    X, y, barcodes = build_features_and_labels(adata, ann)

    # 8. Sanity checks
    print_basic_checks(X, y, barcodes)

    # 9. Split train/test
    X_train, X_test, y_train, y_test, bc_train, bc_test = split_train_test(
        X, y, barcodes=barcodes, test_size=test_size, random_state=random_state, stratify=True
    )

    print("Train shape:", X_train.shape, y_train.shape)
    print("Test shape:", X_test.shape, y_test.shape)

    return {
        "adata": adata,
        "ann": ann,
        "X": X,
        "y": y,
        "barcodes": barcodes,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "bc_train": bc_train,
        "bc_test": bc_test,
    }
