from script.preprocessing import run_preprocessing_pipeline

def main():
    results = run_preprocessing_pipeline(
        annotation_input_full="data/annotations_facs.csv",
        annotation_output_tissue="data/annotations_facs_spleen.csv",
        count_input="data/Spleen-counts.csv",
        sparse_output="data/Spleen-counts_sparse.npz",
        gene_id_output="data/gene_ids.npy",
        cell_barcode_output="data/cell_barcodes.npy",
        tissue="Spleen",
        barcode_col="cell",
        label_col="cell_ontology_class",
        test_size=0.2,
        random_state=42,
    )

    X_train = results["X_train"]
    X_test = results["X_test"]
    y_train = results["y_train"]
    y_test = results["y_test"]

    print("Main file:")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

if __name__ == "__main__":
    main()