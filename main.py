from script.preprocessing import run_preprocessing_pipeline
from script.HVG import select_hvg_by_variance, apply_hvg_selection
from script.pca_simplified import PCA_svd
from script.SVM import fit_svm_and_save
from script.kNN import fit_knn
from script.kNN_pack import fit_knn_pack

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

    # 1. HVG on training set only
    hvg_idx, gene_var = select_hvg_by_variance(X_train, n_top_genes=3000)

    X_train_hvg = apply_hvg_selection(X_train, hvg_idx)
    X_test_hvg = apply_hvg_selection(X_test, hvg_idx)

    print("X_train_hvg shape:", X_train_hvg.shape)
    print("X_test_hvg shape:", X_test_hvg.shape)

    # 2. PCA on HVG-filtered data
    X_train_pca, train_loadings, _, _ = PCA_svd(X_train_hvg, 50, scale=False)

    # IMPORTANT:
    # Use train mean + train loadings to transform test data.
    train_mean = X_train_hvg.mean(axis=0, keepdims=True)
    X_test_centered = X_test_hvg - train_mean
    X_test_pca = X_test_centered @ train_loadings

    print("X_train_pca shape:", X_train_pca.shape)
    print("X_test_pca shape:", X_test_pca.shape)

    # Run svm (package version)
    svm_results = fit_svm_and_save(
        X_train=X_train_pca,
        y_train=y_train,
        X_test=X_test_pca,
        y_test=y_test,
        kernel="rbf",
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        output_dir="output",
        model_name="svm_model.joblib",
        summary_name="svm_summary.txt",
    )

    print("Saved model to:", svm_results["model_path"])
    print("Saved summary to:", svm_results["summary_path"])
    print("Test accuracy:", svm_results["test_accuracy"])

    # Run kNN (scratch implementation)
    knn_results = fit_knn(
        X_train=X_train_pca,
        y_train=y_train,
        X_test=X_test_pca,
        y_test=y_test,
        k=7,
        metric="euclidean",
        n_cv_folds=5,
    )
 
    print("KNN Test accuracy:", knn_results["test_accuracy"])
    print("KNN Classification Report:")
    print(knn_results["classification_report"])
    print("KNN Confusion Matrix:")
    print(knn_results["confusion_matrix"])
    print("KNN CV mean ± std:", f"{knn_results['cv_mean']:.4f} ± {knn_results['cv_std']:.4f}")

    # Run kNN (package version)
    knn_pack_results = fit_knn_pack(
        X_train=X_train_pca,
        y_train=y_train,
        X_test=X_test_pca,
        y_test=y_test,
        k=7
    )
    print("KNN Pack Test accuracy:", knn_pack_results["test_accuracy"])
    print("KNN Pack Classification Report:")
    print(knn_pack_results["classification_report"])
    print("KNN Pack Confusion Matrix:")
    print(knn_pack_results["confusion_matrix"])

if __name__ == "__main__":
    main()