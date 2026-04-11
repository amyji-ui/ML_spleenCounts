import os
import joblib
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def fit_svm_and_save(
    X_train,
    y_train,
    X_test,
    y_test,
    kernel="rbf",
    C=1.0,
    class_weight="balanced",
    max_iter=1000,
    output_dir="output",
    model_name="svm_model.joblib",
    summary_name="svm_summary.txt",
):
    """
    Fit SVM on training data, save model, evaluate on test data, and save summary.
    """

    os.makedirs(output_dir, exist_ok=True)

    # Fit SVM
    svm = SVC(
        C=C,
        kernel=kernel,
        class_weight=class_weight,
        max_iter=max_iter
    )
    svm.fit(X_train, y_train)

    # Save model
    model_path = os.path.join(output_dir, model_name)
    joblib.dump(svm, model_path)

    # Predict on test data
    y_test_pred = svm.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    report = classification_report(y_test, y_test_pred)
    cm = confusion_matrix(y_test, y_test_pred, labels=svm.classes_)

    # Save summary
    summary_path = os.path.join(output_dir, summary_name)
    with open(summary_path, "w") as f:
        f.write(f"Kernel: {kernel}\n")
        f.write(f"C: {C}\n")
        f.write(f"Class weight: {class_weight}\n")
        f.write(f"Max iterations: {max_iter}\n")
        f.write(f"Classes: {list(svm.classes_)}\n")
        f.write(f"Support vectors per class: {svm.n_support_}\n")
        f.write(f"Test accuracy: {test_acc:.6f}\n\n")

        f.write("Confusion matrix:\n")
        f.write(str(cm))
        f.write("\n\n")

        f.write("Classification report:\n")
        f.write(report)

    return {
        "model": svm,
        "y_test_pred": y_test_pred,
        "test_accuracy": test_acc,
        "model_path": model_path,
        "summary_path": summary_path,
        "confusion_matrix": cm,
    }