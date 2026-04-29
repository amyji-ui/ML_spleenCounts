from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from SVM_scratch import MulticlassSVM

def fit_scratch_svm_and_save(
    X_train,
    y_train,
    X_test,
    y_test,
    learning_rate=0.001,
    lambda_param=0.01,
    n_iters=1000,
    class_weight="balanced",
    output_dir="output",
    model_name="scratch_svm_model.joblib",
    summary_name="scratch_svm_summary.txt",
):
    """
    Train scratch multiclass linear SVM (OvR), evaluate, and save outputs.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = MulticlassSVM(
        learning_rate=learning_rate,
        lambda_param=lambda_param,
        n_iters=n_iters,
        class_weight=class_weight,
    )
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    report = classification_report(y_test, y_test_pred, digits=4)
    cm = confusion_matrix(y_test, y_test_pred, labels=model.classes_)

    model_path = output_dir / model_name
    joblib.dump(model, model_path)

    summary_path = output_dir / summary_name
    with open(summary_path, "w") as f:
        f.write("Scratch Multiclass SVM Summary\n")
        f.write("==============================\n\n")
        f.write(f"learning_rate: {learning_rate}\n")
        f.write(f"lambda_param: {lambda_param}\n")
        f.write(f"n_iters: {n_iters}\n")
        f.write(f"class_weight: {class_weight}\n")
        f.write(f"n_train_samples: {X_train.shape[0]}\n")
        f.write(f"n_test_samples: {X_test.shape[0]}\n")
        f.write(f"n_features: {X_train.shape[1]}\n")
        f.write(f"classes: {list(model.classes_)}\n\n")

        f.write(f"Train accuracy: {train_acc:.4f}\n")
        f.write(f"Test accuracy: {test_acc:.4f}\n\n")

        f.write("Classification report (test set):\n")
        f.write("---------------------------------\n")
        f.write(report)
        f.write("\n\n")

        f.write("Confusion matrix (test set):\n")
        f.write("----------------------------\n")
        f.write(f"Labels order: {list(model.classes_)}\n")
        f.write(np.array2string(cm))
        f.write("\n")

    return {
        "model": model,
        "model_path": str(model_path),
        "summary_path": str(summary_path),
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "classification_report": report,
        "confusion_matrix": cm,
        "y_test_pred": y_test_pred,
    }