from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from script.KFoldCrossVal import run_kfold_cross_val
from script.prf import precision_recall_f1, save_evaluation_plots
from pathlib import Path

def fit_multinomial_logistic(X_train, y_train, X_test, y_test, max_iter=5000, C=1.0, n_cv_folds=5, output_dir="output",summary_name="multinomial_logistic_summary.txt"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cv = run_kfold_cross_val(
        clf_factory=lambda: LogisticRegression(solver='lbfgs', max_iter=max_iter, C=C, random_state=42),
        X_train=X_train,
        y_train=y_train,
        n_cv_folds=n_cv_folds
    )
    mlr = LogisticRegression(solver='lbfgs', max_iter=max_iter, C=C, random_state=42)
    mlr.fit(X_train, y_train)

    y_test_pred = mlr.predict(X_test)
    y_probas = mlr.predict_proba(X_test)
    train_accuracy = mlr.score(X_train, y_train)
    test_accuracy = mlr.score(X_test, y_test)

    classes = mlr.classes_
    cm = confusion_matrix(y_test, y_test_pred, labels=classes)
    prf = precision_recall_f1(cm, classes)

    save_evaluation_plots(
        cm=cm,
        prf=prf,
        probas=y_probas,
        y_true=y_test,
        classes=classes,
        output_path="output/multinomial_logistic_evaluation.png",
        prefix="Multinomial Logistic"
    )

    summary_path = output_dir / summary_name
    with open(summary_path, "w") as f:
        f.write("Multinomial Logistic Regression Summary\n")
        f.write("==============================\n\n")
        f.write(f"C: {C}\n")
        f.write(f"max_iter: {max_iter}\n")
        f.write(f"n_cv_folds: {n_cv_folds}\n")
        f.write(f"n_train_samples: {X_train.shape[0]}\n")
        f.write(f"n_test_samples: {X_test.shape[0]}\n")
        f.write(f"n_features: {X_train.shape[1]}\n")
        f.write(f"classes: {list(classes)}\n\n")

        f.write(f"CV mean: {cv['cv_mean']:.4f}\n")
        f.write(f"CV std: {cv['cv_std']:.4f}\n\n")

        f.write(f"Train accuracy: {train_accuracy:.4f}\n")
        f.write(f"Test accuracy: {test_accuracy:.4f}\n\n")

        f.write("Classification report (test set):\n")
        f.write("---------------------------------\n")
        f.write(classification_report(y_test, y_test_pred, digits=4))
        f.write("\n\n")

        f.write("Confusion matrix (test set):\n")
        f.write("----------------------------\n")
        f.write(f"Labels order: {list(classes)}\n")
        f.write(np.array2string(cm))
        f.write("\n")

    return dict(
        model=mlr,
        y_pred=y_test_pred,
        y_probas=y_probas,
        test_accuracy=test_accuracy,
        cv_mean=cv["cv_mean"],
        cv_std=cv["cv_std"],
        classification_report=classification_report(y_test, y_test_pred, digits=4),
        cm=cm,
        confusion_matrix=cm
    )



