from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from script.KFoldCrossVal import run_kfold_cross_val
from script.prf import precision_recall_f1, save_evaluation_plots


def fit_random_forest(
    X_train,
    y_train,
    X_test,
    y_test,
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_cv_folds=5,
    output_dir="output",
    summary_name="random_forest_summary.txt",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # perform k-fold cross-validation on training set
    cv = run_kfold_cross_val(
        clf_factory=lambda: RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight=class_weight,
            n_jobs=-1
        ),
        X_train=X_train,
        y_train=y_train,
        n_cv_folds=n_cv_folds
    )

    # train final model on full training set
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight=class_weight,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # evaluate on test set
    y_test_pred = clf.predict(X_test)
    y_probas = clf.predict_proba(X_test)
    train_accuracy = clf.score(X_train, y_train)
    test_accuracy = clf.score(X_test, y_test)

    # generate evaluation report and plots
    cm = confusion_matrix(y_test, y_test_pred, labels=clf.classes_)
    prf = precision_recall_f1(cm, clf.classes_)
    report = classification_report(y_test, y_test_pred, digits=4)

    # save thee evaluation plots
    save_evaluation_plots(
        cm=cm,
        prf=prf,
        probas=y_probas,
        y_true=y_test,
        classes=clf.classes_,
        output_path="qmetric/random_forest_evaluation.png",
        prefix="Random Forest"
    )

    summary_path = output_dir / summary_name
    with open(summary_path, "w") as f:
        f.write("Random Forest Summary\n")
        f.write("==============================\n\n")
        f.write(f"n_estimators: {n_estimators}\n")
        f.write(f"random_state: {random_state}\n")
        f.write(f"class_weight: {class_weight}\n")
        f.write(f"n_cv_folds: {n_cv_folds}\n")
        f.write(f"n_train_samples: {X_train.shape[0]}\n")
        f.write(f"n_test_samples: {X_test.shape[0]}\n")
        f.write(f"n_features: {X_train.shape[1]}\n")
        f.write(f"classes: {list(clf.classes_)}\n\n")

        f.write(f"CV mean: {cv['cv_mean']:.4f}\n")
        f.write(f"CV std: {cv['cv_std']:.4f}\n\n")

        f.write(f"Train accuracy: {train_accuracy:.4f}\n")
        f.write(f"Test accuracy: {test_accuracy:.4f}\n\n")

        f.write("Classification report (test set):\n")
        f.write("---------------------------------\n")
        f.write(report)
        f.write("\n\n")

        f.write("Confusion matrix (test set):\n")
        f.write("----------------------------\n")
        f.write(f"Labels order: {list(clf.classes_)}\n")
        f.write(np.array2string(cm))
        f.write("\n")

    # return the trained model and evaluation results
    return dict(
        model=clf,
        y_pred=y_test_pred,
        y_probas=y_probas,
        train_accuracy=train_accuracy,
        test_accuracy=test_accuracy,
        cv_mean=cv["cv_mean"],
        cv_std=cv["cv_std"],
        classification_report=report,
        confusion_matrix=cm
    )