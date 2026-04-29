import numpy as np
import matplotlib as plt

"""
A simple version of SVM(support vector machine):
- Layer 1: Binary SVM, y=+1/-1
- Layer 2: Train SVM for each class (B cell, T cell, Macro), compare target class vs the rest. So this
remains binary.
- Apply hinge loss with adjustable weight to punish missclassified points.

Reference:
- https://www.geeksforgeeks.org/machine-learning/implementing-svm-from-scratch-in-python/
- https://www.kaggle.com/code/prabhat12/svm-from-scratch

"""


class BinarySVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000, class_weight=None):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.class_weight = class_weight  # dict like {-1: w_neg, 1: w_pos} or None
        self.w = None
        self.b = None

    def fit(self, X, y):
        """
        X: shape (n_samples, n_features)
        y: shape (n_samples,), values must be in {-1, +1}
        """
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features, dtype=float)
        self.b = 0.0

        for _ in range(self.n_iters):
            for i, Xi in enumerate(X):
                yi = y[i]

                # optional class weighting
                sample_weight = 1.0
                if self.class_weight is not None:
                    sample_weight = self.class_weight.get(int(yi), 1.0)

                margin = yi * (np.dot(Xi, self.w) - self.b)
                """
                - If margin is satisfied: yi(w^T * x_i - b) >= 1, no hinge-loss, w <- w - a(2λw)
                - If margin is violated: yi(w^T * x_i - b) < 1, yes hinge-loss, w <- w - a(2λw - yi*xi), b <- b - a*yi
                """
                if margin >= 1:
                    # only regularization term contributes
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    # regularization + weighted hinge-loss subgradient
                    self.w -= self.lr * (2 * self.lambda_param * self.w - sample_weight * yi * Xi)
                    self.b -= self.lr * (sample_weight * yi)

        return self

    def decision_function(self, X):
        return np.dot(X, self.w) - self.b

    def predict(self, X):
        scores = self.decision_function(X)
        return np.sign(scores)


class MulticlassSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000, class_weight=None):
        """
        class_weight:
            None
            or "balanced"
            or dict mapping original class label -> weight
        """
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.class_weight = class_weight
        self.classifiers = {}
        self.classes_ = None

    def _compute_ovr_weights(self, y, cls):
        """
        Build binary class weights for one-vs-rest:
        +1 = current class
        -1 = all other classes
        """
        y_binary = np.where(y == cls, 1, -1)

        if self.class_weight is None:
            return y_binary, None

        if self.class_weight == "balanced":
            n_total = len(y_binary)
            n_pos = np.sum(y_binary == 1)
            n_neg = np.sum(y_binary == -1)

            # balanced heuristic similar in spirit to sklearn
            w_pos = n_total / (2.0 * n_pos) if n_pos > 0 else 1.0
            w_neg = n_total / (2.0 * n_neg) if n_neg > 0 else 1.0
            return y_binary, {1: w_pos, -1: w_neg}

        if isinstance(self.class_weight, dict):
            # use original multiclass weights, then convert to OvR weights
            pos_weight = self.class_weight.get(cls, 1.0)

            other_classes = [c for c in np.unique(y) if c != cls]
            if len(other_classes) == 0:
                neg_weight = 1.0
            else:
                neg_weight = float(np.mean([self.class_weight.get(c, 1.0) for c in other_classes]))

            return y_binary, {1: pos_weight, -1: neg_weight}

        raise ValueError("class_weight must be None, 'balanced', or a dict")

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.classifiers = {}

        for cls in self.classes_:
            y_binary, binary_weights = self._compute_ovr_weights(y, cls)

            clf = BinarySVM(
                learning_rate=self.lr,
                lambda_param=self.lambda_param,
                n_iters=self.n_iters,
                class_weight=binary_weights,
            )
            clf.fit(X, y_binary)
            self.classifiers[cls] = clf

        return self

    def decision_function(self, X):
        """
        Returns score matrix of shape (n_samples, n_classes)
        """
        scores = np.column_stack([
            self.classifiers[cls].decision_function(X)
            for cls in self.classes_
        ])
        return scores

    def predict(self, X):
        scores = self.decision_function(X)
        best_class_indices = np.argmax(scores, axis=1)
        return self.classes_[best_class_indices]

# =============== Helper functions =====================
# this should probably go to a util.py but I'm too lazy.
from pathlib import Path
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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