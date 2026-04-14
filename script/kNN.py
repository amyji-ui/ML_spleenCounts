import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter

class KNNClassifier:
    def __init__(self, k=3, metric='euclidean'):
        if k < 1:
            raise ValueError("k must be >= 1")
        if metric not in ("euclidean", "manhattan"):
            raise ValueError("metric must be 'euclidean' or 'manhattan'")
        self.k = k
        self.metric = metric
        self._X_train = self._y_train = None

    def _euclidean(self, a, b):
        return np.sqrt(np.sum((a - b) ** 2, axis=1))
    
    def _manhattan(self, a, b):
        return np.sum(np.abs(a - b), axis=1)
    
    def _distance(self, query, X):
        if self.metric == "euclidean":
            return self._euclidean(X, query)
        return self._manhattan(X, query)
    
    def fit(self, X, y):
        self._X_train = np.array(X, dtype=float)
        self._y_train = np.array(y)
        self.classes_ = np.unique(self._y_train)
        return self
    
    def predict(self, X):
        self._check_fitted()
        return np.array([self._vote(x) for x in np.asarray(X, float)])
    
    def predict_proba(self, X):
        self._check_fitted()
        return np.array([self._proba(x) for x in np.asarray(X, float)])
    
    def score(self, X, y):
        return np.mean(self.predict(X) == np.array(y))
    
    def _check_fitted(self):
        if self._X_train is None:
            raise RuntimeError("Call fit() first.")

    def _k_nearest(self, x):
        dists = self._distance(x, self._X_train)
        k_idx = np.argsort(dists)[: self.k]
        return self._y_train[k_idx]
    
    def _predict_single(self, x):
        neighbor_labels = self._k_nearest(x)
        most_common = Counter(neighbor_labels).most_common(1)[0][0]
        return most_common
    
    def _vote(self, x):
        return Counter(self._k_nearest(x)).most_common(1)[0][0]
    
    def _proba(self, x):
        counts = Counter(self._k_nearest(x))
        return np.array([counts.get(c, 0) / self.k for c in self.classes_])
    

def confusion_matrix(y_true, y_pred, classes):
    idx = {c: i for i, c in enumerate(classes)}
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1
    return cm

def precision_recall_f1(cm, classes):
    results = {}
    for i, c in enumerate(classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        results[c] = {"precision": p, "recall": r, "f1": f1}
    
    macro_p = np.mean([v["precision"] for v in results.values()])
    macro_r = np.mean([v["recall"] for v in results.values()])
    macro_f1 = np.mean([v["f1"] for v in results.values()])
    results["macro"] = dict(precision=macro_p, recall=macro_r, f1=macro_f1)
    return results

def kfold_cross_val(clf_factory, X, y, k=5, seed=42):
    X, y = np.asarray(X, float), np.asarray(y)
    classes = np.unique(y)
    rng = np.random.default_rng(seed)

    class_indices = {c: rng.permutation(np.where(y == c)[0]) for c in classes}
    folds = [[] for _ in range(k)]
    for indices in class_indices.values():
        for i, idx in enumerate(indices):
            folds[i % k].append(idx)
    
    fold_scores = []
    for fold_idx in range(k):
        test_idx = np.array(folds[fold_idx])
        train_idx = np.concatenate([folds[j] for j in range(k) if j != fold_idx])

        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        mu, sigma = X_tr.mean(0), X_tr.std(0) + 1e-8
        X_tr = (X_tr - mu) / sigma
        X_te = (X_te - mu) / sigma

        clf = clf_factory()
        clf.fit(X_tr, y_tr)
        fold_scores.append(clf.score(X_te, y_te))

    mean = float(np.mean(fold_scores))
    std = float(np.std(fold_scores))
    return fold_scores, mean, std

def _binary_roc(scores, y_binary):
    thresholds = np.sort(np.unique(scores))[::-1]
    tprs, fprs = [0.0], [0.0]
    pos = y_binary.sum()
    neg = len(y_binary) - pos

    for t in thresholds:
        pred = (scores >= t).astype(int)
        tp = ((pred == 1) & (y_binary == 1)).sum()
        fp = ((pred == 1) & (y_binary == 0)).sum()
        tprs.append(tp /pos if pos > 0 else 0.0)
        fprs.append(fp /neg if neg > 0 else 0.0)

    tprs.append(1.0); fprs.append(1.0)
    fprs, tprs = np.array(fprs), np.array(tprs)
    auc = float(np.trapezoid(tprs, fprs))
    return fprs, tprs, auc

def _binary_prc(scores, y_binary):
    thresholds = np.sort(np.unique(scores))[::-1]
    precisions, recalls = [], []

    for t in thresholds:
        pred = (scores >= t).astype(int)
        tp = ((pred == 1) & (y_binary == 1)).sum()
        fp = ((pred == 1) & (y_binary == 0)).sum()
        fn = ((pred == 0) & (y_binary == 1)).sum()
        p = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precisions.append(p)
        recalls.append(r)

    precisions = np.array([1.0] + precisions)
    recalls = np.arra([0.0] + recalls)
    ap = float(np.sum((recalls[1:] - recalls[:-1]) * precisions[1:]))
    return recalls, precisions, ap

def plot_confusion_matrix(cm, classes, ax, title="Confusion Matrix"):
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, fontsize=9)
    ax.set_yticklabels(classes, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    for i in range(len(classes)):
        for j in range(len(classes)):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=11, color=color, fontweight="bold")
    plt.colorbar(im, ax=ax)

def plot_roc_curves(probas, y_true, classes, ax):
    colors = ["#e74c3c", "#2ecc71", "#3498db", "#9b59b6", "#f39c12"]
    for i, c in enumerate(classes):
        y_bin = (y_true == c).astype(int)
        scores = probas[:, i]
        fprs, tprs, auc = _binary_roc(scores, y_bin)
        ax.plot(fprs, tprs, color=colors[i % len(colors)], lw=2,
                label=f"{c} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title("ROC Curves (One-vs-Rest)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])

def plot_prc_curves(probas, y_true, classes, ax):
    colors = ["#e74c3c", "#2ecc71", "#3498db", "#9b59b6", "#f39c12",
              "#1abc9c", "#e67e22", "#34495e"]
    for i, c in enumerate(classes):
        y_bin = (y_true == c).astype(int)
        recalls, precisions, ap = _binary_prc(probas[:, i], y_bin)
        ax.step(recalls, precisions, where="post",
                color=colors[i % len(colors)], lw=2,
                label=f"{c} (AP={ap:.3f})")
    ax.set_xlabel("Recall", fontsize=10)
    ax.set_ylabel("Precision", fontsize=10)
    ax.set_title("Precision-Recall Curves (One-vs-Rest)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])

def plot_prf_bars(prf, classes, ax):
    names = [str(c) for c in classes]
    p_vals = [prf[c]["precision"] for c in classes]
    r_vals = [prf[c]["recall"] for c in classes]
    f1_vals = [prf[c]["f1"] for c in classes]
    x, w = np.arange(len(names)), 0.25
    ax.bar(x - w, p_vals, w, label="Precision", color="#3498db")
    ax.bar(x, r_vals, w, label="Recall", color="#2ecc71")
    ax.bar(x + w, f1_vals, w, label="F1", color="#e74c3c")
    ax.axhline(prf["macro"]["f1"], color="gray", linestyle="--", linewidth=1.2,
               label=f"Macro F1 = {prf['macro']['f1']:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8, rotation=45, ha="right")
    ax.set_ylim([0, 1.15])
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title("Precision / Recall / F1 per Class", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)

def save_evaluation_plots(cm, prf, probas, y_true, classes, output_path):
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)

    plot_confusion_matrix(cm, [str(c) for c in classes], fig.add_subplot(gs[0, 0]))
    plot_prf_bars(prf, classes, fig.add_subplot(gs[0, 1]))
    plot_roc_curves(probas, y_true, classes, fig.add_subplot(gs[1, 0]))
    plot_prc_curves(probas, y_true, classes, fig.add_subplot(gs[1, 1]))

    fig.suptitle("KNN Classifier - Evaluation Dashboard",
                 fontsize=13, fontweight="bold", y=1.01)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

def fit_knn(
        X_train,
        y_train,
        X_test,
        y_test,
        k=7,
        metric="euclidean",
        n_cv_folds=5
):
    clf = KNNClassifier(k=k, metric=metric)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_probas = clf.predict_proba(X_test)
    accuracy = clf.score(X_test, y_test)

    cm = confusion_matrix(y_test, y_pred, clf.classes_)
    prf = precision_recall_f1(cm, clf.classes_)

    cv_scores = cv_mean = cv_std = None
    if n_cv_folds > 1:
        X_all = np.vstack([X_train, X_test])
        y_all = np.concatenate([y_train, y_test])
        cv_scores, cv_mean, cv_std = kfold_cross_val(
            clf_factory=lambda: KNNClassifier(k=k, metric=metric),
            X=X_all, y=y_all, k=n_cv_folds
        )

    return dict(
        model=clf,
        y_pred=y_pred,
        y_probas=y_probas,
        test_accuracy=accuracy,
        confusion_matrix=cm,
        metrics=prf,
        cv_scores=cv_scores,
        cv_mean=cv_mean,
        cv_std=cv_std
    )