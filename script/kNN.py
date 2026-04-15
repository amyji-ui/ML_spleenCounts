import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter
from sklearn.metrics import classification_report

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

def fit_knn(
        X_train,
        y_train,
        X_test,
        y_test,
        k=7,
        metric="euclidean"
):
    clf = KNNClassifier(k=k, metric=metric)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_probas = clf.predict_proba(X_test)
    accuracy = clf.score(X_test, y_test)

    cm = confusion_matrix(y_test, y_pred, clf.classes_)

    report = classification_report(y_test, y_pred)

    return dict(
        model=clf,
        y_pred=y_pred,
        y_probas=y_probas,
        test_accuracy=accuracy,
        classification_report=report,
        confusion_matrix=cm
    )