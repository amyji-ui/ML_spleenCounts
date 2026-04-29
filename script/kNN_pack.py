from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import numpy as np

def fit_knn_pack(X_train, y_train, X_test, y_test, k=7):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    test_accuracy = knn.score(X_test, y_test)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    results = {
        "test_accuracy": test_accuracy,
        "classification_report": report,
        "confusion_matrix": conf_matrix
    }
    return results