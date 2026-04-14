from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def fit_multinomial_logistic(X_train, y_train, X_test, y_test, max_iter=5000, C=1.0):
    mlr = LogisticRegression(solver='lbfgs', max_iter=max_iter, C=C, random_state=42)
    mlr.fit(X_train, y_train)
    y_pred = mlr.predict(X_test)
    test_accuracy = mlr.score(X_test, y_test)
    train_accuracy = mlr.score(X_train, y_train)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    results = {
        "test_accuracy": test_accuracy,
        "train_accuracy": train_accuracy,
        "classification_report": report,
        "confusion_matrix": conf_matrix
    }
    return results
