from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def fit_random_forest(X_train, y_train, X_test, y_test, n_estimators=100, random_state=42):
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    test_accuracy = rf.score(X_test, y_test)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    results = {
        "test_accuracy": test_accuracy,
        "classification_report": report,
        "confusion_matrix": conf_matrix
    }
    return results