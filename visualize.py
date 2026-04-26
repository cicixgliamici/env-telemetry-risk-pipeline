from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
def evaluate_classifier(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    p, r, f1, _ = precision_recall_fscore_support(y_test, preds, average="weighted", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision_weighted": float(p),
        "recall_weighted": float(r),
        "f1_weighted": float(f1),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "classification_report": classification_report(y_test, preds, zero_division=0),
    }
