from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    confusion_matrix,
)


def evaluate_classification(model, X_test, y_test):
    """
    Evaluate a trained classification model.
    """

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, average="weighted"),
        "recall": recall_score(y_test, y_pred, average="weighted"),
        "precision": precision_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
    }

    cm = confusion_matrix(y_test, y_pred)

    return {
        "metrics": metrics,
        "confusion_matrix": cm,
        "predictions": y_pred,
    }