from .evaluation import evaluate_classification
from .feature_selection import select_features
from .models import build_model


def train_classification(
    model_name,
    X_train,
    y_train,
    X_test,
    y_test,
    model_params=None,
    feature_selection="No Selection",
):
    """
    Train and evaluate a classification model.

    Feature selection is performed only on the
    training data and then applied to the test data.
    """

    model_params = model_params or {}

    # ---------------------------------------------------------
    # Feature Selection
    # ---------------------------------------------------------

    X_train_selected, selected_features = select_features(
        X=X_train,
        y=y_train,
        method=feature_selection,
        model_name=model_name,
    )

    X_test_selected = X_test[selected_features]

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = build_model(
        model_name,
        **model_params,
    )

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    model.fit(
        X_train_selected,
        y_train,
    )

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------

    evaluation = evaluate_classification(
        model,
        X_test_selected,
        y_test,
    )

    return {
        "model": model,
        "model_name": model_name,
        "feature_selection": feature_selection,
        "selected_features": selected_features,
        "metrics": evaluation["metrics"],
        "confusion_matrix": evaluation[
            "confusion_matrix"
        ],
        "predictions": evaluation[
            "predictions"
        ],
    }