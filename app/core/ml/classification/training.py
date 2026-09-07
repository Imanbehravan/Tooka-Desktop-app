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
    feature_selection_params=None,
):
    """
    Train and evaluate a classification model.

    Feature selection is performed only on the
    training data and then applied to the test data.

    Parameters
    ----------
    model_name : str
        Name of the classification model.

    X_train : pandas.DataFrame
        Training features.

    y_train : pandas.Series
        Training target.

    X_test : pandas.DataFrame
        Test features.

    y_test : pandas.Series
        Test target.

    model_params : dict, optional
        Parameters passed to the classification model.

    feature_selection : str, default="No Selection"
        Feature selection method.

    feature_selection_params : dict, optional
        Parameters passed to the selected feature selection method.

        For GA, examples include:
            population_size
            generations
            crossover_rate
            mutation_rate
            alpha
            random_state
            output_path

    Returns
    -------
    dict
        Trained model, selected features, metrics,
        confusion matrix and predictions.
    """

    model_params = model_params or {}
    feature_selection_params = feature_selection_params or {}

    # ---------------------------------------------------------
    # Feature Selection
    # ---------------------------------------------------------

    X_train_selected, selected_features = select_features(
        X=X_train,
        y=y_train,
        method=feature_selection,
        model_name=model_name,
        model_params=model_params,
        ga_params=(
            feature_selection_params
            if feature_selection == "GA"
            else {}
        ),
        pso_params=(
            feature_selection_params
            if feature_selection == "PSO"
            else {}
        ),
)

    # Apply exactly the same selected features to test data
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

    # ---------------------------------------------------------
    # Result
    # ---------------------------------------------------------

    return {
        "model": model,
        "model_name": model_name,
        "feature_selection": feature_selection,
        "selected_features": selected_features,
        "metrics": evaluation["metrics"],
        "confusion_matrix": evaluation["confusion_matrix"],
        "predictions": evaluation["predictions"],
    }