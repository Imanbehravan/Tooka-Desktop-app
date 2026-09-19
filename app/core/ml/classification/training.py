from .evaluation import evaluate_classification
from .feature_selection import select_features
from .models import build_model
from .tuning import tune_hyperparameters


def train_classification(
    model_name,
    X_train,
    y_train,
    X_test,
    y_test,
    model_params=None,
    feature_selection="No Selection",
    feature_selection_params=None,
    hyperparameter_tuning=False,
    tuning_method="Grid Search",
    tuning_params=None,
):
    """
    Train and evaluate a classification model.

    Pipeline:

        Feature Selection
                ↓
        Hyperparameter Tuning
                ↓
        Model Training
                ↓
        Evaluation

    Parameters
    ----------
    model_name : str
        Classification model name.

    X_train : pandas.DataFrame
        Training features.

    y_train : pandas.Series
        Training target.

    X_test : pandas.DataFrame
        Test features.

    y_test : pandas.Series
        Test target.

    model_params : dict, optional
        Parameters for the classification model.

    feature_selection : str
        Feature selection method.

    feature_selection_params : dict, optional
        Parameters for GA or PSO.

    hyperparameter_tuning : bool
        Whether hyperparameter tuning should be performed.

    tuning_method : str
        "Grid Search" or "Random Search".

    tuning_params : dict, optional
        Parameters for hyperparameter tuning.

    Returns
    -------
    dict
        Complete training and evaluation result.
    """

    # ---------------------------------------------------------
    # Defaults
    # ---------------------------------------------------------

    model_params = model_params or {}
    feature_selection_params = (
        feature_selection_params or {}
    )
    tuning_params = tuning_params or {}

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

    # Apply selected features to test set
    X_test_selected = X_test[selected_features]

    # ---------------------------------------------------------
    # Hyperparameter Tuning
    # ---------------------------------------------------------

    tuning_result = None

    if hyperparameter_tuning:

        tuning_result = tune_hyperparameters(
            model_name=model_name,
            X_train=X_train_selected,
            y_train=y_train,
            method=tuning_method,
            param_grid=tuning_params.get(
                "param_grid",
                {},
            ),
            cv=tuning_params.get(
                "cv",
                5,
            ),
            scoring=tuning_params.get(
                "scoring",
                "accuracy",
            ),
            n_iter=tuning_params.get(
                "n_iter",
                20,
            ),
            random_state=tuning_params.get(
                "random_state",
                42,
            ),
            n_jobs=tuning_params.get(
                "n_jobs",
                -1,
            ),
        )

        # Best model found by tuning
        model = tuning_result["model"]

    else:

        # -----------------------------------------------------
        # Standard Model
        # -----------------------------------------------------

        model = build_model(
            model_name,
            **model_params,
        )

        # -----------------------------------------------------
        # Training
        # -----------------------------------------------------

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

    result = {
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

        "hyperparameter_tuning": (
            hyperparameter_tuning
        ),

        "tuning_method": (
            tuning_method
            if hyperparameter_tuning
            else None
        ),

        "best_params": (
            tuning_result["best_params"]
            if tuning_result is not None
            else {}
        ),

        "best_cv_score": (
            tuning_result["best_score"]
            if tuning_result is not None
            else None
        ),
    }

    return result