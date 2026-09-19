from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
)

from ..models import build_model


SUPPORTED_TUNING_METHODS = (
    "Grid Search",
    "Random Search",
)


def tune_hyperparameters(
    model_name,
    X_train,
    y_train,
    method="Grid Search",
    param_grid=None,
    cv=5,
    scoring="accuracy",
    n_iter=20,
    random_state=42,
    n_jobs=-1,
):
    """
    Perform hyperparameter tuning for a classification model.

    Parameters
    ----------
    model_name : str
        Name of the classification model.

    X_train : pandas.DataFrame
        Training features.

    y_train : pandas.Series
        Training target.

    method : str, default="Grid Search"
        Tuning strategy.

    param_grid : dict
        Hyperparameter search space.

    cv : int, default=5
        Number of cross-validation folds.

    scoring : str, default="accuracy"
        Evaluation metric used during tuning.

    n_iter : int, default=20
        Number of parameter settings sampled by
        Random Search.

    random_state : int, default=42
        Random seed used by Random Search.

    n_jobs : int, default=-1
        Number of parallel jobs.

    Returns
    -------
    dict
        Best model, parameters and score.
    """

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    if method not in SUPPORTED_TUNING_METHODS:
        raise ValueError(
            f"Unsupported tuning method: {method}"
        )

    if not param_grid:
        raise ValueError(
            "Parameter search space cannot be empty."
        )

    # ---------------------------------------------------------
    # Base Model
    # ---------------------------------------------------------

    model = build_model(model_name)

    # ---------------------------------------------------------
    # Grid Search
    # ---------------------------------------------------------

    if method == "Grid Search":

        search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
        )

    # ---------------------------------------------------------
    # Random Search
    # ---------------------------------------------------------

    elif method == "Random Search":

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    # ---------------------------------------------------------
    # Training Search
    # ---------------------------------------------------------

    search.fit(
        X_train,
        y_train,
    )

    # ---------------------------------------------------------
    # Result
    # ---------------------------------------------------------

    return {
        "model": search.best_estimator_,
        "best_params": search.best_params_,
        "best_score": search.best_score_,
        "cv_results": search.cv_results_,
        "search": search,
    }