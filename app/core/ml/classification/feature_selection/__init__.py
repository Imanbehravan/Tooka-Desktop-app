from .traditional import (
    select_features as select_traditional_features,
    SUPPORTED_FEATURE_SELECTION as TRADITIONAL_FEATURE_SELECTION,
)

from .ga import ga_feature_selection
from .pso import pso_feature_selection

from ..models import build_model


SUPPORTED_FEATURE_SELECTION = (
    *TRADITIONAL_FEATURE_SELECTION,
    "GA",
    "PSO",
)


def select_features(
    X,
    y,
    method="No Selection",
    model_name="Random Forest",
    model_params=None,
    ga_params=None,
    pso_params=None,
):
    """
    Select features using the requested method.

    Supported methods:
        - No Selection
        - RFE
        - Chi-Square
        - ANOVA
        - Mutual Information
        - GA
        - PSO
    """

    model_params = model_params or {}
    ga_params = ga_params or {}
    pso_params = pso_params or {}

    # ---------------------------------------------------------
    # Traditional Methods
    # ---------------------------------------------------------

    if method in TRADITIONAL_FEATURE_SELECTION:
        return select_traditional_features(
            X=X,
            y=y,
            method=method,
            model_name=model_name,
        )

    # ---------------------------------------------------------
    # Genetic Algorithm
    # ---------------------------------------------------------

    if method == "GA":

        model = build_model(
            model_name,
            **model_params,
        )

        selected_features = ga_feature_selection(
            X=X,
            y=y,
            model=model,
            **ga_params,
        )

        X_selected = X[selected_features]

        return X_selected, selected_features

    # ---------------------------------------------------------
    # Particle Swarm Optimization
    # ---------------------------------------------------------

    if method == "PSO":

        model = build_model(
            model_name,
            **model_params,
        )

        selected_features = pso_feature_selection(
            X=X,
            y=y,
            model=model,
            **pso_params,
        )

        X_selected = X[selected_features]

        return X_selected, selected_features

    # ---------------------------------------------------------
    # Unsupported Method
    # ---------------------------------------------------------

    raise ValueError(
        f"Unsupported feature selection method: {method}"
    )


__all__ = [
    "select_features",
    "SUPPORTED_FEATURE_SELECTION",
    "ga_feature_selection",
    "pso_feature_selection",
]