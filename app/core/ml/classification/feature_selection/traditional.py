import pandas as pd

from sklearn.feature_selection import (
    RFE,
    SelectKBest,
    chi2,
    f_classif,
    mutual_info_classif,
)
from sklearn.preprocessing import MinMaxScaler

from ..models import build_model


SUPPORTED_FEATURE_SELECTION = (
    "No Selection",
    "RFE",
    "Chi-Square",
    "ANOVA",
    "Mutual Information",
)


def _get_feature_count(n_features: int) -> int:
    """
    Select approximately half of the available features.

    The number of selected features is at least 1
    when there is more than one feature.
    """

    if n_features <= 1:
        return n_features

    return max(1, n_features // 2)


def no_selection(X, y):
    """
    Do not perform feature selection.

    Returns the original dataset and all feature names.
    """

    selected_features = list(X.columns)

    return X, selected_features


def rfe_selection(
    X,
    y,
    model_name,
):
    """
    Recursive Feature Elimination (RFE).

    Uses the selected classification model as the estimator
    and keeps approximately half of the available features.
    """

    estimator = build_model(model_name)

    n_features = _get_feature_count(X.shape[1])

    selector = RFE(
        estimator=estimator,
        n_features_to_select=n_features,
    )

    selector.fit(X, y)

    selected_features = list(
        X.columns[selector.support_]
    )

    X_selected = X[selected_features]

    return X_selected, selected_features


def chi_square_selection(X, y):
    """
    Chi-Square feature selection.

    Chi-Square requires non-negative feature values.
    Therefore, MinMaxScaler is applied before selection.
    """

    n_features = _get_feature_count(X.shape[1])

    scaler = MinMaxScaler()

    X_scaled = scaler.fit_transform(X)

    selector = SelectKBest(
        score_func=chi2,
        k=n_features,
    )

    X_selected_array = selector.fit_transform(
        X_scaled,
        y,
    )

    selected_features = list(
        X.columns[selector.get_support()]
    )

    X_selected = pd.DataFrame(
        X_selected_array,
        columns=selected_features,
        index=X.index,
    )

    return X_selected, selected_features


def anova_selection(X, y):
    """
    ANOVA F-test feature selection.

    Selects approximately half of the available features
    based on their ANOVA F-score.
    """

    n_features = _get_feature_count(X.shape[1])

    selector = SelectKBest(
        score_func=f_classif,
        k=n_features,
    )

    X_selected_array = selector.fit_transform(
        X,
        y,
    )

    selected_features = list(
        X.columns[selector.get_support()]
    )

    X_selected = pd.DataFrame(
        X_selected_array,
        columns=selected_features,
        index=X.index,
    )

    return X_selected, selected_features


def mutual_information_selection(X, y):
    """
    Mutual Information feature selection.

    Selects approximately half of the available features
    based on mutual information between features and target.
    """

    n_features = _get_feature_count(X.shape[1])

    selector = SelectKBest(
        score_func=mutual_info_classif,
        k=n_features,
    )

    X_selected_array = selector.fit_transform(
        X,
        y,
    )

    selected_features = list(
        X.columns[selector.get_support()]
    )

    X_selected = pd.DataFrame(
        X_selected_array,
        columns=selected_features,
        index=X.index,
    )

    return X_selected, selected_features


def select_features(
    X,
    y,
    method="No Selection",
    model_name="Random Forest",
):
    """
    Main entry point for traditional feature selection.

    Supported methods:
        - No Selection
        - RFE
        - Chi-Square
        - ANOVA
        - Mutual Information

    Returns:
        X_selected: DataFrame containing selected features
        selected_features: List of selected feature names
    """

    if method == "No Selection":
        return no_selection(
            X,
            y,
        )

    if method == "RFE":
        return rfe_selection(
            X,
            y,
            model_name,
        )

    if method == "Chi-Square":
        return chi_square_selection(
            X,
            y,
        )

    if method == "ANOVA":
        return anova_selection(
            X,
            y,
        )

    if method == "Mutual Information":
        return mutual_information_selection(
            X,
            y,
        )

    raise ValueError(
        f"Unsupported traditional feature selection method: {method}"
    )