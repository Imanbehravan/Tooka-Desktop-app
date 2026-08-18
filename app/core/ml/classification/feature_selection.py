import pandas as pd

from sklearn.feature_selection import (
    RFE,
    SelectKBest,
    chi2,
    f_classif,
    mutual_info_classif,
)
from sklearn.preprocessing import MinMaxScaler

from .models import build_model


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
    """

    if n_features <= 1:
        return n_features

    return max(1, n_features // 2)


def no_selection(X, y):
    """
    Do not perform feature selection.
    """

    return X, list(X.columns)


def rfe_selection(
    X,
    y,
    model_name,
):
    """
    Recursive Feature Elimination.
    """

    estimator = build_model(model_name)

    n_features = _get_feature_count(X.shape[1])

    selector = RFE(
        estimator=estimator,
        n_features_to_select=n_features,
    )

    selector.fit(X, y)

    selected_columns = list(
        X.columns[selector.support_]
    )

    X_selected = X[selected_columns]

    return X_selected, selected_columns


def chi_square_selection(X, y):
    """
    Chi-Square feature selection.

    Chi-Square requires non-negative values,
    therefore MinMaxScaler is applied first.
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

    selected_columns = list(
        X.columns[selector.get_support()]
    )

    X_selected = pd.DataFrame(
        X_selected_array,
        columns=selected_columns,
        index=X.index,
    )

    return X_selected, selected_columns


def anova_selection(X, y):
    """
    ANOVA F-test feature selection.
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

    selected_columns = list(
        X.columns[selector.get_support()]
    )

    X_selected = pd.DataFrame(
        X_selected_array,
        columns=selected_columns,
        index=X.index,
    )

    return X_selected, selected_columns


def mutual_information_selection(X, y):
    """
    Mutual Information feature selection.
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

    selected_columns = list(
        X.columns[selector.get_support()]
    )

    X_selected = pd.DataFrame(
        X_selected_array,
        columns=selected_columns,
        index=X.index,
    )

    return X_selected, selected_columns


def select_features(
    X,
    y,
    method="No Selection",
    model_name="Random Forest",
):
    """
    Main feature selection entry point.
    """

    if method == "No Selection":
        return no_selection(X, y)

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
        f"Unsupported feature selection method: {method}"
    )