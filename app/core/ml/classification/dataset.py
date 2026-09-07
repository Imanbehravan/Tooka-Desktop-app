from pathlib import Path

import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def load_csv(path: str | Path) -> pd.DataFrame:
    """
    Load a CSV dataset.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(
            "The dataset is empty."
        )

    return df


def _detect_identifier_columns(
    df: pd.DataFrame,
    target_column: str,
) -> list[str]:
    """
    Detect common identifier columns.

    We intentionally use conservative rules here.
    A column is considered an identifier when its name
    strongly suggests that it is an ID column.
    """

    identifier_columns = []

    for column in df.columns:

        if column == target_column:
            continue

        normalized = column.strip().lower()

        if (
            normalized == "id"
            or normalized.endswith("_id")
            or normalized.endswith("id")
        ):
            identifier_columns.append(column)

    return identifier_columns


def _split_features_target(
    df: pd.DataFrame,
    target_column: str,
    drop_identifier_columns: bool = True,
):
    """
    Separate features and target.
    """

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' "
            "not found in dataset."
        )

    data = df.copy()

    if drop_identifier_columns:

        identifier_columns = _detect_identifier_columns(
            data,
            target_column,
        )

        if identifier_columns:
            data = data.drop(
                columns=identifier_columns
            )

    X = data.drop(
        columns=[target_column]
    )

    y = data[target_column]

    return X, y


def _prepare_target(
    y: pd.Series,
):
    """
    Clean the target column.

    Rows with missing target values cannot be used
    for supervised classification.
    """

    missing_target = y.isna()

    if missing_target.any():
        y = y.loc[~missing_target]

    return y


def _align_target_and_features(
    X: pd.DataFrame,
    y: pd.Series,
):
    """
    Keep X and y aligned after removing invalid target rows.
    """

    valid_indices = y.index

    X = X.loc[valid_indices]

    return X, y


def _preprocess_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
):
    """
    Preprocess numerical and categorical features.

    Numerical:
        Missing values -> median

    Categorical:
        Missing values -> most frequent
        Encoding -> One-Hot Encoding

    The preprocessing objects are fitted only on the
    training data.
    """

    X_train = X_train.copy()
    X_test = X_test.copy()

    numeric_columns = list(
        X_train.select_dtypes(
            include=["number"]
        ).columns
    )

    categorical_columns = list(
        X_train.select_dtypes(
            exclude=["number"]
        ).columns
    )

    # ---------------------------------------------------------
    # Numerical features
    # ---------------------------------------------------------

    if numeric_columns:

        numeric_imputer = SimpleImputer(
            strategy="median"
        )

        X_train_numeric = (
            numeric_imputer.fit_transform(
                X_train[numeric_columns]
            )
        )

        X_test_numeric = (
            numeric_imputer.transform(
                X_test[numeric_columns]
            )
        )

        X_train_numeric = pd.DataFrame(
            X_train_numeric,
            columns=numeric_columns,
            index=X_train.index,
        )

        X_test_numeric = pd.DataFrame(
            X_test_numeric,
            columns=numeric_columns,
            index=X_test.index,
        )

    else:

        X_train_numeric = pd.DataFrame(
            index=X_train.index
        )

        X_test_numeric = pd.DataFrame(
            index=X_test.index
        )

    # ---------------------------------------------------------
    # Categorical features
    # ---------------------------------------------------------

    if categorical_columns:

        categorical_imputer = SimpleImputer(
            strategy="most_frequent"
        )

        X_train_categorical = (
            categorical_imputer.fit_transform(
                X_train[categorical_columns]
            )
        )

        X_test_categorical = (
            categorical_imputer.transform(
                X_test[categorical_columns]
            )
        )

        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )

        X_train_encoded = (
            encoder.fit_transform(
                X_train_categorical
            )
        )

        X_test_encoded = (
            encoder.transform(
                X_test_categorical
            )
        )

        encoded_columns = (
            encoder.get_feature_names_out(
                categorical_columns
            )
        )

        X_train_categorical = pd.DataFrame(
            X_train_encoded,
            columns=encoded_columns,
            index=X_train.index,
        )

        X_test_categorical = pd.DataFrame(
            X_test_encoded,
            columns=encoded_columns,
            index=X_test.index,
        )

    else:

        X_train_categorical = pd.DataFrame(
            index=X_train.index
        )

        X_test_categorical = pd.DataFrame(
            index=X_test.index
        )

    # ---------------------------------------------------------
    # Combine
    # ---------------------------------------------------------

    X_train_processed = pd.concat(
        [
            X_train_numeric,
            X_train_categorical,
        ],
        axis=1,
    )

    X_test_processed = pd.concat(
        [
            X_test_numeric,
            X_test_categorical,
        ],
        axis=1,
    )

    return (
        X_train_processed,
        X_test_processed,
    )


def prepare_classification_data(
    df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
    drop_identifier_columns: bool = True,
):
    """
    Prepare a classification dataset.

    Pipeline:

        DataFrame
            ↓
        Target validation
            ↓
        Identifier removal
            ↓
        Missing target removal
            ↓
        Train/Test split
            ↓
        Numerical imputation
            ↓
        Categorical imputation
            ↓
        One-Hot Encoding
            ↓
        Ready for ML
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "df must be a pandas DataFrame."
        )

    if df.empty:
        raise ValueError(
            "The dataset is empty."
        )

    # ---------------------------------------------------------
    # Split X / y
    # ---------------------------------------------------------

    X, y = _split_features_target(
        df,
        target_column,
        drop_identifier_columns=(
            drop_identifier_columns
        ),
    )

    # ---------------------------------------------------------
    # Remove missing target rows
    # ---------------------------------------------------------

    y = _prepare_target(y)

    X, y = _align_target_and_features(
        X,
        y,
    )

    if len(X) < 2:
        raise ValueError(
            "Not enough valid samples "
            "after removing missing targets."
        )

    # ---------------------------------------------------------
    # Train / Test split
    # ---------------------------------------------------------

    try:

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=y,
            )
        )

    except ValueError as exc:

        raise ValueError(
            "Unable to create a stratified "
            "train/test split. Make sure every "
            "class has enough samples."
        ) from exc

    # ---------------------------------------------------------
    # Feature preprocessing
    # ---------------------------------------------------------

    (
        X_train,
        X_test,
    ) = _preprocess_features(
        X_train,
        X_test,
    )

    if X_train.shape[1] == 0:
        raise ValueError(
            "No usable features remain "
            "after preprocessing."
        )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )