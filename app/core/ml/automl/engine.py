from typing import Callable, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from app.core.ml.classification.training import (
    train_classification,
)


class AutoMLEngine:
    """
    Core AutoML engine.

    This class is independent from PySide6 and the UI.

    Responsibilities:
    - Prepare dataset
    - Run multiple models
    - Collect evaluation results
    - Rank models
    - Select the best model
    - Apply per-model hyperparameter tuning configuration
    """

    def __init__(
        self,
        random_state: int = 42,
    ):
        self.random_state = random_state

    # =========================================================
    # PUBLIC API
    # =========================================================

    def run_classification(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
        models: list[str],
        feature_selection: str = "No Selection",
        test_size: float = 0.2,
        hyperparameter_tuning: bool = False,
        tuning_method: str = "Grid Search",
        tuning_params: Optional[dict] = None,
        scoring: str = "accuracy",
        progress_callback: Optional[
            Callable[[int], None]
        ] = None,
        status_callback: Optional[
            Callable[[str], None]
        ] = None,
    ) -> dict:
        """
        Run AutoML for a classification problem.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            Input dataset.

        target_column : str
            Target column name.

        models : list[str]
            Models that should be trained.

        feature_selection : str
            Feature selection method.

        test_size : float
            Test set ratio.

        hyperparameter_tuning : bool
            Whether hyperparameter tuning is enabled.

        tuning_method : str
            "Grid Search" or "Random Search".

        tuning_params : dict, optional
            Hyperparameter tuning configuration.

            Per-model configuration format:

                {
                    "KNN": {
                        "param_grid": {...},
                        "cv": 5,
                        "scoring": "accuracy",
                        "n_iter": 20,
                        "random_state": 42
                    },

                    "SVM": {
                        "param_grid": {...},
                        "cv": 5,
                        "scoring": "accuracy",
                        "n_iter": 20,
                        "random_state": 42
                    }
                }

            A single shared configuration is also supported
            for backward compatibility:

                {
                    "param_grid": {...},
                    "cv": 5,
                    "scoring": "accuracy",
                    "n_iter": 20,
                    "random_state": 42
                }

        scoring : str
            Metric used to rank models.

        Returns
        -------
        dict
            Complete AutoML result.
        """

        # =====================================================
        # Validation
        # =====================================================

        self._validate_dataset(
            dataframe,
            target_column,
        )

        if not models:
            raise ValueError(
                "At least one model must be selected."
            )

        # =====================================================
        # Prepare Data
        # =====================================================

        if status_callback:
            status_callback(
                "Preparing dataset..."
            )

        X, y = self._prepare_data(
            dataframe,
            target_column,
        )

        # =====================================================
        # Train / Test Split
        # =====================================================

        if status_callback:
            status_callback(
                "Splitting dataset..."
            )

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=self.random_state,
                stratify=y,
            )
        )

        # =====================================================
        # Run Models
        # =====================================================

        total_models = len(models)

        results = []

        best_result = None
        best_score = float("-inf")

        for index, model_name in enumerate(
            models
        ):

            # -------------------------------------------------
            # Status
            # -------------------------------------------------

            if status_callback:

                status_callback(
                    f"Training {model_name}..."
                )

            # -------------------------------------------------
            # Get tuning configuration
            # -------------------------------------------------

            model_tuning_params = (
                self._get_model_tuning_params(
                    tuning_params=tuning_params,
                    model_name=model_name,
                )
            )

            # -------------------------------------------------
            # Train model
            # -------------------------------------------------

            result = train_classification(
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,

                feature_selection=feature_selection,

                hyperparameter_tuning=(
                    hyperparameter_tuning
                ),

                tuning_method=tuning_method,

                tuning_params=model_tuning_params,
            )

            # -------------------------------------------------
            # Metrics
            # -------------------------------------------------

            metrics = result.get(
                "metrics",
                {},
            )

            score = self._get_score(
                metrics,
                scoring,
            )

            # -------------------------------------------------
            # Model Result
            # -------------------------------------------------

            model_result = {
                "rank": None,

                "model": model_name,

                "score": score,

                "accuracy": metrics.get(
                    "accuracy"
                ),

                "precision": metrics.get(
                    "precision"
                ),

                "recall": metrics.get(
                    "recall"
                ),

                "f1": metrics.get(
                    "f1"
                ),

                "feature_selection": result.get(
                    "feature_selection"
                ),

                "selected_features": result.get(
                    "selected_features",
                    [],
                ),

                "confusion_matrix": result.get(
                    "confusion_matrix"
                ),

                "predictions": result.get(
                    "predictions"
                ),

                "trained_model": result.get(
                    "model"
                ),

                "best_params": result.get(
                    "best_params"
                ),

                "best_cv_score": result.get(
                    "best_cv_score"
                ),

                "status": "Completed",
            }

            results.append(
                model_result
            )

            # -------------------------------------------------
            # Best Model
            # -------------------------------------------------

            if score > best_score:

                best_score = score

                best_result = model_result

            # -------------------------------------------------
            # Progress
            # -------------------------------------------------

            if progress_callback:

                progress = int(
                    (
                        (index + 1)
                        / total_models
                    )
                    * 100
                )

                progress_callback(
                    progress
                )

        # =====================================================
        # Rank Models
        # =====================================================

        results.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        for rank, result in enumerate(
            results,
            start=1,
        ):

            result["rank"] = rank

        # =====================================================
        # Best Model
        # =====================================================

        if results:

            best_result = results[0]

        # =====================================================
        # Completed
        # =====================================================

        if status_callback:

            status_callback(
                "AutoML completed."
            )

        # =====================================================
        # Final Result
        # =====================================================

        return {
            "problem_type": "classification",

            "target_column": target_column,

            "models": models,

            "results": results,

            "best_model": (
                best_result["model"]
                if best_result
                else None
            ),

            "best_score": (
                best_result["score"]
                if best_result
                else None
            ),

            "best_result": best_result,
        }

    # =========================================================
    # TUNING CONFIGURATION
    # =========================================================

    def _get_model_tuning_params(
        self,
        tuning_params: Optional[dict],
        model_name: str,
    ) -> Optional[dict]:
        """
        Return the tuning configuration for a specific model.

        Supports two formats.

        -----------------------------------------------------
        Per-model configuration
        -----------------------------------------------------

        {
            "KNN": {
                "param_grid": {...},
                "cv": 5,
                "scoring": "accuracy"
            },

            "SVM": {
                "param_grid": {...},
                "cv": 5,
                "scoring": "accuracy"
            }
        }

        -----------------------------------------------------
        Shared configuration
        -----------------------------------------------------

        {
            "param_grid": {...},
            "cv": 5,
            "scoring": "accuracy"
        }

        The second format is kept for backward compatibility.
        """

        # No configuration
        if not tuning_params:

            return None

        # -----------------------------------------------------
        # Per-model configuration
        # -----------------------------------------------------

        model_config = tuning_params.get(
            model_name
        )

        if isinstance(
            model_config,
            dict,
        ):

            return model_config

        # -----------------------------------------------------
        # Shared configuration
        # -----------------------------------------------------

        if "param_grid" in tuning_params:

            return tuning_params

        # -----------------------------------------------------
        # Invalid configuration
        # -----------------------------------------------------

        raise ValueError(
            f"No hyperparameter configuration found "
            f"for model '{model_name}'."
        )

    # =========================================================
    # DATA PREPARATION
    # =========================================================

    def _prepare_data(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
    ):
        """
        Prepare X and y for sklearn.

        Numeric columns are kept as-is.

        Categorical feature columns are converted using
        one-hot encoding.
        """

        X = dataframe.drop(
            columns=[target_column]
        )

        y = dataframe[
            target_column
        ]

        # -----------------------------------------------------
        # Handle categorical features
        # -----------------------------------------------------

        categorical_columns = X.select_dtypes(
            include=[
                "object",
                "category",
                "bool",
            ]
        ).columns

        if len(categorical_columns) > 0:

            X = pd.get_dummies(
                X,
                columns=list(
                    categorical_columns
                ),
                drop_first=False,
            )

        # -----------------------------------------------------
        # Handle missing values
        # -----------------------------------------------------

        numeric_columns = X.select_dtypes(
            include=["number"]
        ).columns

        if len(numeric_columns) > 0:

            X[numeric_columns] = (
                X[numeric_columns]
                .fillna(
                    X[numeric_columns].median()
                )
            )

        return X, y

    # =========================================================
    # VALIDATION
    # =========================================================

    def _validate_dataset(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
    ):

        if dataframe is None:

            raise ValueError(
                "Dataset cannot be None."
            )

        if not isinstance(
            dataframe,
            pd.DataFrame,
        ):

            raise TypeError(
                "Dataset must be a pandas DataFrame."
            )

        if dataframe.empty:

            raise ValueError(
                "Dataset is empty."
            )

        if target_column not in dataframe.columns:

            raise ValueError(
                f"Target column '{target_column}' "
                "does not exist in the dataset."
            )

        if len(dataframe.columns) < 2:

            raise ValueError(
                "Dataset must contain at least "
                "one feature and one target column."
            )

        if dataframe[target_column].nunique() < 2:

            raise ValueError(
                "Target column must contain at least "
                "two classes."
            )

    # =========================================================
    # SCORING
    # =========================================================

    def _get_score(
        self,
        metrics: dict,
        scoring: str,
    ) -> float:

        scoring_map = {
            "accuracy": "accuracy",
            "f1": "f1",
            "f1_score": "f1",
            "precision": "precision",
            "recall": "recall",
        }

        metric_name = scoring_map.get(
            scoring.lower(),
            "accuracy",
        )

        score = metrics.get(
            metric_name
        )

        if score is None:

            raise ValueError(
                f"Metric '{metric_name}' "
                "was not found in model results."
            )

        return float(score)