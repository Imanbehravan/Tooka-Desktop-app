from PySide6.QtCore import QObject, Signal

from app.core.ml.clustering.training import train_clustering
from app.core.ml.clustering.feature_selection import (
    NoFeatureSelection,
    VarianceThresholdFeatureSelection,
    PCAFeatureSelection,
    FeatureAgglomerationSelection,
)
from app.core.ml.clustering.tuning import (
    tune_kmeans,
    tune_dbscan,
    tune_agglomerative,
)


class ClusteringWorker(QObject):
    finished = Signal(dict)
    error = Signal(str)
    progress = Signal(int)
    status = Signal(str)

    def __init__(
        self,
        dataframe,
        algorithm,
        params=None,
        feature_selection="No Selection",
        feature_selection_params=None,
        use_dim_reduction=False,
        hyperparameter_tuning=False,
        tuning_params=None,
    ):
        super().__init__()

        self.dataframe = dataframe
        self.algorithm = algorithm
        self.params = params or {}

        self.feature_selection = feature_selection
        self.feature_selection_params = feature_selection_params or {}

        self.use_dim_reduction = use_dim_reduction

        self.hyperparameter_tuning = hyperparameter_tuning
        self.tuning_params = tuning_params or {}

    def run(self):
        try:
            self.status.emit("Preparing clustering data...")
            self.progress.emit(10)

            if self.dataframe is None:
                raise ValueError("Dataset cannot be None.")

            if self.dataframe.empty:
                raise ValueError("Dataset is empty.")

            data = self.dataframe.values

            # -------------------------------------------------
            # Feature Selection
            # -------------------------------------------------
            self.status.emit("Applying feature selection...")
            self.progress.emit(20)

            data = self._apply_feature_selection(data)

            # -------------------------------------------------
            # Hyperparameter Tuning
            # -------------------------------------------------
            tuning_result = None

            if self.hyperparameter_tuning:
                self.status.emit("Tuning clustering parameters...")
                self.progress.emit(40)

                tuning_result = self._run_tuning(data)

                if self.algorithm == "KMeans":
                    best_params = tuning_result.get("best_params", {})
                    self.params.update(best_params)

                elif self.algorithm == "DBSCAN":
                    best_params = tuning_result.get("best_params", {})
                    self.params.update(best_params)

                elif self.algorithm == "Agglomerative":
                    best_params = tuning_result.get("best_params", {})
                    self.params.update(best_params)

            # -------------------------------------------------
            # Training
            # -------------------------------------------------
            self.status.emit("Training clustering model...")
            self.progress.emit(60)

            # Create a dataframe from the transformed data.
            # This allows train_clustering() to remain the
            # central training entry point.
            import pandas as pd

            transformed_df = pd.DataFrame(data)

            result = train_clustering(
                dataframe=transformed_df,
                algorithm=self.algorithm,
                params=self.params,
                use_dim_reduction=self.use_dim_reduction,
            )

            self.progress.emit(90)
            self.status.emit("Evaluating clustering results...")

            # -------------------------------------------------
            # Final Result
            # -------------------------------------------------
            result["feature_selection"] = self.feature_selection
            result["hyperparameter_tuning"] = self.hyperparameter_tuning
            result["tuning_result"] = tuning_result

            self.progress.emit(100)
            self.status.emit("Clustering completed successfully.")

            self.finished.emit(result)

        except Exception as exc:
            self.error.emit(str(exc))

    # =========================================================
    # Feature Selection
    # =========================================================

    def _apply_feature_selection(self, data):
        params = self.feature_selection_params

        if self.feature_selection == "No Selection":
            selector = NoFeatureSelection(data)

        elif self.feature_selection == "Variance Threshold":
            threshold = params.get("threshold", 0.0)

            selector = VarianceThresholdFeatureSelection(
                data,
                threshold=threshold,
            )

        elif self.feature_selection == "PCA":
            n_components = params.get("n_components", 2)

            selector = PCAFeatureSelection(
                data,
                n_components=n_components,
            )

        elif self.feature_selection == "Feature Agglomeration":
            n_clusters = params.get("n_clusters", 2)

            selector = FeatureAgglomerationSelection(
                data,
                n_clusters=n_clusters,
            )

        else:
            raise ValueError(
                f"Unsupported feature selection method: "
                f"{self.feature_selection}"
            )

        return selector.select_features()

    # =========================================================
    # Hyperparameter Tuning
    # =========================================================

    def _run_tuning(self, data):
        if self.algorithm == "KMeans":
            cluster_range = self.tuning_params.get(
                "cluster_range",
                (2, 10),
            )

            metric = self.tuning_params.get(
                "metric",
                "silhouette",
            )

            best_clusters, best_score, scores = tune_kmeans(
                data,
                cluster_range=cluster_range,
                metric=metric,
            )

            return {
                "best_params": {
                    "n_clusters": best_clusters,
                },
                "best_score": best_score,
                "scores": scores,
            }

        if self.algorithm == "DBSCAN":
            eps_values = self.tuning_params.get(
                "eps_values"
            )

            min_samples_values = self.tuning_params.get(
                "min_samples_values"
            )

            metric = self.tuning_params.get(
                "metric",
                "silhouette",
            )

            kwargs = {
                "metric": metric,
            }

            if eps_values is not None:
                kwargs["eps_values"] = eps_values

            if min_samples_values is not None:
                kwargs["min_samples_values"] = min_samples_values

            best_params, best_score, details = tune_dbscan(
                data,
                **kwargs,
            )

            return {
                "best_params": best_params,
                "best_score": best_score,
                "details": details,
            }

        if self.algorithm == "Agglomerative":
            cluster_range = self.tuning_params.get(
                "cluster_range",
                (2, 10),
            )

            linkage_options = self.tuning_params.get(
                "linkage_options",
                [
                    "ward",
                    "complete",
                    "average",
                    "single",
                ],
            )

            metric = self.tuning_params.get(
                "metric",
                "silhouette",
            )

            best_params, best_score, details = tune_agglomerative(
                data,
                cluster_range=cluster_range,
                linkage_options=linkage_options,
                metric=metric,
            )

            return {
                "best_params": best_params,
                "best_score": best_score,
                "details": details,
            }

        raise ValueError(
            f"Unsupported clustering algorithm: {self.algorithm}"
        )