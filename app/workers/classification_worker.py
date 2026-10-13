from PySide6.QtCore import QObject, Signal, Slot

from app.core.ml.classification.dataset import (
    prepare_classification_data,
)
from app.core.ml.classification.training import (
    train_classification,
)

from app.core.ml.model_registry import (
    ModelInfo,
    ModelRegistry,
)

from app.core.ml.model_registry.model_serializer import (
    ModelSerializer,
)


class ClassificationWorker(QObject):
    """
    Worker responsible for running classification training
    inside a background QThread.

    Pipeline:

        Dataset Preparation
                ↓
        Feature Selection
                ↓
        Hyperparameter Tuning
                ↓
        Model Training
                ↓
        Evaluation
                ↓
        Model Serialization
                ↓
        Model Registry
                ↓
        Finished
    """

    # =========================================================
    # Signals
    # =========================================================

    finished = Signal(dict)
    error = Signal(str)
    progress = Signal(int)
    status = Signal(str)

    # =========================================================
    # Initialization
    # =========================================================

    def __init__(
        self,
        dataframe,
        target_column,
        model_name,
        model_params=None,
        feature_selection="No Selection",
        feature_selection_params=None,
        hyperparameter_tuning=False,
        tuning_method="Grid Search",
        tuning_params=None,
        test_size=0.2,
        random_state=42,
    ):
        super().__init__()

        self.dataframe = dataframe
        self.target_column = target_column
        self.model_name = model_name

        self.model_params = model_params or {}

        self.feature_selection = feature_selection

        self.feature_selection_params = (
            feature_selection_params or {}
        )

        self.hyperparameter_tuning = (
            hyperparameter_tuning
        )

        self.tuning_method = tuning_method

        self.tuning_params = (
            tuning_params or {}
        )

        self.test_size = test_size
        self.random_state = random_state

        self._cancel_requested = False

    # =========================================================
    # Run
    # =========================================================

    @Slot()
    def run(self):
        

        try:

            # =================================================
            # Cancellation Check
            # =================================================

            if self._cancel_requested:
                return

            # =================================================
            # Dataset Preparation
            # =================================================

            self.status.emit(
                "Preparing dataset..."
            )

            self.progress.emit(10)

            X_train, X_test, y_train, y_test = (
                prepare_classification_data(
                    self.dataframe,
                    self.target_column,
                    test_size=self.test_size,
                    random_state=self.random_state,
                )
            )

            if self._cancel_requested:
                return

            # =================================================
            # Feature Selection
            # =================================================

            if self.feature_selection == "No Selection":

                self.status.emit(
                    "Feature selection skipped."
                )

            else:

                self.status.emit(
                    f"Selecting features using "
                    f"{self.feature_selection}..."
                )

            self.progress.emit(25)

            if self._cancel_requested:
                return

            # =================================================
            # Training / Hyperparameter Tuning
            # =================================================

            if self.hyperparameter_tuning:

                self.status.emit(
                    f"Running {self.tuning_method}..."
                )

            else:

                self.status.emit(
                    f"Training {self.model_name}..."
                )

            self.progress.emit(40)

            if self._cancel_requested:
                return

            # =================================================
            # Train Classification Model
            # =================================================

            result = train_classification(
                model_name=self.model_name,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                model_params=self.model_params,
                feature_selection=self.feature_selection,
                feature_selection_params=(
                    self.feature_selection_params
                ),
                hyperparameter_tuning=(
                    self.hyperparameter_tuning
                ),
                tuning_method=self.tuning_method,
                tuning_params=self.tuning_params,
            )

            if self._cancel_requested:
                return

            # =================================================
            # Training Completed
            # =================================================

            self.status.emit(
                "Training completed."
            )

            self.progress.emit(70)

            # =================================================
            # Validate Training Result
            # =================================================

            if not isinstance(result, dict):

                raise TypeError(
                    "Classification training must return a dictionary."
                )

            if "model" not in result:

                raise KeyError(
                    "Training result does not contain 'model'."
                )

            trained_model = result["model"]

            if trained_model is None:

                raise ValueError(
                    "Training returned an empty model."
                )

            metrics = result.get(
                "metrics",
                {}
            )

            if not isinstance(metrics, dict):

                metrics = {}

            accuracy = float(
                metrics.get(
                    "accuracy",
                    0.0,
                )
            )

            # =================================================
            # Cancellation Check
            # =================================================

            if self._cancel_requested:
                return

            # =================================================
            # Initialize Model Registry
            # =================================================

            self.status.emit(
                "Preparing model registry..."
            )

            self.progress.emit(75)

            registry = ModelRegistry()

            # =================================================
            # Find Existing Model Versions
            # =================================================

            existing_models = [
                model
                for model in registry.list_models()
                if model.name == self.model_name
            ]

            if existing_models:

                version_numbers = []

                for existing_model in existing_models:

                    version = (
                        existing_model.version
                        .lower()
                        .strip()
                    )

                    if version.startswith("v"):
                        version = version[1:]

                    try:

                        version_numbers.append(
                            int(version)
                        )

                    except ValueError:

                        continue

                next_version = (
                    max(version_numbers) + 1
                    if version_numbers
                    else 1
                )

            else:

                next_version = 1

            version = f"v{next_version}"

            # =================================================
            # Create Model Metadata
            # =================================================

            model_info = ModelInfo(
                name=self.model_name,
                model_type="Classification",
                algorithm=self.model_name,
                metric_name="accuracy",
                metric_value=accuracy,
                version=version,
                status="Ready",
                description=(
                    "Classification model trained "
                    "using Tooka Desktop."
                ),
            )

            # =================================================
            # Determine Model File Path
            # =================================================

            self.status.emit(
                "Preparing model file..."
            )

            model_path = (
                registry.storage.get_model_file_path(
                    model_info.name,
                    model_info.version,
                )
            )

            print()
            print("========================================")
            print("MODEL SAVE")
            print("========================================")
            print(
                f"Model name: {model_info.name}"
            )
            print(
                f"Model version: {model_info.version}"
            )
            print(
                f"Model path: {model_path}"
            )
            print(
                f"Parent directory: {model_path.parent}"
            )
            print(
                f"Parent exists: {model_path.parent.exists()}"
            )
            print("========================================")

            # =================================================
            # Serialize Model
            # =================================================

            self.status.emit(
                "Saving trained model..."
            )

            self.progress.emit(85)

            ModelSerializer.save(
                trained_model,
                model_path,
            )

            print(
                f"Model file exists: {model_path.exists()}"
            )

            if not model_path.exists():

                raise FileNotFoundError(
                    "ModelSerializer completed, "
                    "but model.joblib was not created."
                )

            # =================================================
            # Store Model Path in Metadata
            # =================================================

            model_info.model_path = str(
                model_path
            )

            # =================================================
            # Register Model
            # =================================================

            self.status.emit(
                "Registering model..."
            )

            self.progress.emit(92)

            registered_model = (
                registry.register_model(
                    model_info
                )
            )

            # =================================================
            # Verify Registration
            # =================================================

            registered = registry.get_model(
                registered_model.name,
                registered_model.version,
            )

            if registered is None:

                raise RuntimeError(
                    "Model was saved but could not "
                    "be found in the model registry."
                )

            print()
            print("========================================")
            print("MODEL REGISTERED")
            print("========================================")
            print(
                f"Name: {registered.name}"
            )
            print(
                f"Version: {registered.version}"
            )
            print(
                f"Path: {registered.model_path}"
            )
            print(
                f"File exists: "
                f"{Path(registered.model_path).exists()}"
            )
            print("========================================")
            print()

            # =================================================
            # Add Registry Information to Result
            # =================================================

            result["registry"] = {
                "name": registered.name,
                "version": registered.version,
                "model_type": registered.model_type,
                "algorithm": registered.algorithm,
                "metric_name": registered.metric_name,
                "metric_value": registered.metric_value,
                "model_path": registered.model_path,
            }

            # =================================================
            # Completed
            # =================================================

            self.status.emit(
                "Model saved successfully."
            )

            self.progress.emit(100)

            self.finished.emit(
                result
            )

        # =====================================================
        # Error Handling
        # =====================================================

        except Exception as exc:

            import traceback

            print()
            print("========================================")
            print("CLASSIFICATION WORKER ERROR")
            print("========================================")

            traceback.print_exc()

            print("========================================")
            print()

            self.error.emit(
                f"{type(exc).__name__}: {exc}"
            )

    # =========================================================
    # Cancel
    # =========================================================

    def cancel(self):

        self._cancel_requested = True
