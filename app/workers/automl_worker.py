from PySide6.QtCore import QObject, Signal

from app.core.ml.automl.engine import AutoMLEngine


class AutoMLWorker(QObject):
    """
    Worker responsible for running the AutoML engine
    outside the main UI thread.
    """

    finished = Signal(dict)
    error = Signal(str)
    progress = Signal(int)
    status = Signal(str)

    def __init__(
        self,
        dataframe,
        target_column,
        model_names,
        feature_selection="No Selection",
        hyperparameter_tuning=False,
        tuning_method="Grid Search",
        tuning_params=None,
        scoring="accuracy",
        test_size=0.2,
        random_state=42,
    ):
        super().__init__()

        self.dataframe = dataframe
        self.target_column = target_column
        self.model_names = model_names

        self.feature_selection = feature_selection
        self.hyperparameter_tuning = hyperparameter_tuning
        self.tuning_method = tuning_method
        self.tuning_params = tuning_params
        self.scoring = scoring

        self.test_size = test_size
        self.random_state = random_state

        self._cancelled = False

    def cancel(self):
        """Request cancellation of the current AutoML task."""
        self._cancelled = True

    def _progress_callback(self, value):
        if self._cancelled:
            return

        self.progress.emit(int(value))

    def _status_callback(self, message):
        if self._cancelled:
            return

        self.status.emit(str(message))

    def run(self):
        """
        Execute AutoML classification.
        """

        try:
            if self._cancelled:
                return

            self.status.emit("Starting AutoML...")

            engine = AutoMLEngine()

            result = engine.run_classification(
                dataframe=self.dataframe,
                target_column=self.target_column,
                models=self.model_names,
                feature_selection=self.feature_selection,
                hyperparameter_tuning=self.hyperparameter_tuning,
                tuning_method=self.tuning_method,
                tuning_params=self.tuning_params,
                scoring=self.scoring,
                test_size=self.test_size,
                progress_callback=self._progress_callback,
                status_callback=self._status_callback,
            )

            if self._cancelled:
                self.status.emit("AutoML cancelled.")
                return

            self.progress.emit(100)
            self.status.emit(
                "AutoML completed successfully."
            )

            self.finished.emit(result)

        except Exception as exc:
            if not self._cancelled:
                self.error.emit(str(exc))