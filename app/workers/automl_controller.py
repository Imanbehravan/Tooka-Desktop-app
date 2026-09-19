from PySide6.QtCore import QObject, QThread, Signal

from app.workers.automl_worker import AutoMLWorker


class AutoMLController(QObject):
    """
    Controls AutoML worker execution in a separate QThread.
    """

    finished = Signal(dict)
    error = Signal(str)
    progress = Signal(int)
    status = Signal(str)
    started = Signal()
    cancelled = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.thread = None
        self.worker = None

    def start(
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
        """
        Start AutoML in a background thread.
        """

        if self.thread is not None and self.thread.isRunning():
            raise RuntimeError(
                "An AutoML task is already running."
            )

        self.thread = QThread()

        self.worker = AutoMLWorker(
            dataframe=dataframe,
            target_column=target_column,
            model_names=model_names,
            feature_selection=feature_selection,
            hyperparameter_tuning=hyperparameter_tuning,
            tuning_method=tuning_method,
            tuning_params=tuning_params,
            scoring=scoring,
            test_size=test_size,
            random_state=random_state,
        )

        self.worker.moveToThread(self.thread)

        # Thread lifecycle
        self.thread.started.connect(self.worker.run)

        # Worker signals
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.progress.connect(self.progress.emit)
        self.worker.status.connect(self.status.emit)

        # Cleanup
        self.worker.finished.connect(self.thread.quit)
        self.worker.error.connect(self.thread.quit)

        self.thread.finished.connect(self._cleanup)

        self.started.emit()

        self.thread.start()

    def cancel(self):
        """
        Request cancellation.
        """

        if self.worker is not None:
            self.worker.cancel()

        self.cancelled.emit()

    def is_running(self):
        """
        Return True if AutoML is currently running.
        """

        return (
            self.thread is not None
            and self.thread.isRunning()
        )

    def _on_finished(self, result):
        self.finished.emit(result)

    def _on_error(self, message):
        self.error.emit(message)

    def _cleanup(self):
        """
        Release worker/thread references after completion.
        """

        if self.worker is not None:
            self.worker.deleteLater()

        if self.thread is not None:
            self.thread.deleteLater()

        self.worker = None
        self.thread = None