from PySide6.QtCore import QObject, QThread, Signal

from .classification_worker import ClassificationWorker


class ClassificationController(QObject):
    """
    Controls the classification training worker.

    The controller is responsible for:
        - Creating the worker
        - Running it inside a QThread
        - Forwarding worker signals to the UI
        - Handling completion and errors
        - Supporting cancellation
    """

    # ---------------------------------------------------------
    # Signals
    # ---------------------------------------------------------

    finished = Signal(dict)
    error = Signal(str)
    progress = Signal(int)
    status = Signal(str)

    # ---------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------

    def __init__(self, parent=None):
        super().__init__(parent)

        self.thread = None
        self.worker = None

    # ---------------------------------------------------------
    # Start Classification
    # ---------------------------------------------------------

    def start_training(
        self,
        dataframe,
        target_column,
        model_name,
        model_params=None,
        feature_selection="No Selection",
        feature_selection_params=None,
        test_size=0.2,
        random_state=42,
    ):
        """
        Start classification training in a background thread.
        """

        # Prevent starting another training while one
        # is already running.
        if self.thread is not None:
            if self.thread.isRunning():
                self.error.emit(
                    "A classification training task is already running."
                )
                return

        # -----------------------------------------------------
        # Default Parameters
        # -----------------------------------------------------

        model_params = model_params or {}

        feature_selection_params = (
            feature_selection_params or {}
        )

        # -----------------------------------------------------
        # Thread
        # -----------------------------------------------------

        self.thread = QThread()

        # -----------------------------------------------------
        # Worker
        # -----------------------------------------------------

        self.worker = ClassificationWorker(
            dataframe=dataframe,
            target_column=target_column,
            model_name=model_name,
            model_params=model_params,
            feature_selection=feature_selection,
            feature_selection_params=feature_selection_params,
            test_size=test_size,
            random_state=random_state,
        )

        # Move worker to background thread
        self.worker.moveToThread(self.thread)

        # -----------------------------------------------------
        # Worker → Controller
        # -----------------------------------------------------

        self.worker.finished.connect(
            self._on_finished
        )

        self.worker.error.connect(
            self._on_error
        )

        self.worker.progress.connect(
            self.progress.emit
        )

        self.worker.status.connect(
            self.status.emit
        )

        # -----------------------------------------------------
        # Thread Signals
        # -----------------------------------------------------

        self.thread.started.connect(
            self.worker.run
        )

        # -----------------------------------------------------
        # Cleanup
        # -----------------------------------------------------

        self.worker.finished.connect(
            self.thread.quit
        )

        self.worker.error.connect(
            self.thread.quit
        )

        self.thread.finished.connect(
            self._cleanup
        )

        # -----------------------------------------------------
        # Start
        # -----------------------------------------------------

        self.thread.start()

    # ---------------------------------------------------------
    # Finished
    # ---------------------------------------------------------

    def _on_finished(self, result):
        """
        Handle successful training.
        """

        self.finished.emit(result)

    # ---------------------------------------------------------
    # Error
    # ---------------------------------------------------------

    def _on_error(self, message):
        """
        Handle worker errors.
        """

        self.error.emit(message)

    # ---------------------------------------------------------
    # Cancel
    # ---------------------------------------------------------

    def cancel(self):
        """
        Request cancellation of the current training task.
        """

        if self.worker is not None:
            self.worker.cancel()

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------

    def _cleanup(self):
        """
        Clean up worker and thread references.
        """

        if self.worker is not None:
            self.worker.deleteLater()

        if self.thread is not None:
            self.thread.deleteLater()

        self.worker = None
        self.thread = None