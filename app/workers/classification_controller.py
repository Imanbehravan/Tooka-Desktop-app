from PySide6.QtCore import QObject, QThread, Signal

from app.workers.classification_worker import (
    ClassificationWorker,
)


class ClassificationController(QObject):
    """
    Controls the lifecycle of the classification worker.
    """

    finished = Signal(dict)
    error = Signal(str)
    progress = Signal(int)
    status = Signal(str)

    def __init__(self):
        super().__init__()

        self.thread = None
        self.worker = None

    def start_training(
        self,
        dataframe,
        target_column,
        model_name,
        model_params=None,
        feature_selection="No Selection",
        test_size=0.2,
        random_state=42,
    ):
        if self.thread is not None:
            if self.thread.isRunning():
                raise RuntimeError(
                    "A training task is already running."
                )

        self.thread = QThread()

        self.worker = ClassificationWorker(
            dataframe=dataframe,
            target_column=target_column,
            model_name=model_name,
            model_params=model_params,
            feature_selection=feature_selection,
            test_size=test_size,
            random_state=random_state,
        )

        self.worker.moveToThread(self.thread)

        self.thread.started.connect(
            self.worker.run
        )

        self.worker.finished.connect(
            self._on_finished
        )

        self.worker.error.connect(
            self._on_error
        )

        self.worker.progress.connect(
            self.progress
        )

        self.worker.status.connect(
            self.status
        )

        self.worker.finished.connect(
            self.thread.quit
        )

        self.worker.error.connect(
            self.thread.quit
        )

        self.thread.finished.connect(
            self._cleanup
        )

        self.thread.start()

    def cancel(self):
        if self.worker is not None:
            self.worker.cancel()

    def _on_finished(self, result):
        self.finished.emit(result)

    def _on_error(self, message):
        self.error.emit(message)

    def _cleanup(self):
        if self.worker is not None:
            self.worker.deleteLater()

        if self.thread is not None:
            self.thread.deleteLater()

        self.worker = None
        self.thread = None