from PySide6.QtCore import QObject, QThread, Signal

from app.workers.clustering_worker import ClusteringWorker


class ClusteringController(QObject):
    """
    Controls clustering worker and its QThread lifecycle.

    A new QThread is created for every clustering run.
    After the worker finishes, the thread is safely stopped
    and all Python references are cleared.
    """

    finished = Signal(dict)
    error = Signal(str)
    progress = Signal(int)
    status = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.thread = None
        self.worker = None

        self._running = False

        # Temporary result/error waiting for QThread
        # to finish completely.
        self._pending_result = None
        self._pending_error = None

    # =========================================================
    # START
    # =========================================================

    def start_clustering(
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
        """
        Start a new clustering task.
        """

        if self._running:
            raise RuntimeError(
                "A clustering task is already running."
            )

        # -----------------------------------------------------
        # Make sure old references are gone
        # -----------------------------------------------------

        self.thread = None
        self.worker = None

        self._pending_result = None
        self._pending_error = None

        self._running = True

        # -----------------------------------------------------
        # Create thread
        # -----------------------------------------------------

        thread = QThread()

        worker = ClusteringWorker(
            dataframe=dataframe,
            algorithm=algorithm,
            params=params,
            feature_selection=feature_selection,
            feature_selection_params=feature_selection_params,
            use_dim_reduction=use_dim_reduction,
            hyperparameter_tuning=hyperparameter_tuning,
            tuning_params=tuning_params,
        )

        self.thread = thread
        self.worker = worker

        # -----------------------------------------------------
        # Move worker to thread
        # -----------------------------------------------------

        worker.moveToThread(thread)

        # -----------------------------------------------------
        # Start worker
        # -----------------------------------------------------

        thread.started.connect(
            worker.run
        )

        # -----------------------------------------------------
        # Worker signals
        # -----------------------------------------------------

        worker.progress.connect(
            self.progress.emit
        )

        worker.status.connect(
            self.status.emit
        )

        worker.finished.connect(
            self._worker_finished
        )

        worker.error.connect(
            self._worker_error
        )

        # -----------------------------------------------------
        # Stop thread after worker finishes
        # -----------------------------------------------------

        worker.finished.connect(
            thread.quit
        )

        worker.error.connect(
            thread.quit
        )

        # -----------------------------------------------------
        # Worker cleanup
        # -----------------------------------------------------

        thread.finished.connect(
            worker.deleteLater
        )

        # -----------------------------------------------------
        # Final controller cleanup
        # -----------------------------------------------------

        thread.finished.connect(
            self._thread_finished
        )

        # -----------------------------------------------------
        # Start thread
        # -----------------------------------------------------

        thread.start()

    # =========================================================
    # STATE
    # =========================================================

    def is_running(self):
        """
        Return True while clustering is active.
        """

        return self._running

    # =========================================================
    # WORKER FINISHED
    # =========================================================

    def _worker_finished(self, result):
        """
        Worker completed successfully.

        Do NOT emit finished to the page yet.
        First let QThread finish completely.
        """

        self._pending_result = result

    # =========================================================
    # WORKER ERROR
    # =========================================================

    def _worker_error(self, message):
        """
        Worker failed.

        Store the error until the thread has completely stopped.
        """

        self._pending_error = str(message)

    # =========================================================
    # THREAD FINISHED
    # =========================================================

    def _thread_finished(self):
        """
        Called after QThread has completely stopped.

        At this point it is safe to expose the result/error
        and reset the controller state.
        """

        result = self._pending_result
        error = self._pending_error

        self._pending_result = None
        self._pending_error = None

        # -----------------------------------------------------
        # Clear state BEFORE notifying the UI
        # -----------------------------------------------------

        self._running = False

        self.thread = None
        self.worker = None

        # -----------------------------------------------------
        # Notify UI
        # -----------------------------------------------------

        if error is not None:

            self.error.emit(error)

        elif result is not None:

            self.finished.emit(result)

    # =========================================================
    # CANCEL
    # =========================================================

    def cancel(self):
        """
        Request cancellation.

        Actual cooperative cancellation will be implemented
        inside ClusteringWorker later.
        """

        if not self._running:
            return

        self.status.emit(
            "Cancelling clustering..."
        )