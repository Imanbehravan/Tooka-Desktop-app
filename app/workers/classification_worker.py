from PySide6.QtCore import QObject, Signal, Slot

from app.core.ml.classification.dataset import (
    prepare_classification_data,
)

from app.core.ml.classification.training import (
    train_classification,
)


class ClassificationWorker(QObject):

    finished = Signal(dict)
    error = Signal(str)
    progress = Signal(int)
    status = Signal(str)

    def __init__(
        self,
        dataframe,
        target_column,
        model_name,
        model_params=None,
        feature_selection="No Selection",
        test_size=0.2,
        random_state=42,
    ):
        super().__init__()

        self.dataframe = dataframe
        self.target_column = target_column
        self.model_name = model_name
        self.model_params = model_params or {}

        self.feature_selection = (
            feature_selection
        )

        self.test_size = test_size
        self.random_state = random_state

        self._cancel_requested = False

    @Slot()
    def run(self):

        try:

            if self._cancel_requested:
                return

            # -------------------------------------------------
            # Dataset
            # -------------------------------------------------

            self.status.emit(
                "Preparing dataset..."
            )

            self.progress.emit(10)

            (
                X_train,
                X_test,
                y_train,
                y_test,
            ) = prepare_classification_data(
                self.dataframe,
                self.target_column,
                test_size=self.test_size,
                random_state=self.random_state,
            )

            if self._cancel_requested:
                return

            # -------------------------------------------------
            # Feature Selection
            # -------------------------------------------------

            if self.feature_selection == "No Selection":

                self.status.emit(
                    "Feature selection skipped."
                )

                self.progress.emit(20)

            else:

                self.status.emit(
                    f"Selecting features using "
                    f"{self.feature_selection}..."
                )

                self.progress.emit(20)

            if self._cancel_requested:
                return

            # -------------------------------------------------
            # Training
            # -------------------------------------------------

            self.status.emit(
                f"Training {self.model_name}..."
            )

            self.progress.emit(40)

            result = train_classification(
                model_name=self.model_name,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                model_params=self.model_params,
                feature_selection=(
                    self.feature_selection
                ),
            )

            if self._cancel_requested:
                return

            # -------------------------------------------------
            # Finished
            # -------------------------------------------------

            self.status.emit(
                "Training completed."
            )

            self.progress.emit(100)

            self.finished.emit(result)

        except Exception as exc:

            self.error.emit(str(exc))

    def cancel(self):

        self._cancel_requested = True