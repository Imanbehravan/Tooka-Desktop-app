from pathlib import Path

import pandas as pd

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QComboBox,
    QPushButton,
    QProgressBar,
    QFrame,
    QFileDialog,
    QMessageBox,
)

from app.core.ml.classification.models import SUPPORTED_MODELS
from app.workers.classification_controller import (
    ClassificationController,
)

from app.core.ml.classification.feature_selection import (
    SUPPORTED_FEATURE_SELECTION,
)

class AutoMLPage(QWidget):
    def __init__(self):
        super().__init__()

        self.dataset = None
        self.dataset_path = None

        self.controller = ClassificationController()

        self._build_ui()
        self._connect_signals()

    # ---------------------------------------------------------
    # UI
    # ---------------------------------------------------------

    def _build_ui(self):

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(20)

        # -----------------------------------------------------
        # Header
        # -----------------------------------------------------

        title = QLabel("Classification")
        title.setObjectName("pageTitle")

        subtitle = QLabel(
            "Train and evaluate machine learning classification models."
        )
        subtitle.setObjectName("pageSubtitle")

        main_layout.addWidget(title)
        main_layout.addWidget(subtitle)

        # -----------------------------------------------------
        # Configuration Card
        # -----------------------------------------------------

        config_card = QFrame()
        config_card.setObjectName("card")

        config_layout = QVBoxLayout(config_card)
        config_layout.setContentsMargins(20, 20, 20, 20)
        config_layout.setSpacing(15)

        config_title = QLabel("Training Configuration")
        config_title.setObjectName("sectionTitle")

        config_layout.addWidget(config_title)

        # Dataset
        dataset_row = QHBoxLayout()

        dataset_label = QLabel("Dataset")
        dataset_label.setFixedWidth(130)

        self.dataset_label = QLabel("No dataset selected")
        self.dataset_label.setObjectName("datasetLabel")

        self.dataset_button = QPushButton("Select CSV")

        dataset_row.addWidget(dataset_label)
        dataset_row.addWidget(self.dataset_label, 1)
        dataset_row.addWidget(self.dataset_button)

        config_layout.addLayout(dataset_row)

        # Target
        target_row = QHBoxLayout()

        target_label = QLabel("Target Column")
        target_label.setFixedWidth(130)

        self.target_combo = QComboBox()
        self.target_combo.setEnabled(False)

        target_row.addWidget(target_label)
        target_row.addWidget(self.target_combo)

        config_layout.addLayout(target_row)

        # Model
        model_row = QHBoxLayout()

        model_label = QLabel("Model")
        model_label.setFixedWidth(130)

        self.model_combo = QComboBox()
        self.model_combo.addItems(SUPPORTED_MODELS)

        model_row.addWidget(model_label)
        model_row.addWidget(self.model_combo)

        config_layout.addLayout(model_row)

        # Train button
        self.train_button = QPushButton("Start Training")
        self.train_button.setObjectName("primaryButton")
        self.train_button.setMinimumHeight(42)

        config_layout.addWidget(self.train_button)

        main_layout.addWidget(config_card)

        # -----------------------------------------------------
        # Progress Card
        # -----------------------------------------------------

        progress_card = QFrame()
        progress_card.setObjectName("card")

        progress_layout = QVBoxLayout(progress_card)
        progress_layout.setContentsMargins(20, 20, 20, 20)
        progress_layout.setSpacing(10)

        progress_title = QLabel("Training Progress")
        progress_title.setObjectName("sectionTitle")

        self.status_label = QLabel("Ready")

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        progress_layout.addWidget(progress_title)
        progress_layout.addWidget(self.status_label)
        progress_layout.addWidget(self.progress_bar)

        main_layout.addWidget(progress_card)

        # -----------------------------------------------------
        # Results Card
        # -----------------------------------------------------

        results_card = QFrame()
        results_card.setObjectName("card")

        results_layout = QVBoxLayout(results_card)
        results_layout.setContentsMargins(20, 20, 20, 20)
        results_layout.setSpacing(15)

        results_title = QLabel("Results")
        results_title.setObjectName("sectionTitle")

        results_layout.addWidget(results_title)

        self.features_label = QLabel(
           "Selected Features\nNo results yet."
        )

        self.features_label.setWordWrap(True)

        results_layout.addWidget(
           self.features_label
        )

        # Metrics
        metrics_layout = QHBoxLayout()

        self.accuracy_label = self._create_metric(
            "Accuracy"
        )

        self.precision_label = self._create_metric(
            "Precision"
        )

        self.recall_label = self._create_metric(
            "Recall"
        )

        self.f1_label = self._create_metric(
            "F1 Score"
        )

        metrics_layout.addWidget(
            self.accuracy_label
        )

        metrics_layout.addWidget(
            self.precision_label
        )

        metrics_layout.addWidget(
            self.recall_label
        )

        metrics_layout.addWidget(
            self.f1_label
        )

        results_layout.addLayout(metrics_layout)

        # Confusion matrix
        self.confusion_label = QLabel(
            "Confusion Matrix\nNo results yet."
        )

        self.confusion_label.setAlignment(
            Qt.AlignCenter
        )

        results_layout.addWidget(
            self.confusion_label
        )

        main_layout.addWidget(results_card)

        main_layout.addStretch()

        # Feature Selection
        feature_row = QHBoxLayout()

        feature_label = QLabel(
            "Feature Selection"
        )

        feature_label.setFixedWidth(130)

        self.feature_selection_combo = QComboBox()

        self.feature_selection_combo.addItems(
            SUPPORTED_FEATURE_SELECTION
        )

        feature_row.addWidget(
            feature_label
        )

        feature_row.addWidget(
            self.feature_selection_combo
        )

        config_layout.addLayout(
            feature_row
        )

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    def _create_metric(self, title):
        label = QLabel(
            f"{title}\n--"
        )

        label.setAlignment(
            Qt.AlignCenter
        )

        label.setMinimumHeight(70)

        return label

    # ---------------------------------------------------------
    # Signals
    # ---------------------------------------------------------

    def _connect_signals(self):

        self.dataset_button.clicked.connect(
            self._select_dataset
        )

        self.train_button.clicked.connect(
            self._start_training
        )

        self.target_combo.currentTextChanged.connect(
            self._target_changed
        )

        self.controller.status.connect(
            self._on_status
        )

        self.controller.progress.connect(
            self._on_progress
        )

        self.controller.finished.connect(
            self._on_training_finished
        )

        self.controller.error.connect(
            self._on_training_error
        )

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    def _select_dataset(self):

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Dataset",
            "",
            "CSV Files (*.csv)",
        )

        if not file_path:
            return

        try:
            df = pd.read_csv(file_path)

            if df.empty:
                raise ValueError(
                    "The selected dataset is empty."
                )

            self.dataset = df
            self.dataset_path = Path(file_path)

            self.dataset_label.setText(
                self.dataset_path.name
            )

            self.target_combo.clear()

            self.target_combo.addItems(
                list(df.columns)
            )

            self.target_combo.setEnabled(True)

            self.status_label.setText(
                f"Loaded {len(df)} rows."
            )

            self.progress_bar.setValue(0)

            self._clear_results()

        except Exception as exc:

            QMessageBox.critical(
                self,
                "Dataset Error",
                str(exc),
            )

    # ---------------------------------------------------------
    # Target
    # ---------------------------------------------------------

    def _target_changed(self, column):

        if column:
            self.status_label.setText(
                f"Target: {column}"
            )

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    def _start_training(self):

        if self.dataset is None:
            QMessageBox.warning(
                self,
                "Dataset Required",
                "Please select a CSV dataset first.",
            )
            return

        target_column = (
            self.target_combo.currentText()
        )

        if not target_column:
            QMessageBox.warning(
                self,
                "Target Required",
                "Please select the target column.",
            )
            return

        model_name = (
            self.model_combo.currentText()
        )

        feature_selection = (
            self.feature_selection_combo.currentText()
        )

        self.train_button.setEnabled(False)
        self.dataset_button.setEnabled(False)
        self.model_combo.setEnabled(False)
        self.target_combo.setEnabled(False)
        self.feature_selection_combo.setEnabled(False)

        self.progress_bar.setValue(0)

        self.status_label.setText(
            f"Starting {model_name}..."
        )

        self._clear_results()

        self.controller.start_training(
            dataframe=self.dataset,
            target_column=target_column,
            model_name=model_name,
            feature_selection=feature_selection,
        )


    # ---------------------------------------------------------
    # Worker callbacks
    # ---------------------------------------------------------

    def _on_status(self, message):

        self.status_label.setText(
            message
        )

    def _on_progress(self, value):

        self.progress_bar.setValue(
            value
        )

    def _on_training_finished(self, result):

        self.train_button.setEnabled(True)
        self.dataset_button.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.target_combo.setEnabled(True)
        self.feature_selection_combo.setEnabled(True)


        metrics = result["metrics"]

        selected_features = result[
            "selected_features"
        ]

        feature_selection = result[
            "feature_selection"
        ]

        self.features_label.setText(
            "Feature Selection: "
            f"{feature_selection}\n\n"
            "Selected Features:\n"
            + ", ".join(selected_features)
        )

        self.accuracy_label.setText(
            f"Accuracy\n{metrics['accuracy']:.4f}"
        )

        self.precision_label.setText(
            f"Precision\n{metrics['precision']:.4f}"
        )

        self.recall_label.setText(
            f"Recall\n{metrics['recall']:.4f}"
        )

        self.f1_label.setText(
            f"F1 Score\n{metrics['f1']:.4f}"
        )

        confusion_matrix = (
            result["confusion_matrix"]
        )

        self.confusion_label.setText(
            "Confusion Matrix\n\n"
            + str(confusion_matrix)
        )

        self.status_label.setText(
            "Training completed successfully."
        )

        self.progress_bar.setValue(100)

    def _on_training_error(self, message):

        self.train_button.setEnabled(True)
        self.dataset_button.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.target_combo.setEnabled(True)
        self.feature_selection_combo.setEnabled(True)

        self.status_label.setText(
            "Training failed."
        )

        QMessageBox.critical(
            self,
            "Training Error",
            message,
        )

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    def _clear_results(self):

        self.accuracy_label.setText(
            "Accuracy\n--"
        )

        self.precision_label.setText(
            "Precision\n--"
        )

        self.recall_label.setText(
            "Recall\n--"
        )

        self.f1_label.setText(
            "F1 Score\n--"
        )

        self.confusion_label.setText(
            "Confusion Matrix\nNo results yet."
        )
        self.features_label.setText(
            "Selected Features\nNo results yet."
        )