import ast

import pandas as pd

from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QPushButton,
    QComboBox,
    QLineEdit,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QFrame,
    QProgressBar,
    QCheckBox,
    QMessageBox,
    QTextEdit,
)

from app.workers.classification_controller import ClassificationController
from .hyperparameter_dialog import HyperparameterConfigDialog


class ClassificationPage(QWidget):

    def __init__(self):
        super().__init__()

        self.setObjectName("ClassificationPage")

        # =================================================
        # State
        # =================================================

        self.dataframe = None
        self.dataset_path = None

        # Stores Hyperparameter Tuning configuration
        self.tuning_params = None

        # =================================================
        # Controller
        # =================================================

        self.controller = ClassificationController(self)

        # =================================================
        # UI
        # =================================================

        self._build_ui()
        self._connect_controller()

    # =====================================================
    # UI
    # =====================================================

    def _build_ui(self):

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(30, 25, 30, 25)
        main_layout.setSpacing(20)

        # -------------------------------------------------
        # Header
        # -------------------------------------------------

        title = QLabel("Classification")
        title.setObjectName("pageTitle")

        subtitle = QLabel(
            "Build and evaluate a classification model"
        )
        subtitle.setObjectName("pageSubtitle")

        main_layout.addWidget(title)
        main_layout.addWidget(subtitle)

        # -------------------------------------------------
        # Configuration Card
        # -------------------------------------------------

        config_card = QFrame()
        config_card.setObjectName("configCard")

        config_layout = QVBoxLayout(config_card)
        config_layout.setContentsMargins(
            25,
            25,
            25,
            25,
        )
        config_layout.setSpacing(18)

        # =================================================
        # Dataset
        # =================================================

        dataset_label = QLabel("Dataset")
        dataset_label.setObjectName("sectionTitle")

        dataset_layout = QHBoxLayout()

        self.dataset_input = QLineEdit()

        self.dataset_input.setPlaceholderText(
            "Select a CSV dataset..."
        )

        self.dataset_input.setReadOnly(True)

        self.browse_button = QPushButton("Browse")

        self.browse_button.clicked.connect(
            self._browse_dataset
        )

        dataset_layout.addWidget(
            self.dataset_input
        )

        dataset_layout.addWidget(
            self.browse_button
        )

        config_layout.addWidget(
            dataset_label
        )

        config_layout.addLayout(
            dataset_layout
        )

        # =================================================
        # Target Column
        # =================================================

        target_label = QLabel("Target Column")
        target_label.setObjectName("fieldLabel")

        self.target_combo = QComboBox()

        self.target_combo.setPlaceholderText(
            "Select target column..."
        )

        config_layout.addWidget(
            target_label
        )

        config_layout.addWidget(
            self.target_combo
        )

        # =================================================
        # Model + Feature Selection
        # =================================================

        grid = QGridLayout()

        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(10)

        # -------------------------------------------------
        # Model
        # -------------------------------------------------

        model_label = QLabel("Model")
        model_label.setObjectName("fieldLabel")

        self.model_combo = QComboBox()

        self.model_combo.addItems([
            "KNN",
            "SVM",
            "MLP",
            "Random Forest",
            "Naive Bayes",
            "Decision Tree",
        ])

        # -------------------------------------------------
        # Feature Selection
        # -------------------------------------------------

        feature_label = QLabel(
            "Feature Selection"
        )

        feature_label.setObjectName(
            "fieldLabel"
        )

        self.feature_combo = QComboBox()

        self.feature_combo.addItems([
            "No Selection",
            "RFE",
            "Chi-Square",
            "ANOVA",
            "Mutual Information",
            "GA",
            "PSO",
        ])

        grid.addWidget(
            model_label,
            0,
            0,
        )

        grid.addWidget(
            feature_label,
            0,
            1,
        )

        grid.addWidget(
            self.model_combo,
            1,
            0,
        )

        grid.addWidget(
            self.feature_combo,
            1,
            1,
        )

        config_layout.addLayout(
            grid
        )

        # =================================================
        # Hyperparameter Tuning
        # =================================================

        tuning_title = QLabel(
            "Hyperparameter Tuning"
        )

        tuning_title.setObjectName(
            "sectionTitle"
        )

        config_layout.addWidget(
            tuning_title
        )

        tuning_layout = QHBoxLayout()

        # -------------------------------------------------
        # Enable Checkbox
        # -------------------------------------------------

        self.tuning_checkbox = QCheckBox(
            "Enable Hyperparameter Tuning"
        )

        # -------------------------------------------------
        # Method
        # -------------------------------------------------

        self.tuning_method_combo = QComboBox()

        self.tuning_method_combo.addItems([
            "Grid Search",
            "Random Search",
        ])

        self.tuning_method_combo.setEnabled(
            False
        )

        # -------------------------------------------------
        # Configure Button
        # -------------------------------------------------

        self.tuning_config_button = QPushButton(
            "Configure"
        )

        self.tuning_config_button.setEnabled(
            False
        )

        # -------------------------------------------------
        # Signals
        # -------------------------------------------------

        self.tuning_checkbox.toggled.connect(
            self._toggle_tuning
        )

        self.tuning_config_button.clicked.connect(
            self._configure_tuning
        )

        self.model_combo.currentTextChanged.connect(
            self._model_changed
        )

        # -------------------------------------------------
        # Layout
        # -------------------------------------------------

        tuning_layout.addWidget(
            self.tuning_checkbox
        )

        tuning_layout.addWidget(
            self.tuning_method_combo
        )

        tuning_layout.addWidget(
            self.tuning_config_button
        )

        tuning_layout.addStretch()

        config_layout.addLayout(
            tuning_layout
        )

        # =================================================
        # Model Parameters
        # =================================================

        parameters_label = QLabel(
            "Model Parameters"
        )

        parameters_label.setObjectName(
            "sectionTitle"
        )

        self.parameters_input = QLineEdit()

        self.parameters_input.setPlaceholderText(
            "Optional dictionary, "
            "e.g. {'n_estimators': 100}"
        )

        config_layout.addWidget(
            parameters_label
        )

        config_layout.addWidget(
            self.parameters_input
        )

        # =================================================
        # Training Buttons
        # =================================================

        buttons_layout = QHBoxLayout()

        buttons_layout.addStretch()

        self.cancel_button = QPushButton(
            "Cancel"
        )

        self.cancel_button.setEnabled(
            False
        )

        self.start_button = QPushButton(
            "Start Training"
        )

        self.start_button.setObjectName(
            "primaryButton"
        )

        self.start_button.clicked.connect(
            self._start_training
        )

        self.cancel_button.clicked.connect(
            self._cancel_training
        )

        buttons_layout.addWidget(
            self.cancel_button
        )

        buttons_layout.addWidget(
            self.start_button
        )

        config_layout.addLayout(
            buttons_layout
        )

        main_layout.addWidget(
            config_card
        )

        # =================================================
        # Training Status
        # =================================================

        status_card = QFrame()

        status_card.setObjectName(
            "statusCard"
        )

        status_layout = QVBoxLayout(
            status_card
        )

        status_layout.setContentsMargins(
            20,
            15,
            20,
            15,
        )

        self.status_label = QLabel(
            "Status: Ready"
        )

        self.progress_bar = QProgressBar()

        self.progress_bar.setRange(
            0,
            100,
        )

        self.progress_bar.setValue(
            0
        )

        status_layout.addWidget(
            self.status_label
        )

        status_layout.addWidget(
            self.progress_bar
        )

        main_layout.addWidget(
            status_card
        )

        # =================================================
        # Results
        # =================================================

        results_card = QFrame()

        results_card.setObjectName(
            "resultsCard"
        )

        results_layout = QVBoxLayout(
            results_card
        )

        results_layout.setContentsMargins(
            20,
            20,
            20,
            20,
        )

        results_title = QLabel(
            "Results"
        )

        results_title.setObjectName(
            "sectionTitle"
        )

        self.results_text = QTextEdit()

        self.results_text.setReadOnly(
            True
        )

        self.results_text.setPlaceholderText(
            "Training results will appear here..."
        )

        results_layout.addWidget(
            results_title
        )

        results_layout.addWidget(
            self.results_text
        )

        main_layout.addWidget(
            results_card
        )

        main_layout.addStretch()

    # =====================================================
    # Controller
    # =====================================================

    def _connect_controller(self):

        self.controller.finished.connect(
            self._training_finished
        )

        self.controller.error.connect(
            self._training_error
        )

        self.controller.progress.connect(
            self._update_progress
        )

        self.controller.status.connect(
            self._update_status
        )

    # =====================================================
    # Dataset
    # =====================================================

    def _browse_dataset(self):

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Dataset",
            "",
            "CSV Files (*.csv)",
        )

        if not file_path:
            return

        try:

            dataframe = pd.read_csv(
                file_path
            )

        except Exception as exc:

            QMessageBox.critical(
                self,
                "Dataset Error",
                f"Could not load dataset:\n\n{exc}",
            )

            return

        if dataframe.empty:

            QMessageBox.warning(
                self,
                "Dataset Error",
                "The selected dataset is empty.",
            )

            return

        # -------------------------------------------------
        # Store dataset
        # -------------------------------------------------

        self.dataframe = dataframe
        self.dataset_path = file_path

        self.dataset_input.setText(
            file_path
        )

        # -------------------------------------------------
        # Target columns
        # -------------------------------------------------

        self.target_combo.clear()

        self.target_combo.addItems(
            [
                str(column)
                for column in dataframe.columns
            ]
        )

        # -------------------------------------------------
        # Select last column
        # -------------------------------------------------

        if len(dataframe.columns) > 0:

            self.target_combo.setCurrentIndex(
                len(dataframe.columns) - 1
            )

        # -------------------------------------------------
        # Reset state
        # -------------------------------------------------

        self.tuning_params = None

        self.results_text.clear()

        self.progress_bar.setValue(
            0
        )

        self.status_label.setText(
            f"Status: Dataset loaded "
            f"({dataframe.shape[0]} rows, "
            f"{dataframe.shape[1]} columns)"
        )

    # =====================================================
    # Hyperparameter Tuning
    # =====================================================

    def _toggle_tuning(self, enabled):

        self.tuning_method_combo.setEnabled(
            enabled
        )

        self.tuning_config_button.setEnabled(
            enabled
        )

        if not enabled:

            self.tuning_params = None

            self.status_label.setText(
                "Status: Hyperparameter tuning disabled."
            )

    # -----------------------------------------------------

    def _model_changed(self, model_name):

        # The search space belongs to a specific model.
        # Therefore, changing the model invalidates
        # the previous configuration.

        self.tuning_params = None

        if self.tuning_checkbox.isChecked():

            self.status_label.setText(
                "Status: Model changed. "
                "Please configure hyperparameters again."
            )

    # -----------------------------------------------------

    def _configure_tuning(self):

        model_name = (
            self.model_combo.currentText()
        )

        tuning_method = (
            self.tuning_method_combo.currentText()
        )

        dialog = HyperparameterConfigDialog(
            model_name=model_name,
            tuning_method=tuning_method,
            parent=self,
        )

        if dialog.exec():

            self.tuning_params = (
                dialog.get_config()
            )

            if self.tuning_params:

                self.status_label.setText(
                    "Status: Hyperparameter "
                    "configuration saved."
                )

    # =====================================================
    # Training
    # =====================================================

    def _start_training(self):

        # -------------------------------------------------
        # Validate dataset
        # -------------------------------------------------

        if self.dataframe is None:

            QMessageBox.warning(
                self,
                "Dataset Required",
                "Please select a CSV dataset first.",
            )

            return

        # -------------------------------------------------
        # Validate target
        # -------------------------------------------------

        target_column = (
            self.target_combo.currentText()
        )

        if not target_column:

            QMessageBox.warning(
                self,
                "Target Required",
                "Please select a target column.",
            )

            return

        # -------------------------------------------------
        # Model
        # -------------------------------------------------

        model_name = (
            self.model_combo.currentText()
        )

        # -------------------------------------------------
        # Feature Selection
        # -------------------------------------------------

        feature_selection = (
            self.feature_combo.currentText()
        )

        # -------------------------------------------------
        # Model Parameters
        # -------------------------------------------------

        model_params = {}

        parameters_text = (
            self.parameters_input.text().strip()
        )

        if parameters_text:

            try:

                model_params = ast.literal_eval(
                    parameters_text
                )

                if not isinstance(
                    model_params,
                    dict,
                ):

                    raise ValueError(
                        "Model parameters must be "
                        "a dictionary."
                    )

            except Exception as exc:

                QMessageBox.warning(
                    self,
                    "Invalid Parameters",
                    f"Invalid model parameters:\n\n{exc}",
                )

                return

        # -------------------------------------------------
        # Hyperparameter Tuning
        # -------------------------------------------------

        hyperparameter_tuning = (
            self.tuning_checkbox.isChecked()
        )

        tuning_method = (
            self.tuning_method_combo.currentText()
        )

        tuning_params = (
            self.tuning_params
        )

        # -------------------------------------------------
        # Validate tuning configuration
        # -------------------------------------------------

        if (
            hyperparameter_tuning
            and not tuning_params
        ):

            QMessageBox.warning(
                self,
                "Hyperparameter Tuning",
                "Please configure the hyperparameter "
                "search space first.",
            )

            return

        # -------------------------------------------------
        # Disable UI
        # -------------------------------------------------

        self._set_training_state(
            running=True
        )

        self.progress_bar.setValue(
            0
        )

        self.status_label.setText(
            "Status: Starting training..."
        )

        self.results_text.clear()

        # -------------------------------------------------
        # Start Controller
        # -------------------------------------------------

        self.controller.start_training(

            dataframe=self.dataframe,

            target_column=target_column,

            model_name=model_name,

            model_params=model_params,

            feature_selection=feature_selection,

            feature_selection_params={},

            hyperparameter_tuning=(
                hyperparameter_tuning
            ),

            tuning_method=tuning_method,

            tuning_params=tuning_params,
        )

    # =====================================================
    # Cancel
    # =====================================================

    def _cancel_training(self):

        self.controller.cancel()

        self.status_label.setText(
            "Status: Cancelling..."
        )

        self.cancel_button.setEnabled(
            False
        )

    # =====================================================
    # Training State
    # =====================================================

    def _set_training_state(
        self,
        running,
    ):

        self.start_button.setEnabled(
            not running
        )

        self.cancel_button.setEnabled(
            running
        )

        self.browse_button.setEnabled(
            not running
        )

        self.target_combo.setEnabled(
            not running
        )

        self.model_combo.setEnabled(
            not running
        )

        self.feature_combo.setEnabled(
            not running
        )

        self.parameters_input.setEnabled(
            not running
        )

        self.tuning_checkbox.setEnabled(
            not running
        )

        if running:

            self.tuning_method_combo.setEnabled(
                False
            )

            self.tuning_config_button.setEnabled(
                False
            )

        else:

            tuning_enabled = (
                self.tuning_checkbox.isChecked()
            )

            self.tuning_method_combo.setEnabled(
                tuning_enabled
            )

            self.tuning_config_button.setEnabled(
                tuning_enabled
            )

    # =====================================================
    # Controller Signals
    # =====================================================

    def _update_progress(
        self,
        value,
    ):

        self.progress_bar.setValue(
            value
        )

    # -----------------------------------------------------

    def _update_status(
        self,
        message,
    ):

        self.status_label.setText(
            f"Status: {message}"
        )

    # =====================================================
    # Training Finished
    # =====================================================

    def _training_finished(
        self,
        result,
    ):

        self._set_training_state(
            running=False
        )

        self.progress_bar.setValue(
            100
        )

        self.status_label.setText(
            "Status: Training completed."
        )

        self._display_results(
            result
        )

    # =====================================================
    # Training Error
    # =====================================================

    def _training_error(
        self,
        message,
    ):

        self._set_training_state(
            running=False
        )

        self.status_label.setText(
            "Status: Training failed."
        )

        self.results_text.setPlainText(
            f"Training Error:\n\n{message}"
        )

        QMessageBox.critical(
            self,
            "Training Error",
            message,
        )

    # =====================================================
    # Results
    # =====================================================

    def _display_results(
        self,
        result,
    ):

        lines = []

        lines.append(
            "CLASSIFICATION RESULTS"
        )

        lines.append(
            "=" * 50
        )

        lines.append("")

        # -------------------------------------------------
        # Model
        # -------------------------------------------------

        lines.append(
            f"Model: "
            f"{result.get('model_name', '-')}"
        )

        lines.append(
            f"Feature Selection: "
            f"{result.get('feature_selection', '-')}"
        )

        lines.append("")

        # -------------------------------------------------
        # Selected Features
        # -------------------------------------------------

        lines.append(
            "Selected Features:"
        )

        selected_features = result.get(
            "selected_features",
            [],
        )

        if selected_features:

            for feature in selected_features:

                lines.append(
                    f"  • {feature}"
                )

        else:

            lines.append(
                "  None"
            )

        lines.append("")

        # -------------------------------------------------
        # Metrics
        # -------------------------------------------------

        lines.append(
            "Test Metrics:"
        )

        metrics = result.get(
            "metrics",
            {},
        )

        for metric_name, value in metrics.items():

            try:

                formatted_value = (
                    f"{float(value) * 100:.2f}%"
                )

            except (
                TypeError,
                ValueError,
            ):

                formatted_value = str(
                    value
                )

            lines.append(
                f"  {metric_name.capitalize()}: "
                f"{formatted_value}"
            )

        lines.append("")

        # -------------------------------------------------
        # Hyperparameter Tuning
        # -------------------------------------------------

        tuning_enabled = result.get(
            "hyperparameter_tuning",
            False,
        )

        lines.append(
            f"Hyperparameter Tuning: "
            f"{'Enabled' if tuning_enabled else 'Disabled'}"
        )

        if tuning_enabled:

            lines.append(
                f"Tuning Method: "
                f"{result.get('tuning_method', '-')}"
            )

            lines.append("")

            # -------------------------------------------------
            # Best Parameters
            # -------------------------------------------------

            lines.append(
                "Best Parameters:"
            )

            best_params = result.get(
                "best_params",
                {},
            )

            if best_params:

                for name, value in (
                    best_params.items()
                ):

                    lines.append(
                        f"  {name}: {value}"
                    )

            else:

                lines.append(
                    "  None"
                )

            # -------------------------------------------------
            # Best CV Score
            # -------------------------------------------------

            best_cv_score = result.get(
                "best_cv_score"
            )

            if best_cv_score is not None:

                lines.append("")

                lines.append(
                    f"Best CV Score: "
                    f"{float(best_cv_score) * 100:.2f}%"
                )

        lines.append("")

        # -------------------------------------------------
        # Confusion Matrix
        # -------------------------------------------------

        lines.append(
            "Confusion Matrix:"
        )

        confusion_matrix = result.get(
            "confusion_matrix"
        )

        if confusion_matrix is not None:

            for row in confusion_matrix:

                lines.append(
                    "  " + str(list(row))
                )

        else:

            lines.append(
                "  Not available"
            )

        # -------------------------------------------------
        # Display
        # -------------------------------------------------

        self.results_text.setPlainText(
            "\n".join(lines)
        )