from pathlib import Path

import pandas as pd

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QCheckBox,
    QProgressBar,
    QTableWidget,
    QTableWidgetItem,
    QFileDialog,
    QMessageBox,
    QScrollArea,
    QFrame,
    QAbstractItemView,
    QSizePolicy,
    QDialog,
)

from app.workers.automl_controller import AutoMLController
from app.pages.automl.hyperparameter_dialog import HyperparameterConfigDialog


class AutoMLPage(QWidget):
    """
    AutoML workspace.

    Provides:
    - Dataset loading
    - Target selection
    - Problem type selection
    - Feature selection
    - Model selection
    - Hyperparameter tuning
    - Training progress
    - Results comparison
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.dataframe = None
        self.dataset_path = None

        self.model_tuning_configs = {}

        self.controller = AutoMLController(self)

        self._build_ui()
        self._connect_signals()

    # ============================================================
    # UI
    # ============================================================

    def _build_ui(self):

        # --------------------------------------------------------
        # Root layout
        # --------------------------------------------------------

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # --------------------------------------------------------
        # Scroll Area
        # --------------------------------------------------------

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )

        root_layout.addWidget(self.scroll_area)

        # --------------------------------------------------------
        # Scroll Content
        # --------------------------------------------------------

        self.content_widget = QWidget()
        self.content_widget.setObjectName("AutoMLContent")

        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(24, 20, 24, 32)
        self.content_layout.setSpacing(18)

        self.scroll_area.setWidget(self.content_widget)

        # ========================================================
        # PAGE HEADER
        # ========================================================

        header_layout = QVBoxLayout()
        header_layout.setSpacing(6)

        self.page_title = QLabel("AutoML")
        self.page_title.setObjectName("PageTitle")

        self.page_subtitle = QLabel(
            "Automatically train and compare multiple machine learning models."
        )
        self.page_subtitle.setObjectName("PageSubtitle")

        self.page_subtitle.setWordWrap(True)

        header_layout.addWidget(self.page_title)
        header_layout.addWidget(self.page_subtitle)

        self.content_layout.addLayout(header_layout)

        # ========================================================
        # DATASET CARD
        # ========================================================

        dataset_card, dataset_layout = self._create_card(
            "Dataset",
            "Load a CSV dataset and select the target column."
        )

        # File row
        file_row = QHBoxLayout()
        file_row.setSpacing(10)

        self.dataset_label = QLabel("No dataset selected")
        self.dataset_label.setObjectName("SecondaryText")

        self.dataset_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred
        )

        self.browse_button = QPushButton("Browse Dataset")
        self.browse_button.setObjectName("PrimaryButton")
        self.browse_button.setMinimumHeight(40)

        file_row.addWidget(self.dataset_label)
        file_row.addWidget(self.browse_button)

        dataset_layout.addLayout(file_row)

        # Dataset information
        info_layout = QHBoxLayout()
        info_layout.setSpacing(30)

        self.rows_label = QLabel("Rows: -")
        self.columns_label = QLabel("Columns: -")

        self.rows_label.setObjectName("SecondaryText")
        self.columns_label.setObjectName("SecondaryText")

        info_layout.addWidget(self.rows_label)
        info_layout.addWidget(self.columns_label)
        info_layout.addStretch()

        dataset_layout.addLayout(info_layout)

        # Target
        target_layout = QGridLayout()
        target_layout.setHorizontalSpacing(12)
        target_layout.setVerticalSpacing(8)

        target_label = QLabel("Target Column")
        target_label.setObjectName("FieldLabel")

        self.target_combo = QComboBox()
        self.target_combo.setMinimumHeight(38)

        target_layout.addWidget(target_label, 0, 0)
        target_layout.addWidget(self.target_combo, 0, 1)

        target_layout.setColumnStretch(1, 1)

        dataset_layout.addLayout(target_layout)

        self.content_layout.addWidget(dataset_card)

        # ========================================================
        # CONFIGURATION CARD
        # ========================================================

        config_card, config_layout = self._create_card(
            "AutoML Configuration",
            "Configure preprocessing, feature selection and tuning."
        )

        # --------------------------------------------------------
        # Problem Type
        # --------------------------------------------------------

        problem_grid = QGridLayout()
        problem_grid.setHorizontalSpacing(16)
        problem_grid.setVerticalSpacing(12)

        problem_label = QLabel("Problem Type")
        problem_label.setObjectName("FieldLabel")

        self.problem_type_combo = QComboBox()
        self.problem_type_combo.addItems(
            [
                "Classification",
                "Regression",
            ]
        )
        self.problem_type_combo.setMinimumHeight(38)

        problem_grid.addWidget(problem_label, 0, 0)
        problem_grid.addWidget(self.problem_type_combo, 0, 1)

        # --------------------------------------------------------
        # Feature Selection
        # --------------------------------------------------------

        feature_label = QLabel("Feature Selection")
        feature_label.setObjectName("FieldLabel")

        self.feature_selection_combo = QComboBox()
        self.feature_selection_combo.addItems(
            [
                "No Selection",
                "RFE",
                "Chi-Square",
                "ANOVA",
                "Mutual Information",
                "Genetic Algorithm",
                "Particle Swarm Optimization",
            ]
        )
        self.feature_selection_combo.setMinimumHeight(38)

        problem_grid.addWidget(feature_label, 1, 0)
        problem_grid.addWidget(
            self.feature_selection_combo,
            1,
            1
        )

        problem_grid.setColumnStretch(1, 1)

        config_layout.addLayout(problem_grid)

        # --------------------------------------------------------
        # Separator
        # --------------------------------------------------------

        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setObjectName("SectionSeparator")

        config_layout.addWidget(separator)

        # --------------------------------------------------------
        # Hyperparameter Tuning
        # --------------------------------------------------------

        tuning_title = QLabel("Hyperparameter Tuning")
        tuning_title.setObjectName("SubsectionTitle")

        config_layout.addWidget(tuning_title)

        tuning_grid = QGridLayout()
        tuning_grid.setHorizontalSpacing(12)
        tuning_grid.setVerticalSpacing(10)

        self.tuning_checkbox = QCheckBox(
            "Enable Hyperparameter Tuning"
        )
        self.tuning_checkbox.setMinimumHeight(32)

        self.tuning_method_combo = QComboBox()
        self.tuning_method_combo.addItems(
            [
                "Grid Search",
                "Random Search",
            ]
        )
        self.tuning_method_combo.setEnabled(False)
        self.tuning_method_combo.setMinimumHeight(38)

        self.configure_tuning_button = QPushButton("Configure")
        self.configure_tuning_button.setObjectName("SecondaryButton")
        self.configure_tuning_button.setEnabled(False)
        self.configure_tuning_button.setMinimumHeight(38)

        tuning_grid.addWidget(
            self.tuning_checkbox,
            0,
            0,
            1,
            2
        )

        method_label = QLabel("Search Method")
        method_label.setObjectName("FieldLabel")

        tuning_grid.addWidget(
            method_label,
            1,
            0
        )

        tuning_grid.addWidget(
            self.tuning_method_combo,
            1,
            1
        )

        tuning_grid.addWidget(
            self.configure_tuning_button,
            1,
            2
        )

        tuning_grid.setColumnStretch(1, 1)

        config_layout.addLayout(tuning_grid)

        # --------------------------------------------------------
        # Scoring
        # --------------------------------------------------------

        scoring_grid = QGridLayout()
        scoring_grid.setHorizontalSpacing(16)

        scoring_label = QLabel("Scoring")
        scoring_label.setObjectName("FieldLabel")

        self.scoring_combo = QComboBox()
        self.scoring_combo.addItems(
            [
                "accuracy",
                "f1",
                "precision",
                "recall",
            ]
        )
        self.scoring_combo.setMinimumHeight(38)

        scoring_grid.addWidget(
            scoring_label,
            0,
            0
        )

        scoring_grid.addWidget(
            self.scoring_combo,
            0,
            1
        )

        scoring_grid.setColumnStretch(1, 1)

        config_layout.addLayout(scoring_grid)

        self.content_layout.addWidget(config_card)

        # ========================================================
        # MODELS CARD
        # ========================================================

        models_card, models_layout = self._create_card(
            "Models",
            "Select the machine learning models to evaluate."
        )

        self.model_checkboxes = {}

        models_grid = QGridLayout()
        models_grid.setHorizontalSpacing(20)
        models_grid.setVerticalSpacing(10)

        model_names = [
            "KNN",
            "SVM",
            "MLP",
            "Random Forest",
            "Naive Bayes",
            "Decision Tree",
        ]

        for index, model_name in enumerate(model_names):

            checkbox = QCheckBox(model_name)
            checkbox.setChecked(True)
            checkbox.setMinimumHeight(34)

            self.model_checkboxes[model_name] = checkbox

            row = index // 3
            column = index % 3

            models_grid.addWidget(
                checkbox,
                row,
                column
            )

        for column in range(3):
            models_grid.setColumnStretch(column, 1)

        models_layout.addLayout(models_grid)

        self.content_layout.addWidget(models_card)

        # ========================================================
        # PROGRESS CARD
        # ========================================================

        progress_card, progress_layout = self._create_card(
            "Training Progress",
            "Monitor the AutoML training process."
        )

        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("StatusText")

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setMinimumHeight(22)
        self.progress_bar.setTextVisible(True)

        progress_layout.addWidget(self.status_label)
        progress_layout.addWidget(self.progress_bar)

        self.content_layout.addWidget(progress_card)

        # ========================================================
        # ACTIONS
        # ========================================================

        actions_layout = QHBoxLayout()
        actions_layout.setSpacing(10)

        actions_layout.addStretch()

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setObjectName("SecondaryButton")
        self.cancel_button.setMinimumSize(100, 40)
        self.cancel_button.setEnabled(False)

        self.start_button = QPushButton("Start AutoML")
        self.start_button.setObjectName("PrimaryButton")
        self.start_button.setMinimumSize(130, 40)

        actions_layout.addWidget(self.cancel_button)
        actions_layout.addWidget(self.start_button)

        self.content_layout.addLayout(actions_layout)

        # ========================================================
        # RESULTS CARD
        # ========================================================

        results_card, results_layout = self._create_card(
            "Results",
            "Compare the performance of the trained models."
        )

        # Best model
        best_model_layout = QHBoxLayout()

        best_label = QLabel("Best Model")
        best_label.setObjectName("FieldLabel")

        self.best_model_value = QLabel("-")
        self.best_model_value.setObjectName("BestModelValue")

        best_model_layout.addWidget(best_label)
        best_model_layout.addWidget(self.best_model_value)
        best_model_layout.addStretch()

        results_layout.addLayout(best_model_layout)

        # Results table
        self.results_table = QTableWidget()

        self.results_table.setColumnCount(8)

        self.results_table.setHorizontalHeaderLabels(
            [
                "Rank",
                "Model",
                "Score",
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "Status",
            ]
        )

        self.results_table.setMinimumHeight(260)

        self.results_table.setAlternatingRowColors(True)

        self.results_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )

        self.results_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )

        self.results_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )

        self.results_table.verticalHeader().setVisible(False)

        header = self.results_table.horizontalHeader()

        header.setStretchLastSection(True)

        for column in range(8):
            header.setSectionResizeMode(
                column,
                header.ResizeMode.Stretch
            )

        results_layout.addWidget(self.results_table)

        self.content_layout.addWidget(results_card)

        # Bottom spacing
        self.content_layout.addSpacing(20)

    # ============================================================
    # CARD CREATOR
    # ============================================================

    def _create_card(self, title, subtitle=None):

        card = QFrame()
        card.setObjectName("Card")

        card_layout = QVBoxLayout(card)

        card_layout.setContentsMargins(
            18,
            16,
            18,
            18
        )

        card_layout.setSpacing(12)

        # Header
        title_label = QLabel(title)
        title_label.setObjectName("SectionTitle")

        card_layout.addWidget(title_label)

        if subtitle:

            subtitle_label = QLabel(subtitle)
            subtitle_label.setObjectName("SectionSubtitle")
            subtitle_label.setWordWrap(True)

            card_layout.addWidget(subtitle_label)

        return card, card_layout

    # ============================================================
    # SIGNALS
    # ============================================================

    def _connect_signals(self):

        self.browse_button.clicked.connect(
            self._browse_dataset
        )

        self.problem_type_combo.currentTextChanged.connect(
            self._problem_type_changed
        )

        self.tuning_checkbox.toggled.connect(
            self._toggle_tuning
        )

        self.configure_tuning_button.clicked.connect(
            self._configure_tuning
        )

        self.start_button.clicked.connect(
            self._start_automl
        )

        self.cancel_button.clicked.connect(
            self._cancel_automl
        )

        for checkbox in self.model_checkboxes.values():

            checkbox.toggled.connect(
                self._model_selection_changed
            )

        self.controller.started.connect(
            self._on_started
        )

        self.controller.progress.connect(
            self.progress_bar.setValue
        )

        self.controller.status.connect(
            self.status_label.setText
        )

        self.controller.finished.connect(
            self._on_finished
        )

        self.controller.error.connect(
            self._on_error
        )

    # ============================================================
    # DATASET
    # ============================================================

    def _browse_dataset(self):

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Dataset",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )

        if not file_path:
            return

        try:

            dataframe = pd.read_csv(file_path)

            if dataframe.empty:
                raise ValueError(
                    "The selected dataset is empty."
                )

            self.dataframe = dataframe
            self.dataset_path = Path(file_path)

            self.dataset_label.setText(
                self.dataset_path.name
            )

            self.rows_label.setText(
                f"Rows: {len(dataframe):,}"
            )

            self.columns_label.setText(
                f"Columns: {len(dataframe.columns):,}"
            )

            self.target_combo.clear()

            self.target_combo.addItems(
                [
                    str(column)
                    for column in dataframe.columns
                ]
            )

            self.status_label.setText(
                "Dataset loaded successfully."
            )

            self.progress_bar.setValue(0)

        except Exception as exc:

            QMessageBox.critical(
                self,
                "Dataset Error",
                f"Could not load dataset:\n\n{exc}"
            )

    # ============================================================
    # MODEL SELECTION
    # ============================================================

    def _model_selection_changed(self):

        selected_models = [
            model_name
            for model_name, checkbox
            in self.model_checkboxes.items()
            if checkbox.isChecked()
        ]

        # Remove configurations for models
        # that are no longer selected.

        for model_name in list(
            self.model_tuning_configs.keys()
        ):

            if model_name not in selected_models:

                del self.model_tuning_configs[
                    model_name
                ]

    # ============================================================
    # PROBLEM TYPE
    # ============================================================

    def _problem_type_changed(self, problem_type):

        if problem_type == "Regression":

            self.start_button.setEnabled(False)

            self.status_label.setText(
                "Regression AutoML is not implemented yet."
            )

        else:

            if not self.controller.is_running():

                self.start_button.setEnabled(True)

                if self.dataframe is not None:

                    self.status_label.setText(
                        "Ready"
                    )

    # ============================================================
    # HYPERPARAMETER TOGGLE
    # ============================================================

    def _toggle_tuning(self, enabled):

        self.tuning_method_combo.setEnabled(
            enabled
        )

        self.configure_tuning_button.setEnabled(
            enabled
        )

        if not enabled:

            self.model_tuning_configs.clear()

            self.status_label.setText(
                "Hyperparameter tuning disabled."
            )

        else:

            self.status_label.setText(
                "Configure hyperparameters for the selected models."
            )

    # ============================================================
    # HYPERPARAMETER CONFIGURATION
    # ============================================================

    def _configure_tuning(self):

        selected_models = [
            model_name
            for model_name, checkbox
            in self.model_checkboxes.items()
            if checkbox.isChecked()
        ]

        if not selected_models:

            QMessageBox.warning(
                self,
                "Model Required",
                "Please select at least one model first."
            )

            return

        tuning_method = (
            self.tuning_method_combo.currentText()
        )

        configured_count = 0

        for model_name in selected_models:

            dialog = HyperparameterConfigDialog(
                model_name=model_name,
                tuning_method=tuning_method,
                parent=self,
            )

            result = dialog.exec()

            if result != QDialog.DialogCode.Accepted:

                return

            config = dialog.get_configuration()

            if not config:

                QMessageBox.warning(
                    self,
                    "Configuration Error",
                    f"No configuration was returned for {model_name}."
                )

                return

            self.model_tuning_configs[
                model_name
            ] = config

            configured_count += 1

        self.status_label.setText(
            "Hyperparameter configuration saved."
        )

        QMessageBox.information(
            self,
            "Configuration Saved",
            (
                "Hyperparameter configuration saved "
                f"for {configured_count} model(s)."
            ),
        )

    # ============================================================
    # START AUTOML
    # ============================================================

    def _start_automl(self):

        if self.dataframe is None:

            QMessageBox.warning(
                self,
                "Dataset Required",
                "Please select a dataset first."
            )

            return

        if self.problem_type_combo.currentText() != "Classification":

            QMessageBox.information(
                self,
                "Not Available",
                "Regression AutoML is not implemented yet."
            )

            return

        if not self.target_combo.currentText():

            QMessageBox.warning(
                self,
                "Target Required",
                "Please select a target column."
            )

            return

        selected_models = [
            model_name
            for model_name, checkbox
            in self.model_checkboxes.items()
            if checkbox.isChecked()
        ]

        if not selected_models:

            QMessageBox.warning(
                self,
                "Model Required",
                "Please select at least one model."
            )

            return

        # --------------------------------------------------------
        # Hyperparameter validation
        # --------------------------------------------------------

        tuning_enabled = (
            self.tuning_checkbox.isChecked()
        )

        tuning_params = None

        if tuning_enabled:

            missing_models = [
                model_name
                for model_name in selected_models
                if model_name not in self.model_tuning_configs
            ]

            if missing_models:

                QMessageBox.warning(
                    self,
                    "Configuration Required",
                    (
                        "Please configure hyperparameters for:\n\n"
                        + "\n".join(
                            f"• {model}"
                            for model in missing_models
                        )
                    ),
                )

                return

            tuning_params = {
                model_name: self.model_tuning_configs[
                    model_name
                ]
                for model_name in selected_models
            }

        # --------------------------------------------------------
        # Feature selection
        # --------------------------------------------------------

        feature_selection = (
            self.feature_selection_combo.currentText()
        )

        feature_selection_map = {
            "Genetic Algorithm": "GA",
            "Particle Swarm Optimization": "PSO",
        }

        feature_selection = feature_selection_map.get(
            feature_selection,
            feature_selection
        )

        # --------------------------------------------------------
        # Start
        # --------------------------------------------------------

        self.results_table.setRowCount(0)

        self.best_model_value.setText("-")

        self.progress_bar.setValue(0)

        self.status_label.setText(
            "Starting AutoML..."
        )

        try:

            self.controller.start(
                dataframe=self.dataframe,
                target_column=self.target_combo.currentText(),
                model_names=selected_models,
                feature_selection=feature_selection,
                hyperparameter_tuning=tuning_enabled,
                tuning_method=self.tuning_method_combo.currentText(),
                tuning_params=tuning_params,
                scoring=self.scoring_combo.currentText(),
                test_size=0.2,
                random_state=42,
            )

        except Exception as exc:

            QMessageBox.critical(
                self,
                "AutoML Error",
                str(exc)
            )

    # ============================================================
    # CANCEL
    # ============================================================

    def _cancel_automl(self):

        if self.controller.is_running():

            self.controller.cancel()

            self.status_label.setText(
                "Cancelling AutoML..."
            )

    # ============================================================
    # STARTED
    # ============================================================

    def _on_started(self):

        self.start_button.setEnabled(False)

        self.cancel_button.setEnabled(True)

        self.browse_button.setEnabled(False)

        self.status_label.setText(
            "Starting AutoML..."
        )

    # ============================================================
    # FINISHED
    # ============================================================

    def _on_finished(self, result):

        self.progress_bar.setValue(100)

        self.status_label.setText(
            "AutoML completed successfully."
        )

        best_model = result.get(
            "best_model",
            result.get("model", "-")
        )

        self.best_model_value.setText(
            str(best_model)
        )

        self._display_results(result)

        self._restore_controls()

    # ============================================================
    # ERROR
    # ============================================================

    def _on_error(self, message):

        self.status_label.setText(
            "AutoML failed."
        )

        QMessageBox.critical(
            self,
            "AutoML Error",
            message
        )

        self._restore_controls()

    # ============================================================
    # RESTORE
    # ============================================================

    def _restore_controls(self):

        self.start_button.setEnabled(
            self.problem_type_combo.currentText()
            == "Classification"
        )

        self.cancel_button.setEnabled(False)

        self.browse_button.setEnabled(True)

    # ============================================================
    # RESULTS
    # ============================================================

    def _display_results(self, result):

        results = result.get(
            "results",
            []
        )

        self.results_table.setRowCount(
            len(results)
        )

        for row, item in enumerate(results):

            values = [
                item.get("rank", row + 1),
                item.get("model", "-"),
                item.get("score", 0),
                item.get("accuracy", 0),
                item.get("precision", 0),
                item.get("recall", 0),
                item.get("f1", 0),
                item.get("status", "Completed"),
            ]

            for column, value in enumerate(values):

                if isinstance(value, float):

                    text = f"{value:.4f}"

                else:

                    text = str(value)

                table_item = QTableWidgetItem(
                    text
                )

                table_item.setTextAlignment(
                    Qt.AlignmentFlag.AlignCenter
                )

                self.results_table.setItem(
                    row,
                    column,
                    table_item
                )

        self.results_table.resizeRowsToContents()