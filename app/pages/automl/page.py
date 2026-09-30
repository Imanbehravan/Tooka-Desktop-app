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
QSpinBox,
QDoubleSpinBox,
)

from app.workers.automl_controller import AutoMLController
from app.workers.clustering_controller import ClusteringController

class AutoMLPage(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setObjectName("AutoMLPage")

        # =========================================================
        # State
        # =========================================================

        self.dataframe = None
        self.dataset_path = None

        self.model_tuning_configs = {}
        self.clustering_tuning_params = None

        # =========================================================
        # Controllers
        # =========================================================

        self.classification_controller = AutoMLController(self)
        self.clustering_controller = ClusteringController(self)

        # =========================================================
        # UI
        # =========================================================

        self._build_ui()
        self._connect_signals()

        # Initial state
        self._problem_type_changed(
            self.problem_type_combo.currentText()
        )

        self._clustering_algorithm_changed(
            self.clustering_algorithm_combo.currentText()
        )

        self._clustering_feature_changed(
            self.clustering_feature_combo.currentText()
        )

    # =============================================================
    # UI
    # =============================================================

    def _build_ui(self):

        root_layout = QVBoxLayout(self)

        root_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        root_layout.setSpacing(0)

        # =========================================================
        # Scroll Area
        # =========================================================

        self.scroll_area = QScrollArea()

        self.scroll_area.setWidgetResizable(True)

        self.scroll_area.setFrameShape(
            QFrame.Shape.NoFrame
        )

        self.scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        self.scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )

        root_layout.addWidget(
            self.scroll_area
        )

        # =========================================================
        # Content
        # =========================================================

        self.content_widget = QWidget()

        self.content_widget.setObjectName(
            "AutoMLContent"
        )

        self.content_layout = QVBoxLayout(
            self.content_widget
        )

        self.content_layout.setContentsMargins(
            24,
            20,
            24,
            32,
        )

        self.content_layout.setSpacing(
            18
        )

        self.scroll_area.setWidget(
            self.content_widget
        )

        # =========================================================
        # Header
        # =========================================================

        header_layout = QVBoxLayout()

        header_layout.setSpacing(6)

        self.page_title = QLabel(
            "AutoML"
        )

        self.page_title.setObjectName(
            "PageTitle"
        )

        self.page_subtitle = QLabel(
            "Train, configure and evaluate machine learning models."
        )

        self.page_subtitle.setObjectName(
            "PageSubtitle"
        )

        self.page_subtitle.setWordWrap(
            True
        )

        header_layout.addWidget(
            self.page_title
        )

        header_layout.addWidget(
            self.page_subtitle
        )

        self.content_layout.addLayout(
            header_layout
        )

        # =========================================================
        # Dataset Card
        # =========================================================

        dataset_card, dataset_layout = self._create_card(
            "Dataset",
            "Load a CSV dataset for your machine learning task."
        )

        file_row = QHBoxLayout()

        file_row.setSpacing(10)

        self.dataset_label = QLabel(
            "No dataset selected"
        )

        self.dataset_label.setObjectName(
            "SecondaryText"
        )

        self.dataset_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )

        self.browse_button = QPushButton(
            "Browse Dataset"
        )

        self.browse_button.setObjectName(
            "PrimaryButton"
        )

        self.browse_button.setMinimumHeight(
            40
        )

        file_row.addWidget(
            self.dataset_label
        )

        file_row.addWidget(
            self.browse_button
        )

        dataset_layout.addLayout(
            file_row
        )

        info_layout = QHBoxLayout()

        info_layout.setSpacing(
            30
        )

        self.rows_label = QLabel(
            "Rows: -"
        )

        self.columns_label = QLabel(
            "Columns: -"
        )

        self.rows_label.setObjectName(
            "SecondaryText"
        )

        self.columns_label.setObjectName(
            "SecondaryText"
        )

        info_layout.addWidget(
            self.rows_label
        )

        info_layout.addWidget(
            self.columns_label
        )

        info_layout.addStretch()

        dataset_layout.addLayout(
            info_layout
        )

        self.content_layout.addWidget(
            dataset_card
        )

        # =========================================================
        # Problem Type Card
        # =========================================================

        problem_card, problem_layout = self._create_card(
            "Problem Type",
            "Choose the machine learning problem you want to solve."
        )

        problem_grid = QGridLayout()

        problem_grid.setHorizontalSpacing(
            16
        )

        problem_grid.setVerticalSpacing(
            10
        )

        problem_label = QLabel(
            "Problem Type"
        )

        problem_label.setObjectName(
            "FieldLabel"
        )

        self.problem_type_combo = QComboBox()

        self.problem_type_combo.addItems(
            [
                "Classification",
                "Clustering",
            ]
        )

        self.problem_type_combo.setMinimumHeight(
            38
        )

        problem_grid.addWidget(
            problem_label,
            0,
            0,
        )

        problem_grid.addWidget(
            self.problem_type_combo,
            0,
            1,
        )

        problem_grid.setColumnStretch(
            1,
            1,
        )

        problem_layout.addLayout(
            problem_grid
        )

        self.content_layout.addWidget(
            problem_card
        )

        # =========================================================
        # Classification Configuration
        # =========================================================

        self.classification_card = QFrame()

        self.classification_card.setObjectName(
            "Card"
        )

        classification_layout = QVBoxLayout(
            self.classification_card
        )

        classification_layout.setContentsMargins(
            18,
            16,
            18,
            18,
        )

        classification_layout.setSpacing(
            14
        )

        classification_title = QLabel(
            "Classification Configuration"
        )

        classification_title.setObjectName(
            "SectionTitle"
        )

        classification_layout.addWidget(
            classification_title
        )

        classification_subtitle = QLabel(
            "Configure the target column, feature selection, "
            "models and hyperparameter tuning."
        )

        classification_subtitle.setObjectName(
            "SectionSubtitle"
        )

        classification_subtitle.setWordWrap(
            True
        )

        classification_layout.addWidget(
            classification_subtitle
        )

        # ---------------------------------------------------------
        # Target
        # ---------------------------------------------------------

        target_grid = QGridLayout()

        target_label = QLabel(
            "Target Column"
        )

        target_label.setObjectName(
            "FieldLabel"
        )

        self.target_combo = QComboBox()

        self.target_combo.setMinimumHeight(
            38
        )

        target_grid.addWidget(
            target_label,
            0,
            0,
        )

        target_grid.addWidget(
            self.target_combo,
            0,
            1,
        )

        target_grid.setColumnStretch(
            1,
            1,
        )

        classification_layout.addLayout(
            target_grid
        )

        # ---------------------------------------------------------
        # Feature Selection
        # ---------------------------------------------------------

        feature_grid = QGridLayout()

        feature_label = QLabel(
            "Feature Selection"
        )

        feature_label.setObjectName(
            "FieldLabel"
        )

        self.classification_feature_combo = QComboBox()

        self.classification_feature_combo.addItems(
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

        self.classification_feature_combo.setMinimumHeight(
            38
        )

        feature_grid.addWidget(
            feature_label,
            0,
            0,
        )

        feature_grid.addWidget(
            self.classification_feature_combo,
            0,
            1,
        )

        feature_grid.setColumnStretch(
            1,
            1,
        )

        classification_layout.addLayout(
            feature_grid
        )

        # ---------------------------------------------------------
        # Tuning
        # ---------------------------------------------------------

        tuning_title = QLabel(
            "Hyperparameter Tuning"
        )

        tuning_title.setObjectName(
            "SubsectionTitle"
        )

        classification_layout.addWidget(
            tuning_title
        )

        tuning_grid = QGridLayout()

        self.classification_tuning_checkbox = QCheckBox(
            "Enable Hyperparameter Tuning"
        )

        self.classification_tuning_method = QComboBox()

        self.classification_tuning_method.addItems(
            [
                "Grid Search",
                "Random Search",
            ]
        )

        self.classification_tuning_method.setEnabled(
            False
        )

        self.classification_tuning_button = QPushButton(
            "Configure"
        )

        self.classification_tuning_button.setObjectName(
            "SecondaryButton"
        )

        self.classification_tuning_button.setEnabled(
            False
        )

        tuning_grid.addWidget(
            self.classification_tuning_checkbox,
            0,
            0,
            1,
            2,
        )

        tuning_grid.addWidget(
            QLabel("Search Method"),
            1,
            0,
        )

        tuning_grid.addWidget(
            self.classification_tuning_method,
            1,
            1,
        )

        tuning_grid.addWidget(
            self.classification_tuning_button,
            1,
            2,
        )

        tuning_grid.setColumnStretch(
            1,
            1,
        )

        classification_layout.addLayout(
            tuning_grid
        )

        # ---------------------------------------------------------
        # Models
        # ---------------------------------------------------------

        models_title = QLabel(
            "Models"
        )

        models_title.setObjectName(
            "SubsectionTitle"
        )

        classification_layout.addWidget(
            models_title
        )

        self.model_checkboxes = {}

        models_grid = QGridLayout()

        model_names = [
            "KNN",
            "SVM",
            "MLP",
            "Random Forest",
            "Naive Bayes",
            "Decision Tree",
        ]

        for index, model_name in enumerate(
            model_names
        ):

            checkbox = QCheckBox(
                model_name
            )

            checkbox.setChecked(
                True
            )

            checkbox.setMinimumHeight(
                34
            )

            self.model_checkboxes[
                model_name
            ] = checkbox

            row = index // 3
            column = index % 3

            models_grid.addWidget(
                checkbox,
                row,
                column,
            )

        for column in range(3):

            models_grid.setColumnStretch(
                column,
                1,
            )

        classification_layout.addLayout(
            models_grid
        )

        # ---------------------------------------------------------
        # Scoring
        # ---------------------------------------------------------

        scoring_grid = QGridLayout()

        scoring_label = QLabel(
            "Scoring"
        )

        scoring_label.setObjectName(
            "FieldLabel"
        )

        self.scoring_combo = QComboBox()

        self.scoring_combo.addItems(
            [
                "accuracy",
                "f1",
                "precision",
                "recall",
            ]
        )

        self.scoring_combo.setMinimumHeight(
            38
        )

        scoring_grid.addWidget(
            scoring_label,
            0,
            0,
        )

        scoring_grid.addWidget(
            self.scoring_combo,
            0,
            1,
        )

        scoring_grid.setColumnStretch(
            1,
            1,
        )

        classification_layout.addLayout(
            scoring_grid
        )

        self.content_layout.addWidget(
            self.classification_card
        )

        # =========================================================
        # Clustering Configuration
        # =========================================================

        self.clustering_card = QFrame()

        self.clustering_card.setObjectName(
            "Card"
        )

        clustering_layout = QVBoxLayout(
            self.clustering_card
        )

        clustering_layout.setContentsMargins(
            18,
            16,
            18,
            18,
        )

        clustering_layout.setSpacing(
            14
        )

        clustering_title = QLabel(
            "Clustering Configuration"
        )

        clustering_title.setObjectName(
            "SectionTitle"
        )

        clustering_layout.addWidget(
            clustering_title
        )

        clustering_subtitle = QLabel(
            "Discover hidden groups and patterns in your dataset."
        )

        clustering_subtitle.setObjectName(
            "SectionSubtitle"
        )

        clustering_subtitle.setWordWrap(
            True
        )

        clustering_layout.addWidget(
            clustering_subtitle
        )

        # ---------------------------------------------------------
        # Algorithm / Feature Selection
        # ---------------------------------------------------------

        clustering_grid = QGridLayout()

        clustering_algorithm_label = QLabel(
            "Algorithm"
        )

        clustering_algorithm_label.setObjectName(
            "FieldLabel"
        )

        self.clustering_algorithm_combo = QComboBox()

        self.clustering_algorithm_combo.addItems(
            [
                "KMeans",
                "DBSCAN",
                "Agglomerative",
            ]
        )

        self.clustering_algorithm_combo.setMinimumHeight(
            38
        )

        clustering_feature_label = QLabel(
            "Feature Selection"
        )

        clustering_feature_label.setObjectName(
            "FieldLabel"
        )

        self.clustering_feature_combo = QComboBox()

        self.clustering_feature_combo.addItems(
            [
                "No Selection",
                "Variance Threshold",
                "PCA",
                "Feature Agglomeration",
            ]
        )

        self.clustering_feature_combo.setMinimumHeight(
            38
        )

        clustering_grid.addWidget(
            clustering_algorithm_label,
            0,
            0,
        )

        clustering_grid.addWidget(
            clustering_feature_label,
            0,
            1,
        )

        clustering_grid.addWidget(
            self.clustering_algorithm_combo,
            1,
            0,
        )

        clustering_grid.addWidget(
            self.clustering_feature_combo,
            1,
            1,
        )

        clustering_grid.setColumnStretch(
            0,
            1,
        )

        clustering_grid.setColumnStretch(
            1,
            1,
        )

        clustering_layout.addLayout(
            clustering_grid
        )

        # ---------------------------------------------------------
        # Algorithm Parameters
        # ---------------------------------------------------------

        clustering_parameters_title = QLabel(
            "Algorithm Parameters"
        )

        clustering_parameters_title.setObjectName(
            "SubsectionTitle"
        )

        clustering_layout.addWidget(
            clustering_parameters_title
        )

        # KMeans

        self.kmeans_frame = QFrame()

        kmeans_layout = QHBoxLayout(
            self.kmeans_frame
        )

        kmeans_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        kmeans_layout.addWidget(
            QLabel("Number of Clusters")
        )

        self.kmeans_clusters = QSpinBox()

        self.kmeans_clusters.setRange(
            2,
            100,
        )

        self.kmeans_clusters.setValue(
            3
        )

        kmeans_layout.addWidget(
            self.kmeans_clusters
        )

        kmeans_layout.addStretch()

        clustering_layout.addWidget(
            self.kmeans_frame
        )

        # DBSCAN

        self.dbscan_frame = QFrame()

        dbscan_layout = QGridLayout(
            self.dbscan_frame
        )

        dbscan_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        dbscan_layout.addWidget(
            QLabel("Epsilon"),
            0,
            0,
        )

        self.dbscan_eps = QDoubleSpinBox()

        self.dbscan_eps.setRange(
            0.01,
            1000.0,
        )

        self.dbscan_eps.setSingleStep(
            0.1
        )

        self.dbscan_eps.setValue(
            0.5
        )

        dbscan_layout.addWidget(
            self.dbscan_eps,
            0,
            1,
        )

        dbscan_layout.addWidget(
            QLabel("Min Samples"),
            1,
            0,
        )

        self.dbscan_min_samples = QSpinBox()

        self.dbscan_min_samples.setRange(
            1,
            1000,
        )

        self.dbscan_min_samples.setValue(
            5
        )

        dbscan_layout.addWidget(
            self.dbscan_min_samples,
            1,
            1,
        )

        clustering_layout.addWidget(
            self.dbscan_frame
        )

        # Agglomerative

        self.agglomerative_frame = QFrame()

        agglomerative_layout = QGridLayout(
            self.agglomerative_frame
        )

        agglomerative_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        agglomerative_layout.addWidget(
            QLabel("Number of Clusters"),
            0,
            0,
        )

        self.agg_clusters = QSpinBox()

        self.agg_clusters.setRange(
            2,
            100,
        )

        self.agg_clusters.setValue(
            3
        )

        agglomerative_layout.addWidget(
            self.agg_clusters,
            0,
            1,
        )

        agglomerative_layout.addWidget(
            QLabel("Linkage"),
            1,
            0,
        )

        self.linkage_combo = QComboBox()

        self.linkage_combo.addItems(
            [
                "ward",
                "complete",
                "average",
                "single",
            ]
        )

        agglomerative_layout.addWidget(
            self.linkage_combo,
            1,
            1,
        )

        clustering_layout.addWidget(
            self.agglomerative_frame
        )

        # ---------------------------------------------------------
        # Feature Selection Parameters
        # ---------------------------------------------------------

        feature_parameters_title = QLabel(
            "Feature Selection Parameters"
        )

        feature_parameters_title.setObjectName(
            "SubsectionTitle"
        )

        clustering_layout.addWidget(
            feature_parameters_title
        )

        self.variance_frame = QFrame()

        variance_layout = QHBoxLayout(
            self.variance_frame
        )

        variance_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        variance_layout.addWidget(
            QLabel("Threshold")
        )

        self.variance_threshold = QDoubleSpinBox()

        self.variance_threshold.setRange(
            0.0,
            100000.0,
        )

        self.variance_threshold.setDecimals(
            6
        )

        self.variance_threshold.setValue(
            0.0
        )

        variance_layout.addWidget(
            self.variance_threshold
        )

        variance_layout.addStretch()

        clustering_layout.addWidget(
            self.variance_frame
        )

        self.pca_frame = QFrame()

        pca_layout = QHBoxLayout(
            self.pca_frame
        )

        pca_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        pca_layout.addWidget(
            QLabel("Number of Components")
        )

        self.pca_components = QSpinBox()

        self.pca_components.setRange(
            1,
            100,
        )

        self.pca_components.setValue(
            2
        )

        pca_layout.addWidget(
            self.pca_components
        )

        pca_layout.addStretch()

        clustering_layout.addWidget(
            self.pca_frame
        )

        self.feature_agglomeration_frame = QFrame()

        fa_layout = QHBoxLayout(
            self.feature_agglomeration_frame
        )

        fa_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        fa_layout.addWidget(
            QLabel("Number of Clusters")
        )

        self.feature_agglomeration_clusters = QSpinBox()

        self.feature_agglomeration_clusters.setRange(
            2,
            100,
        )

        self.feature_agglomeration_clusters.setValue(
            2
        )

        fa_layout.addWidget(
            self.feature_agglomeration_clusters
        )

        fa_layout.addStretch()

        clustering_layout.addWidget(
            self.feature_agglomeration_frame
        )

        # ---------------------------------------------------------
        # Clustering Tuning
        # ---------------------------------------------------------

        clustering_tuning_title = QLabel(
            "Hyperparameter Tuning"
        )

        clustering_tuning_title.setObjectName(
            "SubsectionTitle"
        )

        clustering_layout.addWidget(
            clustering_tuning_title
        )

        clustering_tuning_layout = QHBoxLayout()

        self.clustering_tuning_checkbox = QCheckBox(
            "Enable Hyperparameter Tuning"
        )

        self.clustering_tuning_button = QPushButton(
            "Configure"
        )

        self.clustering_tuning_button.setObjectName(
            "SecondaryButton"
        )

        self.clustering_tuning_button.setEnabled(
            False
        )

        clustering_tuning_layout.addWidget(
            self.clustering_tuning_checkbox
        )

        clustering_tuning_layout.addWidget(
            self.clustering_tuning_button
        )

        clustering_tuning_layout.addStretch()

        clustering_layout.addLayout(
            clustering_tuning_layout
        )

        # ---------------------------------------------------------
        # Dimensionality Reduction
        # ---------------------------------------------------------

        self.dim_reduction_checkbox = QCheckBox(
            "Apply PCA Dimensionality Reduction"
        )

        clustering_layout.addWidget(
            self.dim_reduction_checkbox
        )

        self.content_layout.addWidget(
            self.clustering_card
        )

        # =========================================================
        # Training Card
        # =========================================================

        training_card, training_layout = self._create_card(
            "Training",
            "Start or cancel the current AutoML operation."
        )

        buttons_layout = QHBoxLayout()

        buttons_layout.addStretch()

        self.cancel_button = QPushButton(
            "Cancel"
        )

        self.cancel_button.setObjectName(
            "SecondaryButton"
        )

        self.cancel_button.setMinimumSize(
            100,
            40,
        )

        self.cancel_button.setEnabled(
            False
        )

        self.start_button = QPushButton(
            "Start AutoML"
        )

        self.start_button.setObjectName(
            "PrimaryButton"
        )

        self.start_button.setMinimumSize(
            130,
            40,
        )

        buttons_layout.addWidget(
            self.cancel_button
        )

        buttons_layout.addWidget(
            self.start_button
        )

        training_layout.addLayout(
            buttons_layout
        )

        self.content_layout.addWidget(
            training_card
        )

        # =========================================================
        # Progress Card
        # =========================================================

        progress_card, progress_layout = self._create_card(
            "Training Progress",
            "Monitor the current operation."
        )

        self.status_label = QLabel(
            "Ready"
        )

        self.status_label.setObjectName(
            "StatusText"
        )

        self.progress_bar = QProgressBar()

        self.progress_bar.setRange(
            0,
            100,
        )

        self.progress_bar.setValue(
            0
        )

        progress_layout.addWidget(
            self.status_label
        )

        progress_layout.addWidget(
            self.progress_bar
        )

        self.content_layout.addWidget(
            progress_card
        )

        # =========================================================
        # Classification Results
        # =========================================================

        self.classification_results_card = QFrame()

        self.classification_results_card.setObjectName(
            "Card"
        )

        classification_results_layout = QVBoxLayout(
            self.classification_results_card
        )

        classification_results_layout.setContentsMargins(
            18,
            16,
            18,
            18,
        )

        results_title = QLabel(
            "Classification Results"
        )

        results_title.setObjectName(
            "SectionTitle"
        )

        classification_results_layout.addWidget(
            results_title
        )

        best_model_layout = QHBoxLayout()

        best_model_layout.addWidget(
            QLabel("Best Model")
        )

        self.best_model_value = QLabel(
            "-"
        )

        self.best_model_value.setObjectName(
            "BestModelValue"
        )

        best_model_layout.addWidget(
            self.best_model_value
        )

        best_model_layout.addStretch()

        classification_results_layout.addLayout(
            best_model_layout
        )

        self.results_table = QTableWidget()

        self.results_table.setColumnCount(
            8
        )

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

        self.results_table.setMinimumHeight(
            260
        )

        self.results_table.setAlternatingRowColors(
            True
        )

        self.results_table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers
        )

        self.results_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )

        self.results_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )

        self.results_table.verticalHeader().setVisible(
            False
        )

        header = self.results_table.horizontalHeader()

        header.setStretchLastSection(
            True
        )

        for column in range(8):

            header.setSectionResizeMode(
                column,
                header.ResizeMode.Stretch,
            )

        classification_results_layout.addWidget(
            self.results_table
        )

        self.content_layout.addWidget(
            self.classification_results_card
        )

        # =========================================================
        # Clustering Results
        # =========================================================

        self.clustering_results_card = QFrame()

        self.clustering_results_card.setObjectName(
            "Card"
        )

        clustering_results_layout = QVBoxLayout(
            self.clustering_results_card
        )

        clustering_results_layout.setContentsMargins(
            18,
            16,
            18,
            18,
        )

        clustering_results_title = QLabel(
            "Clustering Results"
        )

        clustering_results_title.setObjectName(
            "SectionTitle"
        )

        clustering_results_layout.addWidget(
            clustering_results_title
        )

        self.clustering_results_text = QLabel(
            "No clustering results yet."
        )

        self.clustering_results_text.setWordWrap(
            True
        )

        self.clustering_results_text.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        clustering_results_layout.addWidget(
            self.clustering_results_text
        )

        self.content_layout.addWidget(
            self.clustering_results_card
        )

        self.content_layout.addSpacing(
            20
        )

    # =============================================================
    # Card
    # =============================================================

    def _create_card(
        self,
        title,
        subtitle=None,
    ):

        card = QFrame()

        card.setObjectName(
            "Card"
        )

        layout = QVBoxLayout(
            card
        )

        layout.setContentsMargins(
            18,
            16,
            18,
            18,
        )

        layout.setSpacing(
            12
        )

        title_label = QLabel(
            title
        )

        title_label.setObjectName(
            "SectionTitle"
        )

        layout.addWidget(
            title_label
        )

        if subtitle:

            subtitle_label = QLabel(
                subtitle
            )

            subtitle_label.setObjectName(
                "SectionSubtitle"
            )

            subtitle_label.setWordWrap(
                True
            )

            layout.addWidget(
                subtitle_label
            )

        return card, layout

    # =============================================================
    # Signals
    # =============================================================

    def _connect_signals(self):

        # Dataset

        self.browse_button.clicked.connect(
            self._browse_dataset
        )

        # Problem type

        self.problem_type_combo.currentTextChanged.connect(
            self._problem_type_changed
        )

        # Classification

        self.classification_tuning_checkbox.toggled.connect(
            self._classification_tuning_toggled
        )

        for checkbox in self.model_checkboxes.values():

            checkbox.toggled.connect(
                self._classification_model_changed
            )

        self.start_button.clicked.connect(
            self._start_automl
        )

        self.cancel_button.clicked.connect(
            self._cancel
        )

        # Clustering

        self.clustering_algorithm_combo.currentTextChanged.connect(
            self._clustering_algorithm_changed
        )

        self.clustering_feature_combo.currentTextChanged.connect(
            self._clustering_feature_changed
        )

        self.clustering_tuning_checkbox.toggled.connect(
            self._clustering_tuning_toggled
        )

        self.clustering_tuning_button.clicked.connect(
            self._configure_clustering_tuning
        )

        # Classification controller

        self.classification_controller.started.connect(
            self._classification_started
        )

        self.classification_controller.progress.connect(
            self._update_progress
        )

        self.classification_controller.status.connect(
            self._update_status
        )

        self.classification_controller.finished.connect(
            self._classification_finished
        )

        self.classification_controller.error.connect(
            self._operation_error
        )

        # Clustering controller

        self.clustering_controller.progress.connect(
            self._update_progress
        )

        self.clustering_controller.status.connect(
            self._update_status
        )

        self.clustering_controller.finished.connect(
            self._clustering_finished
        )

        self.clustering_controller.error.connect(
            self._operation_error
        )

    # =============================================================
    # Dataset
    # =============================================================

    def _browse_dataset(self):

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Dataset",
            "",
            "CSV Files (*.csv);;All Files (*)",
        )

        if not file_path:
            return

        try:

            dataframe = pd.read_csv(
                file_path
            )

            if dataframe.empty:

                raise ValueError(
                    "The selected dataset is empty."
                )

            self.dataframe = dataframe
            self.dataset_path = Path(
                file_path
            )

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

            self.model_tuning_configs.clear()
            self.clustering_tuning_params = None

            self.results_table.setRowCount(
                0
            )

            self.best_model_value.setText(
                "-"
            )

            self.clustering_results_text.setText(
                "No clustering results yet."
            )

            self.progress_bar.setValue(
                0
            )

            self.status_label.setText(
                "Dataset loaded successfully."
            )

        except Exception as exc:

            QMessageBox.critical(
                self,
                "Dataset Error",
                f"Could not load dataset:\n\n{exc}",
            )

    # =============================================================
    # Problem Type
    # =============================================================

    def _problem_type_changed(
        self,
        problem_type,
    ):

        classification = (
            problem_type == "Classification"
        )

        clustering = (
            problem_type == "Clustering"
        )

        self.classification_card.setVisible(
            classification
        )

        self.classification_results_card.setVisible(
            classification
        )

        self.clustering_card.setVisible(
            clustering
        )

        self.clustering_results_card.setVisible(
            clustering
        )

        if classification:

            self.start_button.setText(
                "Start Classification"
            )

            self.page_subtitle.setText(
                "Automatically train and compare classification models."
            )

        else:

            self.start_button.setText(
                "Start Clustering"
            )

            self.page_subtitle.setText(
                "Discover hidden groups and patterns in your data."
            )

        if not self._is_any_controller_running():

            self.start_button.setEnabled(
                True
            )

            self.cancel_button.setEnabled(
                False
            )

    # =============================================================
    # Classification Tuning
    # =============================================================

    def _classification_tuning_toggled(
        self,
        enabled,
    ):

        self.classification_tuning_method.setEnabled(
            enabled
        )

        self.classification_tuning_button.setEnabled(
            enabled
        )

        if not enabled:

            self.model_tuning_configs.clear()

            self.status_label.setText(
                "Hyperparameter tuning disabled."
            )

    # =============================================================
    # Classification Models
    # =============================================================

    def _classification_model_changed(
        self,
    ):

        selected_models = [
            name
            for name, checkbox
            in self.model_checkboxes.items()
            if checkbox.isChecked()
        ]

        for model_name in list(
            self.model_tuning_configs.keys()
        ):

            if model_name not in selected_models:

                del self.model_tuning_configs[
                    model_name
                ]

    # =============================================================
    # Classification Start
    # =============================================================

    def _start_classification(self):

        if self.dataframe is None:

            QMessageBox.warning(
                self,
                "Dataset Required",
                "Please select a dataset first.",
            )

            return

        target = (
            self.target_combo.currentText()
        )

        if not target:

            QMessageBox.warning(
                self,
                "Target Required",
                "Please select a target column.",
            )

            return

        selected_models = [
            name
            for name, checkbox
            in self.model_checkboxes.items()
            if checkbox.isChecked()
        ]

        if not selected_models:

            QMessageBox.warning(
                self,
                "Model Required",
                "Please select at least one model.",
            )

            return

        tuning_enabled = (
            self.classification_tuning_checkbox.isChecked()
        )

        tuning_params = None

        if tuning_enabled:

            missing = [
                model
                for model in selected_models
                if model not in self.model_tuning_configs
            ]

            if missing:

                QMessageBox.warning(
                    self,
                    "Configuration Required",
                    (
                        "Please configure hyperparameters for:\n\n"
                        + "\n".join(
                            f"• {model}"
                            for model in missing
                        )
                    ),
                )

                return

            tuning_params = {
                model: self.model_tuning_configs[
                    model
                ]
                for model in selected_models
            }

        feature_selection = (
            self.classification_feature_combo.currentText()
        )

        feature_selection_map = {
            "Genetic Algorithm": "GA",
            "Particle Swarm Optimization": "PSO",
        }

        feature_selection = (
            feature_selection_map.get(
                feature_selection,
                feature_selection,
            )
        )

        self.results_table.setRowCount(
            0
        )

        self.best_model_value.setText(
            "-"
        )

        self.progress_bar.setValue(
            0
        )

        self.status_label.setText(
            "Starting classification..."
        )

        try:

            self.classification_controller.start(
                dataframe=self.dataframe,
                target_column=target,
                model_names=selected_models,
                feature_selection=feature_selection,
                hyperparameter_tuning=tuning_enabled,
                tuning_method=(
                    self.classification_tuning_method.currentText()
                ),
                tuning_params=tuning_params,
                scoring=self.scoring_combo.currentText(),
                test_size=0.2,
                random_state=42,
            )

        except Exception as exc:

            QMessageBox.critical(
                self,
                "Classification Error",
                str(exc),
            )

    # =============================================================
    # Clustering Algorithm
    # =============================================================

    def _clustering_algorithm_changed(
        self,
        algorithm,
    ):

        self.kmeans_frame.setVisible(
            algorithm == "KMeans"
        )

        self.dbscan_frame.setVisible(
            algorithm == "DBSCAN"
        )

        self.agglomerative_frame.setVisible(
            algorithm == "Agglomerative"
        )

        self.clustering_tuning_params = None

    # =============================================================
    # Clustering Feature Selection
    # =============================================================

    def _clustering_feature_changed(
        self,
        method,
    ):

        self.variance_frame.setVisible(
            method == "Variance Threshold"
        )

        self.pca_frame.setVisible(
            method == "PCA"
        )

        self.feature_agglomeration_frame.setVisible(
            method == "Feature Agglomeration"
        )

    # =============================================================
    # Clustering Tuning
    # =============================================================

    def _clustering_tuning_toggled(
        self,
        enabled,
    ):

        self.clustering_tuning_button.setEnabled(
            enabled
        )

        if not enabled:

            self.clustering_tuning_params = None

    # =============================================================
    # Configure Clustering Tuning
    # =============================================================

    def _configure_clustering_tuning(
        self,
    ):

        algorithm = (
            self.clustering_algorithm_combo.currentText()
        )

        if algorithm == "KMeans":

            self.clustering_tuning_params = {
                "cluster_range": (
                    2,
                    10,
                ),
                "metric": "silhouette",
            }

        elif algorithm == "DBSCAN":

            self.clustering_tuning_params = {
                "metric": "silhouette",
            }

        elif algorithm == "Agglomerative":

            self.clustering_tuning_params = {
                "cluster_range": (
                    2,
                    10,
                ),
                "linkage_options": [
                    "ward",
                    "complete",
                    "average",
                    "single",
                ],
                "metric": "silhouette",
            }

        self.status_label.setText(
            "Clustering hyperparameter configuration saved."
        )

    # =============================================================
    # Clustering Start
    # =============================================================

    def _start_clustering(self):

        if self.dataframe is None:

            QMessageBox.warning(
                self,
                "Dataset Required",
                "Please select a dataset first.",
            )

            return

        algorithm = (
            self.clustering_algorithm_combo.currentText()
        )

        params = {}

        if algorithm == "KMeans":

            params = {
                "n_clusters": (
                    self.kmeans_clusters.value()
                )
            }

        elif algorithm == "DBSCAN":

            params = {
                "eps": self.dbscan_eps.value(),
                "min_samples": (
                    self.dbscan_min_samples.value()
                ),
            }

        elif algorithm == "Agglomerative":

            params = {
                "n_clusters": (
                    self.agg_clusters.value()
                ),
                "linkage": (
                    self.linkage_combo.currentText()
                ),
            }

        feature_selection = (
            self.clustering_feature_combo.currentText()
        )

        feature_selection_params = {}

        if feature_selection == "Variance Threshold":

            feature_selection_params = {
                "threshold": (
                    self.variance_threshold.value()
                )
            }

        elif feature_selection == "PCA":

            feature_selection_params = {
                "n_components": (
                    self.pca_components.value()
                )
            }

        elif feature_selection == "Feature Agglomeration":

            feature_selection_params = {
                "n_clusters": (
                    self.feature_agglomeration_clusters.value()
                )
            }

        tuning_enabled = (
            self.clustering_tuning_checkbox.isChecked()
        )

        if (
            tuning_enabled
            and not self.clustering_tuning_params
        ):

            QMessageBox.warning(
                self,
                "Hyperparameter Tuning",
                "Please configure the hyperparameter search first.",
            )

            return

        use_dim_reduction = (
            self.dim_reduction_checkbox.isChecked()
        )

        self.clustering_results_text.setText(
            "Training clustering model..."
        )

        self.progress_bar.setValue(
            0
        )

        self.status_label.setText(
            "Starting clustering..."
        )

        try:

            self.clustering_controller.start_clustering(
                dataframe=self.dataframe,
                algorithm=algorithm,
                params=params,
                feature_selection=feature_selection,
                feature_selection_params=(
                    feature_selection_params
                ),
                use_dim_reduction=(
                    use_dim_reduction
                ),
                hyperparameter_tuning=(
                    tuning_enabled
                ),
                tuning_params=(
                    self.clustering_tuning_params
                ),
            )

        except Exception as exc:

            QMessageBox.critical(
                self,
                "Clustering Error",
                str(exc),
            )

            self._set_running_state(
                False
            )

    # =============================================================
    # Start AutoML
    # =============================================================

    def _start_automl(self):

        problem_type = (
            self.problem_type_combo.currentText()
        )

        if problem_type == "Classification":

            self._start_classification()

        elif problem_type == "Clustering":

            self._start_clustering()

    # =============================================================
    # Cancel
    # =============================================================

    def _cancel(self):

        if self._is_any_controller_running():

            QMessageBox.information(
                self,
                "Cancellation",
                "Cancellation is not available yet.",
            )

    # =============================================================
    # Controller State
    # =============================================================

    def _is_controller_running(
        self,
        controller,
    ):

        thread = getattr(
            controller,
            "thread",
            None,
        )

        if thread is None:

            return False

        try:

            return thread.isRunning()

        except RuntimeError:

            return False

    # -------------------------------------------------------------

    def _is_any_controller_running(
        self,
    ):

        return (
            self._is_controller_running(
                self.classification_controller
            )
            or
            self._is_controller_running(
                self.clustering_controller
            )
        )

    # =============================================================
    # Running State
    # =============================================================

    def _set_running_state(
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

        self.problem_type_combo.setEnabled(
            not running
        )

        # Classification controls

        self.target_combo.setEnabled(
            not running
        )

        self.classification_feature_combo.setEnabled(
            not running
        )

        self.classification_tuning_checkbox.setEnabled(
            not running
        )

        self.classification_tuning_method.setEnabled(
            (
                not running
                and self.classification_tuning_checkbox.isChecked()
            )
        )

        self.classification_tuning_button.setEnabled(
            (
                not running
                and self.classification_tuning_checkbox.isChecked()
            )
        )

        self.scoring_combo.setEnabled(
            not running
        )

        for checkbox in self.model_checkboxes.values():

            checkbox.setEnabled(
                not running
            )

        # Clustering controls

        self.clustering_algorithm_combo.setEnabled(
            not running
        )

        self.clustering_feature_combo.setEnabled(
            not running
        )

        self.kmeans_clusters.setEnabled(
            not running
        )

        self.dbscan_eps.setEnabled(
            not running
        )

        self.dbscan_min_samples.setEnabled(
            not running
        )

        self.agg_clusters.setEnabled(
            not running
        )

        self.linkage_combo.setEnabled(
            not running
        )

        self.variance_threshold.setEnabled(
            not running
        )

        self.pca_components.setEnabled(
            not running
        )

        self.feature_agglomeration_clusters.setEnabled(
            not running
        )

        self.clustering_tuning_checkbox.setEnabled(
            not running
        )

        self.clustering_tuning_button.setEnabled(
            (
                not running
                and self.clustering_tuning_checkbox.isChecked()
            )
        )

        self.dim_reduction_checkbox.setEnabled(
            not running
        )

    # =============================================================
    # Classification Started
    # =============================================================

    def _classification_started(
        self,
    ):

        self._set_running_state(
            True
        )

        self.progress_bar.setValue(
            0
        )

        self.status_label.setText(
            "Classification started."
        )

    # =============================================================
    # Progress
    # =============================================================

    def _update_progress(
        self,
        value,
    ):

        self.progress_bar.setValue(
            value
        )

    # =============================================================
    # Status
    # =============================================================

    def _update_status(
        self,
        message,
    ):

        self.status_label.setText(
            message
        )

    # =============================================================
    # Classification Finished
    # =============================================================

    def _classification_finished(
        self,
        result,
    ):

        self.progress_bar.setValue(
            100
        )

        self.status_label.setText(
            "Classification completed successfully."
        )

        best_model = result.get(
            "best_model",
            result.get(
                "model",
                "-"
            ),
        )

        self.best_model_value.setText(
            str(best_model)
        )

        self._display_classification_results(
            result
        )

        self._set_running_state(
            False
        )

    # =============================================================
    # Classification Results
    # =============================================================

    def _display_classification_results(
        self,
        result,
    ):

        results = result.get(
            "results",
            []
        )

        self.results_table.setRowCount(
            len(results)
        )

        for row, item in enumerate(
            results
        ):

            values = [
                item.get(
                    "rank",
                    row + 1
                ),
                item.get(
                    "model",
                    "-"
                ),
                item.get(
                    "score",
                    0
                ),
                item.get(
                    "accuracy",
                    0
                ),
                item.get(
                    "precision",
                    0
                ),
                item.get(
                    "recall",
                    0
                ),
                item.get(
                    "f1",
                    0
                ),
                item.get(
                    "status",
                    "Completed"
                ),
            ]

            for column, value in enumerate(
                values
            ):

                if isinstance(
                    value,
                    float
                ):

                    text = f"{value:.4f}"

                else:

                    text = str(
                        value
                    )

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

    # =============================================================
    # Clustering Finished
    # =============================================================

    def _clustering_finished(
        self,
        result,
    ):

        self.progress_bar.setValue(
            100
        )

        self.status_label.setText(
            "Clustering completed successfully."
        )

        self._display_clustering_results(
            result
        )

        self._set_running_state(
            False
        )

    # =============================================================
    # Clustering Results
    # =============================================================

    def _display_clustering_results(
        self,
        result,
    ):

        lines = []

        lines.append(
            "CLUSTERING RESULTS"
        )

        lines.append(
            "=" * 50
        )

        lines.append("")

        model_info = result.get(
            "model_info",
            {}
        )

        algorithm = model_info.get(
            "algorithm",
            self.clustering_algorithm_combo.currentText()
        )

        lines.append(
            f"Algorithm: {algorithm}"
        )

        lines.append(
            "Feature Selection: "
            f"{result.get('feature_selection', '-')}"
        )

        tuning_enabled = result.get(
            "hyperparameter_tuning",
            False
        )

        lines.append(
            "Hyperparameter Tuning: "
            f"{'Enabled' if tuning_enabled else 'Disabled'}"
        )

        lines.append("")

        lines.append(
            "Model Parameters:"
        )

        for key, value in model_info.items():

            if key == "algorithm":

                continue

            lines.append(
                f"  {key}: {value}"
            )

        lines.append("")

        metrics = result.get(
            "metrics",
            {}
        )

        lines.append(
            "Clustering Metrics:"
        )

        for metric_name, value in metrics.items():

            if value is None:

                formatted_value = "N/A"

            elif isinstance(
                value,
                float
            ):

                formatted_value = (
                    f"{value:.6f}"
                )

            else:

                formatted_value = str(
                    value
                )

            lines.append(
                f"  {metric_name}: "
                f"{formatted_value}"
            )

        lines.append("")

        tuning_result = result.get(
            "tuning_result"
        )

        if tuning_result:

            lines.append(
                "Hyperparameter Tuning Result:"
            )

            best_params = tuning_result.get(
                "best_params",
                {}
            )

            best_score = tuning_result.get(
                "best_score"
            )

            for key, value in best_params.items():

                lines.append(
                    f"  Best {key}: {value}"
                )

            if best_score is not None:

                lines.append(
                    "  Best Score: "
                    f"{float(best_score):.6f}"
                )

            lines.append("")

        data = result.get(
            "data"
        )

        if data is not None:

            try:

                rows, columns = data.shape

                lines.append(
                    "Transformed Data Shape: "
                    f"({rows}, {columns})"
                )

            except AttributeError:

                pass

        labels = result.get(
            "labels"
        )

        if labels is not None:

            lines.append(
                f"Number of Samples: {len(labels)}"
            )

        self.clustering_results_text.setText(
            "<pre>"
            + "\n".join(lines)
            + "</pre>"
        )

    # =============================================================
    # Error
    # =============================================================

    def _operation_error(
        self,
        message,
    ):

        self.progress_bar.setValue(
            0
        )

        self.status_label.setText(
            "Training failed."
        )

        self._set_running_state(
            False
        )

        QMessageBox.critical(
            self,
            "AutoML Error",
            message,
        )
    
