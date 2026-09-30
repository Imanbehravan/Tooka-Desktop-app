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
    QSpinBox,
    QDoubleSpinBox,
)

from app.workers.clustering_controller import ClusteringController


class ClusteringPage(QWidget):

    def __init__(self):
        super().__init__()

        self.setObjectName("ClusteringPage")

        # =================================================
        # State
        # =================================================

        self.dataframe = None
        self.dataset_path = None

        # =================================================
        # Controller
        # =================================================

        self.controller = ClusteringController(self)

        # =================================================
        # UI
        # =================================================

        self._build_ui()
        self._connect_controller()

        # Set initial algorithm UI
        self._algorithm_changed(
            self.algorithm_combo.currentText()
        )

    # =====================================================
    # UI
    # =====================================================

    def _build_ui(self):

        main_layout = QVBoxLayout(self)

        main_layout.setContentsMargins(
            30,
            25,
            30,
            25,
        )

        main_layout.setSpacing(20)

        # -------------------------------------------------
        # Header
        # -------------------------------------------------

        title = QLabel("Clustering")
        title.setObjectName("pageTitle")

        subtitle = QLabel(
            "Discover hidden groups and patterns in your data"
        )

        subtitle.setObjectName("pageSubtitle")

        main_layout.addWidget(title)
        main_layout.addWidget(subtitle)

        # =================================================
        # Configuration Card
        # =================================================

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
        # Algorithm + Feature Selection
        # =================================================

        grid = QGridLayout()

        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(10)

        # -------------------------------------------------
        # Algorithm
        # -------------------------------------------------

        algorithm_label = QLabel("Algorithm")
        algorithm_label.setObjectName("fieldLabel")

        self.algorithm_combo = QComboBox()

        self.algorithm_combo.addItems([
            "KMeans",
            "DBSCAN",
            "Agglomerative",
        ])

        self.algorithm_combo.currentTextChanged.connect(
            self._algorithm_changed
        )

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
            "Variance Threshold",
            "PCA",
            "Feature Agglomeration",
        ])

        self.feature_combo.currentTextChanged.connect(
            self._feature_selection_changed
        )

        grid.addWidget(
            algorithm_label,
            0,
            0,
        )

        grid.addWidget(
            feature_label,
            0,
            1,
        )

        grid.addWidget(
            self.algorithm_combo,
            1,
            0,
        )

        grid.addWidget(
            self.feature_combo,
            1,
            1,
        )

        config_layout.addLayout(grid)

        # =================================================
        # Algorithm Parameters
        # =================================================

        parameters_title = QLabel(
            "Algorithm Parameters"
        )

        parameters_title.setObjectName(
            "sectionTitle"
        )

        config_layout.addWidget(
            parameters_title
        )

        # -------------------------------------------------
        # KMeans Parameters
        # -------------------------------------------------

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

        kmeans_label = QLabel(
            "Number of Clusters"
        )

        self.kmeans_clusters = QSpinBox()

        self.kmeans_clusters.setMinimum(2)
        self.kmeans_clusters.setMaximum(100)
        self.kmeans_clusters.setValue(3)

        kmeans_layout.addWidget(
            kmeans_label
        )

        kmeans_layout.addWidget(
            self.kmeans_clusters
        )

        kmeans_layout.addStretch()

        config_layout.addWidget(
            self.kmeans_frame
        )

        # -------------------------------------------------
        # DBSCAN Parameters
        # -------------------------------------------------

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

        eps_label = QLabel("Epsilon")

        self.dbscan_eps = QDoubleSpinBox()

        self.dbscan_eps.setMinimum(0.01)
        self.dbscan_eps.setMaximum(1000.0)
        self.dbscan_eps.setSingleStep(0.1)
        self.dbscan_eps.setValue(0.5)

        min_samples_label = QLabel(
            "Min Samples"
        )

        self.dbscan_min_samples = QSpinBox()

        self.dbscan_min_samples.setMinimum(1)
        self.dbscan_min_samples.setMaximum(1000)
        self.dbscan_min_samples.setValue(5)

        dbscan_layout.addWidget(
            eps_label,
            0,
            0,
        )

        dbscan_layout.addWidget(
            self.dbscan_eps,
            0,
            1,
        )

        dbscan_layout.addWidget(
            min_samples_label,
            1,
            0,
        )

        dbscan_layout.addWidget(
            self.dbscan_min_samples,
            1,
            1,
        )

        config_layout.addWidget(
            self.dbscan_frame
        )

        # -------------------------------------------------
        # Agglomerative Parameters
        # -------------------------------------------------

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

        agg_clusters_label = QLabel(
            "Number of Clusters"
        )

        self.agg_clusters = QSpinBox()

        self.agg_clusters.setMinimum(2)
        self.agg_clusters.setMaximum(100)
        self.agg_clusters.setValue(3)

        linkage_label = QLabel(
            "Linkage"
        )

        self.linkage_combo = QComboBox()

        self.linkage_combo.addItems([
            "ward",
            "complete",
            "average",
            "single",
        ])

        agglomerative_layout.addWidget(
            agg_clusters_label,
            0,
            0,
        )

        agglomerative_layout.addWidget(
            self.agg_clusters,
            0,
            1,
        )

        agglomerative_layout.addWidget(
            linkage_label,
            1,
            0,
        )

        agglomerative_layout.addWidget(
            self.linkage_combo,
            1,
            1,
        )

        config_layout.addWidget(
            self.agglomerative_frame
        )

        # =================================================
        # Feature Selection Parameters
        # =================================================

        self.feature_parameters_title = QLabel(
            "Feature Selection Parameters"
        )

        self.feature_parameters_title.setObjectName(
            "sectionTitle"
        )

        config_layout.addWidget(
            self.feature_parameters_title
        )

        # -------------------------------------------------
        # Variance Threshold
        # -------------------------------------------------

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

        variance_label = QLabel(
            "Threshold"
        )

        self.variance_threshold = QDoubleSpinBox()

        self.variance_threshold.setMinimum(
            0.0
        )

        self.variance_threshold.setMaximum(
            100000.0
        )

        self.variance_threshold.setDecimals(
            6
        )

        self.variance_threshold.setSingleStep(
            0.01
        )

        self.variance_threshold.setValue(
            0.0
        )

        variance_layout.addWidget(
            variance_label
        )

        variance_layout.addWidget(
            self.variance_threshold
        )

        variance_layout.addStretch()

        config_layout.addWidget(
            self.variance_frame
        )

        # -------------------------------------------------
        # PCA
        # -------------------------------------------------

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

        pca_label = QLabel(
            "Number of Components"
        )

        self.pca_components = QSpinBox()

        self.pca_components.setMinimum(1)
        self.pca_components.setMaximum(100)
        self.pca_components.setValue(2)

        pca_layout.addWidget(
            pca_label
        )

        pca_layout.addWidget(
            self.pca_components
        )

        pca_layout.addStretch()

        config_layout.addWidget(
            self.pca_frame
        )

        # -------------------------------------------------
        # Feature Agglomeration
        # -------------------------------------------------

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

        fa_label = QLabel(
            "Number of Clusters"
        )

        self.feature_agglomeration_clusters = QSpinBox()

        self.feature_agglomeration_clusters.setMinimum(
            2
        )

        self.feature_agglomeration_clusters.setMaximum(
            100
        )

        self.feature_agglomeration_clusters.setValue(
            2
        )

        fa_layout.addWidget(
            fa_label
        )

        fa_layout.addWidget(
            self.feature_agglomeration_clusters
        )

        fa_layout.addStretch()

        config_layout.addWidget(
            self.feature_agglomeration_frame
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

        self.tuning_checkbox = QCheckBox(
            "Enable Hyperparameter Tuning"
        )

        self.tuning_checkbox.toggled.connect(
            self._toggle_tuning
        )

        self.tuning_config_button = QPushButton(
            "Configure"
        )

        self.tuning_config_button.setEnabled(
            False
        )

        self.tuning_config_button.clicked.connect(
            self._configure_tuning
        )

        tuning_layout.addWidget(
            self.tuning_checkbox
        )

        tuning_layout.addWidget(
            self.tuning_config_button
        )

        tuning_layout.addStretch()

        config_layout.addLayout(
            tuning_layout
        )

        # =================================================
        # Dimensionality Reduction
        # =================================================

        self.dim_reduction_checkbox = QCheckBox(
            "Apply PCA Dimensionality Reduction"
        )

        config_layout.addWidget(
            self.dim_reduction_checkbox
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
            "Start Clustering"
        )

        self.start_button.setObjectName(
            "primaryButton"
        )

        self.start_button.clicked.connect(
            self._start_clustering
        )

        self.cancel_button.clicked.connect(
            self._cancel_clustering
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
        # Status
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
            "Clustering results will appear here..."
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
            self._clustering_finished
        )

        self.controller.error.connect(
            self._clustering_error
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

        self.dataframe = dataframe
        self.dataset_path = file_path

        self.dataset_input.setText(
            file_path
        )

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
    # Algorithm
    # =====================================================

    def _algorithm_changed(
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

        # Changing algorithm invalidates tuning
        self.tuning_params = None

    # =====================================================
    # Feature Selection
    # =====================================================

    def _feature_selection_changed(self, method):

        self.variance_frame.setVisible(
            method == "Variance Threshold"
        )

        self.pca_frame.setVisible(
            method == "PCA"
        )

        self.feature_agglomeration_frame.setVisible(
            method == "Feature Agglomeration"
        )

        # PCA feature selection already performs
        # dimensionality reduction.
        is_pca = method == "PCA"

        if is_pca:

            self.dim_reduction_checkbox.setChecked(
                False
            )

        self.dim_reduction_checkbox.setEnabled(
            not is_pca
        )

    # =====================================================
    # Hyperparameter Tuning
    # =====================================================

    def _toggle_tuning(
        self,
        enabled,
    ):

        self.tuning_config_button.setEnabled(
            enabled
        )

        if not enabled:

            self.tuning_params = None

            self.status_label.setText(
                "Status: Hyperparameter tuning disabled."
            )

    # -----------------------------------------------------

    def _configure_tuning(self):

        algorithm = (
            self.algorithm_combo.currentText()
        )

        # For now we use sensible defaults.
        # A dedicated dialog can be added after
        # the basic Clustering Page is working.

        if algorithm == "KMeans":

            self.tuning_params = {
                "cluster_range": (2, 10),
                "metric": "silhouette",
            }

        elif algorithm == "DBSCAN":

            self.tuning_params = {
                "metric": "silhouette",
            }

        elif algorithm == "Agglomerative":

            self.tuning_params = {
                "cluster_range": (2, 10),
                "linkage_options": [
                    "ward",
                    "complete",
                    "average",
                    "single",
                ],
                "metric": "silhouette",
            }

        self.status_label.setText(
            "Status: Hyperparameter configuration saved."
        )

    # =====================================================
    # Start Clustering
    # =====================================================

    def _start_clustering(self):

        # -------------------------------------------------
        # Dataset
        # -------------------------------------------------

        if self.dataframe is None:

            QMessageBox.warning(
                self,
                "Dataset Required",
                "Please select a CSV dataset first.",
            )

            return

        # -------------------------------------------------
        # Algorithm
        # -------------------------------------------------

        algorithm = (
            self.algorithm_combo.currentText()
        )

        # -------------------------------------------------
        # Algorithm Parameters
        # -------------------------------------------------

        params = {}

        if algorithm == "KMeans":

            params = {
                "n_clusters": (
                    self.kmeans_clusters.value()
                ),
            }

        elif algorithm == "DBSCAN":

            params = {
                "eps": (
                    self.dbscan_eps.value()
                ),
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

        # -------------------------------------------------
        # Feature Selection
        # -------------------------------------------------

        feature_selection = (
            self.feature_combo.currentText()
        )

        feature_selection_params = {}

        if feature_selection == "Variance Threshold":

            feature_selection_params = {
                "threshold": (
                    self.variance_threshold.value()
                ),
            }

        elif feature_selection == "PCA":

            feature_selection_params = {
                "n_components": (
                    self.pca_components.value()
                ),
            }

        elif feature_selection == "Feature Agglomeration":

            feature_selection_params = {
                "n_clusters": (
                    self.feature_agglomeration_clusters.value()
                ),
            }

        # -------------------------------------------------
        # Hyperparameter Tuning
        # -------------------------------------------------

        hyperparameter_tuning = (
            self.tuning_checkbox.isChecked()
        )

        if (
            hyperparameter_tuning
            and not self.tuning_params
        ):

            QMessageBox.warning(
                self,
                "Hyperparameter Tuning",
                "Please configure the hyperparameter "
                "search space first.",
            )

            return

        # -------------------------------------------------
        # Dimensionality Reduction
        # -------------------------------------------------

        use_dim_reduction = (
            self.dim_reduction_checkbox.isChecked()
        )

        # -------------------------------------------------
        # UI State
        # -------------------------------------------------

        self._set_clustering_state(
            running=True
        )

        self.progress_bar.setValue(
            0
        )

        self.status_label.setText(
            "Status: Starting clustering..."
        )

        self.results_text.clear()

        # -------------------------------------------------
        # Controller
        # -------------------------------------------------

        try:

            self.controller.start_clustering(
                dataframe=self.dataframe,
                algorithm=algorithm,
                params=params,
                feature_selection=feature_selection,
                feature_selection_params=feature_selection_params,
                use_dim_reduction=use_dim_reduction,
                hyperparameter_tuning=hyperparameter_tuning,
                tuning_params=self.tuning_params,
            )

        except Exception as exc:

            self._set_clustering_state(
                running=False
            )

            self.status_label.setText(
                "Status: Clustering failed."
            )

            QMessageBox.critical(
                self,
                "Clustering Error",
                str(exc),
            )

    # =====================================================
    # Cancel
    # =====================================================
    def _cancel_clustering(self):

        if not self.controller.is_running():
            return

        self.controller.cancel()

        self.cancel_button.setEnabled(False)

        self.status_label.setText(
            "Status: Cancelling..."
        )

    # =====================================================
    # UI State
    # =====================================================

    def _set_clustering_state(
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

        self.algorithm_combo.setEnabled(
            not running
        )

        self.feature_combo.setEnabled(
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

        self.tuning_checkbox.setEnabled(
            not running
        )

        self.dim_reduction_checkbox.setEnabled(
            not running
        )

        if running:

            self.tuning_config_button.setEnabled(
                False
            )

        else:

            self.tuning_config_button.setEnabled(
                self.tuning_checkbox.isChecked()
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
    # Finished
    # =====================================================

    def _clustering_finished(
        self,
        result,
    ):

        self._set_clustering_state(
            running=False
        )

        self.progress_bar.setValue(
            100
        )

        self.status_label.setText(
            "Status: Clustering completed."
        )

        self._display_results(
            result
        )

    # =====================================================
    # Error
    # =====================================================

    def _clustering_error(
        self,
        message,
    ):

        self._set_clustering_state(
            running=False
        )

        self.status_label.setText(
            "Status: Clustering failed."
        )

        self.results_text.setPlainText(
            f"Clustering Error:\n\n{message}"
        )

        QMessageBox.critical(
            self,
            "Clustering Error",
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
            "CLUSTERING RESULTS"
        )

        lines.append(
            "=" * 50
        )

        lines.append("")

        # -------------------------------------------------
        # Algorithm
        # -------------------------------------------------

        model_info = result.get(
            "model_info",
            {},
        )

        algorithm = model_info.get(
            "algorithm",
            self.algorithm_combo.currentText(),
        )

        lines.append(
            f"Algorithm: {algorithm}"
        )

        lines.append(
            f"Feature Selection: "
            f"{result.get('feature_selection', '-')}"
        )

        lines.append(
            "Hyperparameter Tuning: "
            f"{'Enabled' if result.get('hyperparameter_tuning', False) else 'Disabled'}"
        )

        lines.append("")

        # -------------------------------------------------
        # Model Parameters
        # -------------------------------------------------

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

        # -------------------------------------------------
        # Metrics
        # -------------------------------------------------

        lines.append(
            "Clustering Metrics:"
        )

        metrics = result.get(
            "metrics",
            {},
        )

        for metric_name, value in metrics.items():

            if value is None:

                formatted_value = "N/A"

            elif isinstance(
                value,
                float,
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

        # -------------------------------------------------
        # Tuning Result
        # -------------------------------------------------

        tuning_result = result.get(
            "tuning_result"
        )

        if tuning_result:

            lines.append(
                "Hyperparameter Tuning Result:"
            )

            best_params = tuning_result.get(
                "best_params",
                {},
            )

            best_score = tuning_result.get(
                "best_score"
            )

            if best_params:

                for key, value in (
                    best_params.items()
                ):

                    lines.append(
                        f"  Best {key}: {value}"
                    )

            if best_score is not None:

                lines.append(
                    f"  Best Score: "
                    f"{float(best_score):.6f}"
                )

            lines.append("")

        # -------------------------------------------------
        # Data
        # -------------------------------------------------

        data = result.get(
            "data"
        )

        if data is not None:

            try:

                rows, columns = data.shape

                lines.append(
                    f"Transformed Data Shape: "
                    f"({rows}, {columns})"
                )

            except AttributeError:

                pass

        # -------------------------------------------------
        # Labels
        # -------------------------------------------------

        labels = result.get(
            "labels"
        )

        if labels is not None:

            lines.append(
                f"Number of Samples: "
                f"{len(labels)}"
            )

        lines.append("")

        # -------------------------------------------------
        # Display
        # -------------------------------------------------

        self.results_text.setPlainText(
            "\n".join(lines)
        )