from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
    QGroupBox,
)


class ClusteringTuningDialog(QDialog):

    def __init__(
        self,
        algorithm,
        parent=None,
        initial_params=None,
    ):
        super().__init__(parent)

        self.algorithm = algorithm
        self.initial_params = initial_params or {}

        self.setWindowTitle(
            f"{algorithm} Hyperparameter Tuning"
        )

        self.setMinimumWidth(460)

        self._build_ui()
        self._load_initial_values()

    # =========================================================
    # UI
    # =========================================================

    def _build_ui(self):

        layout = QVBoxLayout(self)

        layout.setContentsMargins(
            24,
            20,
            24,
            20,
        )

        layout.setSpacing(16)

        # -----------------------------------------------------
        # Header
        # -----------------------------------------------------

        title = QLabel(
            f"{self.algorithm} Hyperparameter Tuning"
        )

        title.setObjectName(
            "SectionTitle"
        )

        layout.addWidget(title)

        subtitle = QLabel(
            "Configure the hyperparameter search space."
        )

        subtitle.setObjectName(
            "SectionSubtitle"
        )

        subtitle.setWordWrap(True)

        layout.addWidget(subtitle)

        # -----------------------------------------------------
        # Search Configuration
        # -----------------------------------------------------

        search_group = QGroupBox(
            "Search Configuration"
        )

        search_layout = QFormLayout(
            search_group
        )

        search_layout.setSpacing(12)

        self.metric_combo = QComboBox()

        self.metric_combo.addItems(
            [
                "silhouette",
                "calinski_harabasz",
                "davies_bouldin",
            ]
        )

        search_layout.addRow(
            "Evaluation Metric",
            self.metric_combo,
        )

        layout.addWidget(
            search_group
        )

        # -----------------------------------------------------
        # KMeans
        # -----------------------------------------------------

        self.kmeans_group = QGroupBox(
            "KMeans Parameters"
        )

        kmeans_layout = QFormLayout(
            self.kmeans_group
        )

        self.kmeans_min_clusters = QSpinBox()

        self.kmeans_min_clusters.setRange(
            2,
            100,
        )

        self.kmeans_max_clusters = QSpinBox()

        self.kmeans_max_clusters.setRange(
            2,
            100,
        )

        kmeans_layout.addRow(
            "Minimum Clusters",
            self.kmeans_min_clusters,
        )

        kmeans_layout.addRow(
            "Maximum Clusters",
            self.kmeans_max_clusters,
        )

        layout.addWidget(
            self.kmeans_group
        )

        # -----------------------------------------------------
        # DBSCAN
        # -----------------------------------------------------

        self.dbscan_group = QGroupBox(
            "DBSCAN Parameters"
        )

        dbscan_layout = QFormLayout(
            self.dbscan_group
        )

        self.dbscan_eps_min = QDoubleSpinBox()

        self.dbscan_eps_min.setRange(
            0.01,
            1000.0,
        )

        self.dbscan_eps_min.setDecimals(
            4
        )

        self.dbscan_eps_min.setSingleStep(
            0.1
        )

        self.dbscan_eps_max = QDoubleSpinBox()

        self.dbscan_eps_max.setRange(
            0.01,
            1000.0,
        )

        self.dbscan_eps_max.setDecimals(
            4
        )

        self.dbscan_eps_max.setSingleStep(
            0.1
        )

        self.dbscan_min_samples_min = QSpinBox()

        self.dbscan_min_samples_min.setRange(
            1,
            1000,
        )

        self.dbscan_min_samples_max = QSpinBox()

        self.dbscan_min_samples_max.setRange(
            1,
            1000,
        )

        dbscan_layout.addRow(
            "Minimum Epsilon",
            self.dbscan_eps_min,
        )

        dbscan_layout.addRow(
            "Maximum Epsilon",
            self.dbscan_eps_max,
        )

        dbscan_layout.addRow(
            "Minimum Samples",
            self.dbscan_min_samples_min,
        )

        dbscan_layout.addRow(
            "Maximum Samples",
            self.dbscan_min_samples_max,
        )

        layout.addWidget(
            self.dbscan_group
        )

        # -----------------------------------------------------
        # Agglomerative
        # -----------------------------------------------------

        self.agglomerative_group = QGroupBox(
            "Agglomerative Parameters"
        )

        agglomerative_layout = QFormLayout(
            self.agglomerative_group
        )

        self.agg_min_clusters = QSpinBox()

        self.agg_min_clusters.setRange(
            2,
            100,
        )

        self.agg_max_clusters = QSpinBox()

        self.agg_max_clusters.setRange(
            2,
            100,
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

        agglomerative_layout.addRow(
            "Minimum Clusters",
            self.agg_min_clusters,
        )

        agglomerative_layout.addRow(
            "Maximum Clusters",
            self.agg_max_clusters,
        )

        agglomerative_layout.addRow(
            "Linkage",
            self.linkage_combo,
        )

        layout.addWidget(
            self.agglomerative_group
        )

        # -----------------------------------------------------
        # Buttons
        # -----------------------------------------------------

        buttons_layout = QHBoxLayout()

        buttons_layout.addStretch()

        self.cancel_button = QPushButton(
            "Cancel"
        )

        self.cancel_button.setObjectName(
            "SecondaryButton"
        )

        self.save_button = QPushButton(
            "Save Configuration"
        )

        self.save_button.setObjectName(
            "PrimaryButton"
        )

        buttons_layout.addWidget(
            self.cancel_button
        )

        buttons_layout.addWidget(
            self.save_button
        )

        layout.addLayout(
            buttons_layout
        )

        self.cancel_button.clicked.connect(
            self.reject
        )

        self.save_button.clicked.connect(
            self._save
        )

        self._update_algorithm_visibility()

    # =========================================================
    # INITIAL VALUES
    # =========================================================

    def _load_initial_values(self):

        self.kmeans_min_clusters.setValue(
            self.initial_params.get(
                "min_clusters",
                2,
            )
        )

        self.kmeans_max_clusters.setValue(
            self.initial_params.get(
                "max_clusters",
                10,
            )
        )

        self.dbscan_eps_min.setValue(
            self.initial_params.get(
                "eps_min",
                0.1,
            )
        )

        self.dbscan_eps_max.setValue(
            self.initial_params.get(
                "eps_max",
                2.0,
            )
        )

        self.dbscan_min_samples_min.setValue(
            self.initial_params.get(
                "min_samples_min",
                2,
            )
        )

        self.dbscan_min_samples_max.setValue(
            self.initial_params.get(
                "min_samples_max",
                10,
            )
        )

        self.agg_min_clusters.setValue(
            self.initial_params.get(
                "min_clusters",
                2,
            )
        )

        self.agg_max_clusters.setValue(
            self.initial_params.get(
                "max_clusters",
                10,
            )
        )

        metric = self.initial_params.get(
            "metric",
            "silhouette",
        )

        index = self.metric_combo.findText(
            metric
        )

        if index >= 0:
            self.metric_combo.setCurrentIndex(
                index
            )

        linkage = self.initial_params.get(
            "linkage",
            "ward",
        )

        index = self.linkage_combo.findText(
            linkage
        )

        if index >= 0:
            self.linkage_combo.setCurrentIndex(
                index
            )

    # =========================================================
    # VISIBILITY
    # =========================================================

    def _update_algorithm_visibility(self):

        self.kmeans_group.setVisible(
            self.algorithm == "KMeans"
        )

        self.dbscan_group.setVisible(
            self.algorithm == "DBSCAN"
        )

        self.agglomerative_group.setVisible(
            self.algorithm == "Agglomerative"
        )

    # =========================================================
    # SAVE
    # =========================================================

    def _save(self):

        if self.algorithm == "KMeans":

            min_clusters = (
                self.kmeans_min_clusters.value()
            )

            max_clusters = (
                self.kmeans_max_clusters.value()
            )

            if min_clusters > max_clusters:

                self.kmeans_max_clusters.setValue(
                    min_clusters
                )

                return

            self.params = {
                "cluster_range": (
                    min_clusters,
                    max_clusters,
                ),
                "metric": (
                    self.metric_combo.currentText()
                ),
            }

        elif self.algorithm == "DBSCAN":

            eps_min = (
                self.dbscan_eps_min.value()
            )

            eps_max = (
                self.dbscan_eps_max.value()
            )

            min_samples_min = (
                self.dbscan_min_samples_min.value()
            )

            min_samples_max = (
                self.dbscan_min_samples_max.value()
            )

            if eps_min > eps_max:

                self.dbscan_eps_max.setValue(
                    eps_min
                )

                return

            if (
                min_samples_min
                > min_samples_max
            ):

                self.dbscan_min_samples_max.setValue(
                    min_samples_min
                )

                return

            self.params = {
                "eps_values": [
                    eps_min,
                    eps_max,
                ],
                "min_samples_values": [
                    min_samples_min,
                    min_samples_max,
                ],
                "metric": (
                    self.metric_combo.currentText()
                ),
            }

        elif self.algorithm == "Agglomerative":

            min_clusters = (
                self.agg_min_clusters.value()
            )

            max_clusters = (
                self.agg_max_clusters.value()
            )

            if min_clusters > max_clusters:

                self.agg_max_clusters.setValue(
                    min_clusters
                )

                return

            self.params = {
                "cluster_range": (
                    min_clusters,
                    max_clusters,
                ),
                "linkage_options": [
                    self.linkage_combo.currentText()
                ],
                "metric": (
                    self.metric_combo.currentText()
                ),
            }

        else:

            self.params = {}

        self.accept()

    # =========================================================
    # RESULT
    # =========================================================

    def get_params(self):

        return getattr(
            self,
            "params",
            None,
        )