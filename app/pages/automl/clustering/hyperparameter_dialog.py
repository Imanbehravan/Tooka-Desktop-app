from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QPushButton,
    QMessageBox,
)


class ClusteringHyperparameterDialog(QDialog):

    def __init__(
        self,
        algorithm,
        parent=None,
    ):
        super().__init__(parent)

        self.algorithm = algorithm
        self.configuration = None

        self.setWindowTitle(
            f"{algorithm} - Hyperparameter Tuning"
        )

        self.resize(550, 450)

        self._build_ui()

        self._update_parameters()

    # =========================================================
    # UI
    # =========================================================

    def _build_ui(self):

        layout = QVBoxLayout(self)

        title = QLabel(
            f"{self.algorithm} Hyperparameter Tuning"
        )

        title.setObjectName(
            "SectionTitle"
        )

        layout.addWidget(title)

        description = QLabel(
            "Configure the search range and evaluation metric."
        )

        layout.addWidget(description)

        # -----------------------------------------------------
        # Metric
        # -----------------------------------------------------

        metric_layout = QHBoxLayout()

        metric_label = QLabel(
            "Metric:"
        )

        self.metric_combo = QComboBox()

        self.metric_combo.addItems(
            [
                "silhouette",
                "calinski",
                "davies",
            ]
        )

        metric_layout.addWidget(
            metric_label
        )

        metric_layout.addWidget(
            self.metric_combo
        )

        layout.addLayout(
            metric_layout
        )

        # -----------------------------------------------------
        # Cluster minimum
        # -----------------------------------------------------

        min_layout = QHBoxLayout()

        self.min_clusters_label = QLabel(
            "Minimum Clusters:"
        )

        self.min_clusters_spinbox = QSpinBox()

        self.min_clusters_spinbox.setRange(
            2,
            100,
        )

        self.min_clusters_spinbox.setValue(
            2
        )

        min_layout.addWidget(
            self.min_clusters_label
        )

        min_layout.addWidget(
            self.min_clusters_spinbox
        )

        layout.addLayout(
            min_layout
        )

        # -----------------------------------------------------
        # Cluster maximum
        # -----------------------------------------------------

        max_layout = QHBoxLayout()

        self.max_clusters_label = QLabel(
            "Maximum Clusters:"
        )

        self.max_clusters_spinbox = QSpinBox()

        self.max_clusters_spinbox.setRange(
            2,
            100,
        )

        self.max_clusters_spinbox.setValue(
            10
        )

        max_layout.addWidget(
            self.max_clusters_label
        )

        max_layout.addWidget(
            self.max_clusters_spinbox
        )

        layout.addLayout(
            max_layout
        )

        # -----------------------------------------------------
        # DBSCAN epsilon
        # -----------------------------------------------------

        self.eps_label = QLabel(
            "Epsilon:"
        )

        self.eps_spinbox = QDoubleSpinBox()

        self.eps_spinbox.setRange(
            0.01,
            100.0,
        )

        self.eps_spinbox.setDecimals(
            3
        )

        self.eps_spinbox.setValue(
            0.5
        )

        layout.addWidget(
            self.eps_label
        )

        layout.addWidget(
            self.eps_spinbox
        )

        # -----------------------------------------------------
        # DBSCAN epsilon step
        # -----------------------------------------------------

        self.eps_step_label = QLabel(
            "Epsilon Step:"
        )

        self.eps_step_spinbox = QDoubleSpinBox()

        self.eps_step_spinbox.setRange(
            0.01,
            10.0,
        )

        self.eps_step_spinbox.setDecimals(
            3
        )

        self.eps_step_spinbox.setValue(
            0.1
        )

        layout.addWidget(
            self.eps_step_label
        )

        layout.addWidget(
            self.eps_step_spinbox
        )

        # -----------------------------------------------------
        # Min Samples
        # -----------------------------------------------------

        self.min_samples_label = QLabel(
            "Minimum Samples:"
        )

        self.min_samples_spinbox = QSpinBox()

        self.min_samples_spinbox.setRange(
            1,
            100,
        )

        self.min_samples_spinbox.setValue(
            3
        )

        layout.addWidget(
            self.min_samples_label
        )

        layout.addWidget(
            self.min_samples_spinbox
        )

        # -----------------------------------------------------
        # Max Samples
        # -----------------------------------------------------

        self.max_samples_label = QLabel(
            "Maximum Samples:"
        )

        self.max_samples_spinbox = QSpinBox()

        self.max_samples_spinbox.setRange(
            1,
            100,
        )

        self.max_samples_spinbox.setValue(
            6
        )

        layout.addWidget(
            self.max_samples_label
        )

        layout.addWidget(
            self.max_samples_spinbox
        )

        # -----------------------------------------------------
        # Linkage
        # -----------------------------------------------------

        self.linkage_label = QLabel(
            "Linkage:"
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

        layout.addWidget(
            self.linkage_label
        )

        layout.addWidget(
            self.linkage_combo
        )

        # -----------------------------------------------------
        # Buttons
        # -----------------------------------------------------

        button_layout = QHBoxLayout()

        button_layout.addStretch()

        cancel_button = QPushButton(
            "Cancel"
        )

        save_button = QPushButton(
            "Save"
        )

        save_button.setObjectName(
            "PrimaryButton"
        )

        cancel_button.clicked.connect(
            self.reject
        )

        save_button.clicked.connect(
            self._save
        )

        button_layout.addWidget(
            cancel_button
        )

        button_layout.addWidget(
            save_button
        )

        layout.addLayout(
            button_layout
        )

    # =========================================================
    # Visibility
    # =========================================================

    def _update_parameters(self):

        is_kmeans = (
            self.algorithm == "KMeans"
        )

        is_dbscan = (
            self.algorithm == "DBSCAN"
        )

        is_agglomerative = (
            self.algorithm == "Agglomerative"
        )

        cluster_visible = (
            is_kmeans
            or is_agglomerative
        )

        self.min_clusters_label.setVisible(
            cluster_visible
        )

        self.min_clusters_spinbox.setVisible(
            cluster_visible
        )

        self.max_clusters_label.setVisible(
            cluster_visible
        )

        self.max_clusters_spinbox.setVisible(
            cluster_visible
        )

        self.eps_label.setVisible(
            is_dbscan
        )

        self.eps_spinbox.setVisible(
            is_dbscan
        )

        self.eps_step_label.setVisible(
            is_dbscan
        )

        self.eps_step_spinbox.setVisible(
            is_dbscan
        )

        self.min_samples_label.setVisible(
            is_dbscan
        )

        self.min_samples_spinbox.setVisible(
            is_dbscan
        )

        self.max_samples_label.setVisible(
            is_dbscan
        )

        self.max_samples_spinbox.setVisible(
            is_dbscan
        )

        self.linkage_label.setVisible(
            is_agglomerative
        )

        self.linkage_combo.setVisible(
            is_agglomerative
        )

    # =========================================================
    # Save
    # =========================================================

    def _save(self):

        metric = (
            self.metric_combo.currentText()
        )

        if self.algorithm in (
            "KMeans",
            "Agglomerative",
        ):

            minimum = (
                self.min_clusters_spinbox.value()
            )

            maximum = (
                self.max_clusters_spinbox.value()
            )

            if minimum > maximum:

                QMessageBox.warning(
                    self,
                    "Invalid Configuration",
                    "Minimum clusters cannot be greater than maximum clusters.",
                )

                return

        if self.algorithm == "KMeans":

            self.configuration = {
                "metric": metric,
                "cluster_range": (
                    minimum,
                    maximum,
                ),
            }

        elif self.algorithm == "DBSCAN":

            eps_start = (
                self.eps_spinbox.value()
            )

            eps_step = (
                self.eps_step_spinbox.value()
            )

            min_samples_start = (
                self.min_samples_spinbox.value()
            )

            min_samples_end = (
                self.max_samples_spinbox.value()
            )

            if (
                min_samples_start
                > min_samples_end
            ):

                QMessageBox.warning(
                    self,
                    "Invalid Configuration",
                    "Minimum samples cannot be greater than maximum samples.",
                )

                return

            import numpy as np

            eps_values = np.arange(
                eps_start,
                eps_start + eps_step * 20,
                eps_step,
            )

            min_samples_values = list(
                range(
                    min_samples_start,
                    min_samples_end + 1,
                )
            )

            self.configuration = {
                "metric": metric,
                "eps_values": eps_values,
                "min_samples_values": (
                    min_samples_values
                ),
            }

        elif self.algorithm == "Agglomerative":

            self.configuration = {
                "metric": metric,
                "cluster_range": (
                    minimum,
                    maximum,
                ),
                "linkage_options": [
                    "ward",
                    "complete",
                    "average",
                    "single",
                ],
            }

        self.accept()

    # =========================================================
    # Result
    # =========================================================

    def get_configuration(self):

        return self.configuration