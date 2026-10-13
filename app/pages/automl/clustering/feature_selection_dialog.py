from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QDoubleSpinBox,
    QSpinBox,
    QPushButton,
)


class ClusteringFeatureSelectionDialog(QDialog):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.configuration = None

        self.setWindowTitle(
            "Clustering - Feature Selection"
        )

        self.resize(500, 350)

        self._build_ui()

    # =========================================================
    # UI
    # =========================================================

    def _build_ui(self):

        layout = QVBoxLayout(self)

        title = QLabel(
            "Feature Selection"
        )

        title.setObjectName(
            "SectionTitle"
        )

        layout.addWidget(title)

        description = QLabel(
            "Select a feature selection method."
        )

        layout.addWidget(description)

        # -----------------------------------------------------
        # Method
        # -----------------------------------------------------

        method_layout = QHBoxLayout()

        method_label = QLabel(
            "Method:"
        )

        self.method_combo = QComboBox()

        self.method_combo.addItems(
            [
                "No Selection",
                "Variance Threshold",
                "PCA",
                "Feature Agglomeration",
            ]
        )

        self.method_combo.currentTextChanged.connect(
            self._update_parameters
        )

        method_layout.addWidget(
            method_label
        )

        method_layout.addWidget(
            self.method_combo
        )

        layout.addLayout(
            method_layout
        )

        # -----------------------------------------------------
        # Variance Threshold
        # -----------------------------------------------------

        self.threshold_label = QLabel(
            "Threshold:"
        )

        self.threshold_spinbox = QDoubleSpinBox()

        self.threshold_spinbox.setRange(
            0.0,
            1000000.0,
        )

        self.threshold_spinbox.setDecimals(
            6
        )

        self.threshold_spinbox.setValue(
            0.0
        )

        layout.addWidget(
            self.threshold_label
        )

        layout.addWidget(
            self.threshold_spinbox
        )

        # -----------------------------------------------------
        # PCA Components
        # -----------------------------------------------------

        self.pca_label = QLabel(
            "Number of Components:"
        )

        self.pca_spinbox = QSpinBox()

        self.pca_spinbox.setRange(
            1,
            100,
        )

        self.pca_spinbox.setValue(
            2
        )

        layout.addWidget(
            self.pca_label
        )

        layout.addWidget(
            self.pca_spinbox
        )

        # -----------------------------------------------------
        # Feature Agglomeration
        # -----------------------------------------------------

        self.agglomeration_label = QLabel(
            "Number of Clusters:"
        )

        self.agglomeration_spinbox = QSpinBox()

        self.agglomeration_spinbox.setRange(
            1,
            100,
        )

        self.agglomeration_spinbox.setValue(
            2
        )

        layout.addWidget(
            self.agglomeration_label
        )

        layout.addWidget(
            self.agglomeration_spinbox
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

        self._update_parameters(
            self.method_combo.currentText()
        )

    # =========================================================
    # Parameters
    # =========================================================

    def _update_parameters(
        self,
        method,
    ):

        self.threshold_label.setVisible(
            method == "Variance Threshold"
        )

        self.threshold_spinbox.setVisible(
            method == "Variance Threshold"
        )

        self.pca_label.setVisible(
            method == "PCA"
        )

        self.pca_spinbox.setVisible(
            method == "PCA"
        )

        self.agglomeration_label.setVisible(
            method == "Feature Agglomeration"
        )

        self.agglomeration_spinbox.setVisible(
            method == "Feature Agglomeration"
        )

    # =========================================================
    # Save
    # =========================================================

    def _save(self):

        method = (
            self.method_combo.currentText()
        )

        params = {}

        if method == "Variance Threshold":

            params["threshold"] = (
                self.threshold_spinbox.value()
            )

        elif method == "PCA":

            params["n_components"] = (
                self.pca_spinbox.value()
            )

        elif method == "Feature Agglomeration":

            params["n_clusters"] = (
                self.agglomeration_spinbox.value()
            )

        self.configuration = {
            "method": method,
            "params": params,
        }

        self.accept()

    # =========================================================
    # Result
    # =========================================================

    def get_configuration(self):

        return self.configuration