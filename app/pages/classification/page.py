from PySide6.QtCore import Qt
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
)


class ClassificationPage(QWidget):

    def __init__(self):
        super().__init__()

        self.setObjectName("ClassificationPage")

        self._build_ui()

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
        config_layout.setContentsMargins(25, 25, 25, 25)
        config_layout.setSpacing(18)

        # -------------------------------------------------
        # Dataset
        # -------------------------------------------------

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

        config_layout.addWidget(dataset_label)
        config_layout.addLayout(dataset_layout)

        # -------------------------------------------------
        # Target Column
        # -------------------------------------------------

        target_label = QLabel("Target Column")
        target_label.setObjectName("fieldLabel")

        self.target_combo = QComboBox()

        self.target_combo.setPlaceholderText(
            "Select target column..."
        )

        config_layout.addWidget(target_label)
        config_layout.addWidget(
            self.target_combo
        )

        # -------------------------------------------------
        # Model + Feature Selection
        # -------------------------------------------------

        grid = QGridLayout()
        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(10)

        model_label = QLabel("Model")
        model_label.setObjectName("fieldLabel")

        feature_label = QLabel(
            "Feature Selection"
        )
        feature_label.setObjectName("fieldLabel")

        self.model_combo = QComboBox()

        self.model_combo.addItems([
            "KNN",
            "SVM",
            "MLP",
            "Random Forest",
            "Naive Bayes",
            "Decision Tree",
        ])

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

        config_layout.addLayout(grid)

        # -------------------------------------------------
        # Model Parameters
        # -------------------------------------------------

        parameters_label = QLabel(
            "Model Parameters"
        )
        parameters_label.setObjectName(
            "sectionTitle"
        )

        self.parameters_input = QLineEdit()

        self.parameters_input.setPlaceholderText(
            "Optional model parameters..."
        )

        config_layout.addWidget(
            parameters_label
        )

        config_layout.addWidget(
            self.parameters_input
        )

        # -------------------------------------------------
        # Buttons
        # -------------------------------------------------

        buttons_layout = QHBoxLayout()

        buttons_layout.addStretch()

        self.cancel_button = QPushButton(
            "Cancel"
        )

        self.cancel_button.setEnabled(False)

        self.start_button = QPushButton(
            "Start Training"
        )

        self.start_button.setObjectName(
            "primaryButton"
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

        # -------------------------------------------------
        # Training Status
        # -------------------------------------------------

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

        self.progress_bar.setValue(0)

        status_layout.addWidget(
            self.status_label
        )

        status_layout.addWidget(
            self.progress_bar
        )

        main_layout.addWidget(
            status_card
        )

        main_layout.addStretch()

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

        self.dataset_input.setText(
            file_path
        )

        self.status_label.setText(
            "Status: Dataset selected"
        )