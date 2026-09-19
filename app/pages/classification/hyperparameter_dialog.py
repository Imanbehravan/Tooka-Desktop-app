import ast

from .hyperparameter_dialog import HyperparameterConfigDialog

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QSpinBox,
    QComboBox,
    QPushButton,
    QMessageBox,
    QDialogButtonBox,
    QHeaderView,
)


class HyperparameterConfigDialog(QDialog):
    """
    Dialog for configuring hyperparameter search space.
    """

    DEFAULT_SPACES = {
        "KNN": {
            "n_neighbors": "3, 5, 7",
            "weights": "'uniform', 'distance'",
            "p": "1, 2",
        },

        "SVM": {
            "C": "0.1, 1, 10",
            "kernel": "'linear', 'rbf'",
            "gamma": "'scale', 'auto'",
        },

        "MLP": {
            "hidden_layer_sizes": "(50,), (100,), (50, 50)",
            "activation": "'relu', 'tanh'",
            "alpha": "0.0001, 0.001",
            "learning_rate": "'constant', 'adaptive'",
        },

        "Random Forest": {
            "n_estimators": "50, 100, 200",
            "max_depth": "None, 5, 10",
            "min_samples_split": "2, 5, 10",
        },

        "Naive Bayes": {
            "var_smoothing": "1e-9, 1e-8, 1e-7",
        },

        "Decision Tree": {
            "criterion": "'gini', 'entropy'",
            "max_depth": "None, 5, 10",
            "min_samples_split": "2, 5, 10",
        },
    }

    def __init__(
        self,
        model_name,
        tuning_method="Grid Search",
        parent=None,
    ):
        super().__init__(parent)

        self.model_name = model_name
        self.tuning_method = tuning_method

        self.setWindowTitle(
            f"{model_name} - Hyperparameter Tuning"
        )

        self.setMinimumWidth(700)
        self.setMinimumHeight(500)

        self._build_ui()

    # ---------------------------------------------------------
    # UI
    # ---------------------------------------------------------

    def _build_ui(self):

        layout = QVBoxLayout(self)

        # -----------------------------------------------------
        # Description
        # -----------------------------------------------------

        description = QLabel(
            "Configure the hyperparameter search space for the "
            f"{self.model_name} model."
        )

        description.setWordWrap(True)

        layout.addWidget(description)

        # -----------------------------------------------------
        # Search method
        # -----------------------------------------------------

        method_layout = QHBoxLayout()

        method_layout.addWidget(
            QLabel("Search Method:")
        )

        self.method_combo = QComboBox()

        self.method_combo.addItems([
            "Grid Search",
            "Random Search",
        ])

        index = self.method_combo.findText(
            self.tuning_method
        )

        if index >= 0:
            self.method_combo.setCurrentIndex(index)

        method_layout.addWidget(
            self.method_combo
        )

        method_layout.addStretch()

        layout.addLayout(method_layout)

        # -----------------------------------------------------
        # Parameter table
        # -----------------------------------------------------

        layout.addWidget(
            QLabel("Hyperparameter Search Space")
        )

        self.parameter_table = QTableWidget()

        self.parameter_table.setColumnCount(2)

        self.parameter_table.setHorizontalHeaderLabels([
            "Parameter",
            "Search Values",
        ])

        self.parameter_table.horizontalHeader().setSectionResizeMode(
            0,
            QHeaderView.ResizeToContents
        )

        self.parameter_table.horizontalHeader().setSectionResizeMode(
            1,
            QHeaderView.Stretch
        )

        layout.addWidget(
            self.parameter_table
        )

        self._populate_parameters()

        # -----------------------------------------------------
        # CV
        # -----------------------------------------------------

        cv_layout = QHBoxLayout()

        cv_layout.addWidget(
            QLabel("Cross Validation (CV):")
        )

        self.cv_spin = QSpinBox()

        self.cv_spin.setMinimum(2)
        self.cv_spin.setMaximum(20)
        self.cv_spin.setValue(5)

        cv_layout.addWidget(
            self.cv_spin
        )

        cv_layout.addStretch()

        layout.addLayout(cv_layout)

        # -----------------------------------------------------
        # Scoring
        # -----------------------------------------------------

        scoring_layout = QHBoxLayout()

        scoring_layout.addWidget(
            QLabel("Scoring:")
        )

        self.scoring_combo = QComboBox()

        self.scoring_combo.addItems([
            "accuracy",
            "f1_macro",
            "precision_macro",
            "recall_macro",
        ])

        scoring_layout.addWidget(
            self.scoring_combo
        )

        scoring_layout.addStretch()

        layout.addLayout(scoring_layout)

        # -----------------------------------------------------
        # Random Search options
        # -----------------------------------------------------

        random_layout = QHBoxLayout()

        random_layout.addWidget(
            QLabel("Random Search Iterations:")
        )

        self.n_iter_spin = QSpinBox()

        self.n_iter_spin.setMinimum(1)
        self.n_iter_spin.setMaximum(1000)
        self.n_iter_spin.setValue(20)

        random_layout.addWidget(
            self.n_iter_spin
        )

        random_layout.addStretch()

        layout.addLayout(random_layout)

        self.method_combo.currentTextChanged.connect(
            self._update_random_options
        )

        self._update_random_options(
            self.method_combo.currentText()
        )

        # -----------------------------------------------------
        # Buttons
        # -----------------------------------------------------

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok |
            QDialogButtonBox.Cancel
        )

        buttons.accepted.connect(
            self._validate_and_accept
        )

        buttons.rejected.connect(
            self.reject
        )

        layout.addWidget(buttons)

    # ---------------------------------------------------------
    # Populate parameters
    # ---------------------------------------------------------

    def _populate_parameters(self):

        parameters = self.DEFAULT_SPACES.get(
            self.model_name,
            {}
        )

        self.parameter_table.setRowCount(
            len(parameters)
        )

        for row, (parameter, values) in enumerate(
            parameters.items()
        ):

            parameter_item = QTableWidgetItem(
                parameter
            )

            parameter_item.setFlags(
                parameter_item.flags()
                & ~parameter_item.flags().ItemIsEditable
            )

            self.parameter_table.setItem(
                row,
                0,
                parameter_item
            )

            values_item = QTableWidgetItem(
                values
            )

            self.parameter_table.setItem(
                row,
                1,
                values_item
            )

    # ---------------------------------------------------------
    # Random Search visibility
    # ---------------------------------------------------------

    def _update_random_options(self, method):

        enabled = method == "Random Search"

        self.n_iter_spin.setEnabled(
            enabled
        )

    # ---------------------------------------------------------
    # Parse values
    # ---------------------------------------------------------

    @staticmethod
    def _parse_values(text):

        text = text.strip()

        if not text:
            raise ValueError(
                "Search values cannot be empty."
            )

        # Wrap values in a list so we can parse
        # comma-separated Python literals.

        expression = f"[{text}]"

        try:
            values = ast.literal_eval(
                expression
            )
        except Exception as exc:
            raise ValueError(
                f"Invalid search values: {text}"
            ) from exc

        if not isinstance(values, list):
            raise ValueError(
                "Search values must be a list."
            )

        if not values:
            raise ValueError(
                "Search values cannot be empty."
            )

        return values

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    def _validate_and_accept(self):

        try:

            param_grid = {}

            for row in range(
                self.parameter_table.rowCount()
            ):

                parameter_item = (
                    self.parameter_table.item(
                        row,
                        0
                    )
                )

                values_item = (
                    self.parameter_table.item(
                        row,
                        1
                    )
                )

                if parameter_item is None:
                    continue

                if values_item is None:
                    raise ValueError(
                        "Invalid parameter row."
                    )

                parameter = (
                    parameter_item.text().strip()
                )

                values_text = (
                    values_item.text().strip()
                )

                values = self._parse_values(
                    values_text
                )

                param_grid[parameter] = values

            if not param_grid:
                raise ValueError(
                    "At least one hyperparameter "
                    "must be configured."
                )

            self.config = {
                "param_grid": param_grid,
                "cv": self.cv_spin.value(),
                "scoring": self.scoring_combo.currentText(),
                "n_iter": self.n_iter_spin.value(),
                "random_state": 42,
            }

            self.accept()

        except ValueError as exc:

            QMessageBox.warning(
                self,
                "Invalid Configuration",
                str(exc)
            )

    # ---------------------------------------------------------
    # Get configuration
    # ---------------------------------------------------------

    def get_config(self):

        return getattr(
            self,
            "config",
            None
        )