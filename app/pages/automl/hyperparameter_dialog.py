import ast

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QPushButton,
    QSpinBox,
    QComboBox,
    QMessageBox,
    QHeaderView,
)


MODEL_SEARCH_SPACES = {
    "KNN": {
        "n_neighbors": ["3", "5", "7"],
        "weights": ["uniform", "distance"],
        "p": ["1", "2"],
    },

    "SVM": {
        "C": ["0.1", "1", "10"],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"],
    },

    "MLP": {
        "hidden_layer_sizes": [
            "(50,)",
            "(100,)",
            "(50, 50)",
        ],
        "activation": [
            "relu",
            "tanh",
        ],
        "alpha": [
            "0.0001",
            "0.001",
        ],
        "learning_rate": [
            "constant",
            "adaptive",
        ],
    },

    "Random Forest": {
        "n_estimators": [
            "50",
            "100",
            "200",
        ],
        "max_depth": [
            "None",
            "5",
            "10",
        ],
        "min_samples_split": [
            "2",
            "5",
            "10",
        ],
    },

    "Naive Bayes": {
        "var_smoothing": [
            "1e-9",
            "1e-8",
            "1e-7",
        ],
    },

    "Decision Tree": {
        "criterion": [
            "gini",
            "entropy",
        ],
        "max_depth": [
            "None",
            "5",
            "10",
        ],
        "min_samples_split": [
            "2",
            "5",
            "10",
        ],
    },
}


class HyperparameterConfigDialog(QDialog):

    def __init__(
        self,
        model_name,
        tuning_method="Grid Search",
        parent=None,
    ):
        super().__init__(parent)

        self.model_name = model_name
        self.tuning_method = tuning_method
        self.configuration = None

        self.setWindowTitle(
            f"{model_name} - Hyperparameter Configuration"
        )

        self.resize(700, 600)

        self._build_ui()
        self._load_search_space()

    # =========================================================
    # UI
    # =========================================================

    def _build_ui(self):

        layout = QVBoxLayout(self)

        # -----------------------------------------------------
        # Title
        # -----------------------------------------------------

        title = QLabel(
            f"Hyperparameters for {self.model_name}"
        )

        title.setObjectName(
            "SectionTitle"
        )

        layout.addWidget(title)

        description = QLabel(
            "Enter comma-separated values for each parameter."
        )

        layout.addWidget(description)

        # -----------------------------------------------------
        # Parameter table
        # -----------------------------------------------------

        self.parameter_table = QTableWidget()

        self.parameter_table.setColumnCount(
            2
        )

        self.parameter_table.setHorizontalHeaderLabels(
            [
                "Parameter",
                "Search Values",
            ]
        )

        self.parameter_table.horizontalHeader().setSectionResizeMode(
            0,
            QHeaderView.ResizeToContents,
        )

        self.parameter_table.horizontalHeader().setSectionResizeMode(
            1,
            QHeaderView.Stretch,
        )

        layout.addWidget(
            self.parameter_table
        )

        # -----------------------------------------------------
        # CV
        # -----------------------------------------------------

        cv_layout = QHBoxLayout()

        cv_label = QLabel(
            "Cross Validation:"
        )

        self.cv_spinbox = QSpinBox()

        self.cv_spinbox.setRange(
            2,
            20,
        )

        self.cv_spinbox.setValue(
            5
        )

        cv_layout.addWidget(
            cv_label
        )

        cv_layout.addWidget(
            self.cv_spinbox
        )

        cv_layout.addStretch()

        layout.addLayout(
            cv_layout
        )

        # -----------------------------------------------------
        # Scoring
        # -----------------------------------------------------

        scoring_layout = QHBoxLayout()

        scoring_label = QLabel(
            "Scoring:"
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

        scoring_layout.addWidget(
            scoring_label
        )

        scoring_layout.addWidget(
            self.scoring_combo
        )

        scoring_layout.addStretch()

        layout.addLayout(
            scoring_layout
        )

        # -----------------------------------------------------
        # Random Search iterations
        # -----------------------------------------------------

        self.n_iter_layout = QHBoxLayout()

        n_iter_label = QLabel(
            "Random Search Iterations:"
        )

        self.n_iter_spinbox = QSpinBox()

        self.n_iter_spinbox.setRange(
            1,
            1000,
        )

        self.n_iter_spinbox.setValue(
            20
        )

        self.n_iter_layout.addWidget(
            n_iter_label
        )

        self.n_iter_layout.addWidget(
            self.n_iter_spinbox
        )

        self.n_iter_layout.addStretch()

        layout.addLayout(
            self.n_iter_layout
        )

        self.n_iter_layout_widget = (
            n_iter_label,
            self.n_iter_spinbox,
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
            "Save Configuration"
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

        self._update_random_search_visibility()

    # =========================================================
    # SEARCH SPACE
    # =========================================================

    def _load_search_space(self):

        search_space = MODEL_SEARCH_SPACES.get(
            self.model_name,
            {},
        )

        self.parameter_table.setRowCount(
            len(search_space)
        )

        for row, (
            parameter,
            values,
        ) in enumerate(
            search_space.items()
        ):

            parameter_item = QTableWidgetItem(
                parameter
            )

            values_item = QTableWidgetItem(
                ", ".join(values)
            )

            self.parameter_table.setItem(
                row,
                0,
                parameter_item
            )

            self.parameter_table.setItem(
                row,
                1,
                values_item
            )

    # =========================================================
    # RANDOM SEARCH
    # =========================================================

    def _update_random_search_visibility(
        self,
    ):

        is_random = (
            self.tuning_method
            == "Random Search"
        )

        for item in self.n_iter_layout_widget:

            item.setVisible(
                is_random
            )

    # =========================================================
    # PARSE SINGLE VALUE
    # =========================================================

    def _parse_value(
        self,
        value,
    ):
        """
        Convert a single string value into the
        appropriate Python type.

        Examples:

            "10"          -> 10
            "0.1"         -> 0.1
            "None"        -> None
            "true"        -> True
            "(50,)"       -> (50,)
            "(50, 50)"    -> (50, 50)
            "rbf"         -> "rbf"
        """

        value = value.strip()

        if not value:
            raise ValueError(
                "Empty hyperparameter value."
            )

        # -----------------------------------------------------
        # None
        # -----------------------------------------------------

        if value == "None":
            return None

        # -----------------------------------------------------
        # Boolean
        # -----------------------------------------------------

        if value.lower() == "true":
            return True

        if value.lower() == "false":
            return False

        # -----------------------------------------------------
        # Tuple / List / Literal values
        # -----------------------------------------------------

        if (
            value.startswith("(")
            and value.endswith(")")
        ) or (
            value.startswith("[")
            and value.endswith("]")
        ):

            try:
                parsed = ast.literal_eval(
                    value
                )

                return parsed

            except (
                ValueError,
                SyntaxError,
            ) as exc:

                raise ValueError(
                    f"Invalid tuple/list value: {value}"
                ) from exc

        # -----------------------------------------------------
        # Integer
        # -----------------------------------------------------

        try:
            return int(value)

        except ValueError:
            pass

        # -----------------------------------------------------
        # Float
        # -----------------------------------------------------

        try:
            return float(value)

        except ValueError:
            pass

        # -----------------------------------------------------
        # String
        # -----------------------------------------------------

        return value

    # =========================================================
    # PARSE SEARCH VALUES
    # =========================================================

    def _parse_search_values(
        self,
        raw_values,
    ):
        """
        Parse comma-separated search values while
        preserving tuples such as:

            (50, 50)

        Example:

            3, 5, 7
                ->
            [3, 5, 7]

            (50,), (100,), (50, 50)
                ->
            [(50,), (100,), (50, 50)]
        """

        raw_values = raw_values.strip()

        if not raw_values:
            raise ValueError(
                "Search values cannot be empty."
            )

        values = []

        current = []
        parentheses_depth = 0
        brackets_depth = 0

        for char in raw_values:

            # -------------------------------------------------
            # Opening brackets
            # -------------------------------------------------

            if char == "(":

                parentheses_depth += 1

            elif char == ")":

                parentheses_depth -= 1

                if parentheses_depth < 0:
                    raise ValueError(
                        "Invalid parentheses in search values."
                    )

            elif char == "[":

                brackets_depth += 1

            elif char == "]":

                brackets_depth -= 1

                if brackets_depth < 0:
                    raise ValueError(
                        "Invalid brackets in search values."
                    )

            # -------------------------------------------------
            # Comma
            # -------------------------------------------------

            if (
                char == ","
                and parentheses_depth == 0
                and brackets_depth == 0
            ):

                value = "".join(
                    current
                ).strip()

                if value:
                    values.append(
                        self._parse_value(
                            value
                        )
                    )

                current = []

            else:

                current.append(
                    char
                )

        # -----------------------------------------------------
        # Invalid brackets
        # -----------------------------------------------------

        if parentheses_depth != 0:
            raise ValueError(
                "Unbalanced parentheses in search values."
            )

        if brackets_depth != 0:
            raise ValueError(
                "Unbalanced brackets in search values."
            )

        # -----------------------------------------------------
        # Last value
        # -----------------------------------------------------

        value = "".join(
            current
        ).strip()

        if value:
            values.append(
                self._parse_value(
                    value
                )
            )

        if not values:
            raise ValueError(
                "At least one search value is required."
            )

        return values

    # =========================================================
    # SAVE
    # =========================================================

    def _save(self):

        param_grid = {}

        for row in range(
            self.parameter_table.rowCount()
        ):

            parameter_item = (
                self.parameter_table.item(
                    row,
                    0,
                )
            )

            values_item = (
                self.parameter_table.item(
                    row,
                    1,
                )
            )

            if (
                parameter_item is None
                or values_item is None
            ):
                continue

            parameter = (
                parameter_item.text().strip()
            )

            raw_values = (
                values_item.text().strip()
            )

            if not parameter:
                continue

            if not raw_values:

                QMessageBox.warning(
                    self,
                    "Invalid Configuration",
                    f"No values specified for '{parameter}'.",
                )

                return

            try:

                values = (
                    self._parse_search_values(
                        raw_values
                    )
                )

            except ValueError as exc:

                QMessageBox.warning(
                    self,
                    "Invalid Value",
                    f"{parameter}: {exc}",
                )

                return

            param_grid[
                parameter
            ] = values

        if not param_grid:

            QMessageBox.warning(
                self,
                "Invalid Configuration",
                "At least one hyperparameter is required.",
            )

            return

        self.configuration = {
            "param_grid": param_grid,

            "cv": self.cv_spinbox.value(),

            "scoring": (
                self.scoring_combo.currentText()
            ),

            "n_iter": (
                self.n_iter_spinbox.value()
            ),

            "random_state": 42,
        }

        self.accept()

    # =========================================================
    # RESULT
    # =========================================================

    def get_configuration(self):

        return self.configuration