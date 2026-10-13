from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
QWidget,
QLabel,
QVBoxLayout,
QHBoxLayout,
QLineEdit,
QComboBox,
QPushButton,
QTableWidget,
QTableWidgetItem,
QHeaderView,
QMenu,
QMessageBox,
QSizePolicy,
QInputDialog,
)

from app.core.ml.model_registry import ModelRegistry

class ModelPage(QWidget):


    def __init__(self):
        super().__init__()

        self.setObjectName("ModelsPage")

        self.registry = ModelRegistry()

        self.models = []
        self.filtered_models = []

        self.build_ui()
        self.refresh_models()

    # =============================================================
    # UI
    # =============================================================

    def build_ui(self):

        main_layout = QVBoxLayout(self)

        main_layout.setContentsMargins(
            32,
            28,
            32,
            32
        )

        main_layout.setSpacing(24)

        # =========================================================
        # Page Header
        # =========================================================

        header_layout = QHBoxLayout()
        header_layout.setSpacing(16)

        title_layout = QVBoxLayout()
        title_layout.setSpacing(4)

        title = QLabel("Models")
        title.setObjectName("PageTitle")

        subtitle = QLabel(
            "Manage and organize your trained machine learning models."
        )
        subtitle.setObjectName("PageSubtitle")

        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)

        header_layout.addLayout(title_layout)
        header_layout.addStretch()

        main_layout.addLayout(header_layout)

        # =========================================================
        # Statistics
        # =========================================================

        stats_layout = QHBoxLayout()
        stats_layout.setSpacing(16)

        self.total_card = self.create_stat_card(
            "Total Models",
            "0"
        )

        self.classification_card = self.create_stat_card(
            "Classifiers",
            "0"
        )

        self.regression_card = self.create_stat_card(
            "Regressors",
            "0"
        )

        self.ready_card = self.create_stat_card(
            "Ready",
            "0"
        )

        stats_layout.addWidget(self.total_card)
        stats_layout.addWidget(self.classification_card)
        stats_layout.addWidget(self.regression_card)
        stats_layout.addWidget(self.ready_card)

        main_layout.addLayout(stats_layout)

        # =========================================================
        # Search + Filters
        # =========================================================

        toolbar_layout = QHBoxLayout()
        toolbar_layout.setSpacing(10)

        self.search_box = QLineEdit()
        self.search_box.setObjectName("SearchBox")
        self.search_box.setPlaceholderText(
            "Search models..."
        )
        self.search_box.setMinimumHeight(40)

        self.type_filter = QComboBox()
        self.type_filter.setObjectName(
            "ModelFilter"
        )
        self.type_filter.setMinimumHeight(40)

        self.type_filter.addItems([
            "All Types",
            "Classification",
            "Regression",
            "Clustering",
            "Time Series",
        ])

        self.status_filter = QComboBox()
        self.status_filter.setObjectName(
            "ModelFilter"
        )
        self.status_filter.setMinimumHeight(40)

        self.status_filter.addItems([
            "All Status",
            "Ready",
            "Training",
            "Failed",
        ])

        toolbar_layout.addWidget(
            self.search_box,
            1
        )

        toolbar_layout.addWidget(
            self.type_filter
        )

        toolbar_layout.addWidget(
            self.status_filter
        )

        main_layout.addLayout(
            toolbar_layout
        )

        # =========================================================
        # Section Header
        # =========================================================

        section_layout = QHBoxLayout()

        section_title = QLabel(
            "My Models"
        )
        section_title.setObjectName(
            "SectionTitle"
        )

        self.model_count_label = QLabel()
        self.model_count_label.setObjectName(
            "PageSubtitle"
        )

        section_layout.addWidget(
            section_title
        )

        section_layout.addStretch()

        section_layout.addWidget(
            self.model_count_label
        )

        main_layout.addLayout(
            section_layout
        )

        # =========================================================
        # Models Table
        # =========================================================

        self.models_table = QTableWidget()

        self.models_table.setObjectName(
            "ProjectsTable"
        )

        self.models_table.setColumnCount(7)

        self.models_table.setHorizontalHeaderLabels([
            "Model",
            "Type",
            "Algorithm",
            "Metric",
            "Version",
            "Status",
            "Actions",
        ])

        self.models_table.setSelectionBehavior(
            QTableWidget.SelectRows
        )

        self.models_table.setSelectionMode(
            QTableWidget.SingleSelection
        )

        self.models_table.setEditTriggers(
            QTableWidget.NoEditTriggers
        )

        self.models_table.setShowGrid(False)

        self.models_table.verticalHeader().setVisible(
            False
        )

        header = self.models_table.horizontalHeader()

        header.setSectionResizeMode(
            0,
            QHeaderView.Stretch
        )

        header.setSectionResizeMode(
            1,
            QHeaderView.ResizeToContents
        )

        header.setSectionResizeMode(
            2,
            QHeaderView.Stretch
        )

        header.setSectionResizeMode(
            3,
            QHeaderView.ResizeToContents
        )

        header.setSectionResizeMode(
            4,
            QHeaderView.ResizeToContents
        )

        header.setSectionResizeMode(
            5,
            QHeaderView.ResizeToContents
        )

        header.setSectionResizeMode(
            6,
            QHeaderView.ResizeToContents
        )

        self.models_table.verticalHeader().setDefaultSectionSize(
            54
        )

        main_layout.addWidget(
            self.models_table,
            1
        )

        # =========================================================
        # Signals
        # =========================================================

        self.search_box.textChanged.connect(
            self.filter_models
        )

        self.type_filter.currentTextChanged.connect(
            self.filter_models
        )

        self.status_filter.currentTextChanged.connect(
            self.filter_models
        )

    # =============================================================
    # Statistics Card
    # =============================================================

    def create_stat_card(
        self,
        title_text,
        value_text
    ):

        card = QWidget()

        card.setObjectName(
            "Card"
        )

        card.setMinimumHeight(
            105
        )

        layout = QVBoxLayout(card)

        layout.setContentsMargins(
            18,
            16,
            18,
            16
        )

        layout.setSpacing(5)

        title = QLabel(
            title_text
        )
        title.setObjectName(
            "StatTitle"
        )

        value = QLabel(
            value_text
        )
        value.setObjectName(
            "StatValue"
        )

        layout.addWidget(title)
        layout.addWidget(value)

        card.value_label = value

        return card

    # =============================================================
    # Refresh
    # =============================================================

    def refresh_models(self):

        self.models = (
            self.registry.list_models()
        )

        self.filtered_models = list(
            self.models
        )

        self.load_models()

    # =============================================================
    # Load Table
    # =============================================================

    def load_models(self):

        self.models_table.setRowCount(0)

        for model in self.filtered_models:

            row = self.models_table.rowCount()

            self.models_table.insertRow(
                row
            )

            # -----------------------------------------------------
            # Model
            # -----------------------------------------------------

            name_item = QTableWidgetItem(
                model.name
            )

            name_item.setData(
                Qt.UserRole,
                model
            )

            self.models_table.setItem(
                row,
                0,
                name_item
            )

            # -----------------------------------------------------
            # Type
            # -----------------------------------------------------

            self.models_table.setItem(
                row,
                1,
                QTableWidgetItem(
                    model.model_type
                )
            )

            # -----------------------------------------------------
            # Algorithm
            # -----------------------------------------------------

            self.models_table.setItem(
                row,
                2,
                QTableWidgetItem(
                    model.algorithm
                )
            )

            # -----------------------------------------------------
            # Metric
            # -----------------------------------------------------

            metric_text = self.format_metric(
                model
            )

            metric_item = QTableWidgetItem(
                metric_text
            )

            metric_item.setTextAlignment(
                Qt.AlignCenter
            )

            self.models_table.setItem(
                row,
                3,
                metric_item
            )

            # -----------------------------------------------------
            # Version
            # -----------------------------------------------------

            version_item = QTableWidgetItem(
                model.version
            )

            version_item.setTextAlignment(
                Qt.AlignCenter
            )

            self.models_table.setItem(
                row,
                4,
                version_item
            )

            # -----------------------------------------------------
            # Status
            # -----------------------------------------------------

            status_item = QTableWidgetItem(
                "●  " + model.status
            )

            status_item.setTextAlignment(
                Qt.AlignCenter
            )

            self.models_table.setItem(
                row,
                5,
                status_item
            )

            # -----------------------------------------------------
            # Actions
            # -----------------------------------------------------

            action_button = QPushButton(
                "⋮"
            )

            action_button.setObjectName(
                "HeaderIcon"
            )

            action_button.setMinimumWidth(
                38
            )

            action_button.setSizePolicy(
                QSizePolicy.Fixed,
                QSizePolicy.Fixed
            )

            action_button.clicked.connect(
                lambda checked=False,
                model=model:
                self.show_model_menu(model)
            )

            self.models_table.setCellWidget(
                row,
                6,
                action_button
            )

        self.update_statistics()

        self.update_count_label()

    # =============================================================
    # Metric Formatting
    # =============================================================

    @staticmethod
    def format_metric(model):

        value = model.metric_value

        if model.metric_name.lower() in {
            "accuracy",
            "precision",
            "recall",
            "f1",
            "f1-score",
            "r2",
            "r²",
        }:

            if value <= 1:
                return f"{value * 100:.1f}%"

            return f"{value:.1f}%"

        return f"{value:.3f}"

    # =============================================================
    # Filtering
    # =============================================================

    def filter_models(self):

        search_text = (
            self.search_box
            .text()
            .strip()
            .lower()
        )

        selected_type = (
            self.type_filter.currentText()
        )

        selected_status = (
            self.status_filter.currentText()
        )

        self.filtered_models = []

        for model in self.models:

            searchable_text = (
                f"{model.name} "
                f"{model.model_type} "
                f"{model.algorithm} "
                f"{model.metric_name}"
            ).lower()

            matches_search = (
                not search_text
                or search_text in searchable_text
            )

            matches_type = (
                selected_type == "All Types"
                or model.model_type == selected_type
            )

            matches_status = (
                selected_status == "All Status"
                or model.status == selected_status
            )

            if (
                matches_search
                and matches_type
                and matches_status
            ):
                self.filtered_models.append(
                    model
                )

        self.load_models()

    # =============================================================
    # Statistics
    # =============================================================

    def update_statistics(self):

        total = len(
            self.models
        )

        classifiers = sum(
            1
            for model in self.models
            if model.model_type == "Classification"
        )

        regressors = sum(
            1
            for model in self.models
            if model.model_type == "Regression"
        )

        ready = sum(
            1
            for model in self.models
            if model.status == "Ready"
        )

        self.total_card.value_label.setText(
            str(total)
        )

        self.classification_card.value_label.setText(
            str(classifiers)
        )

        self.regression_card.value_label.setText(
            str(regressors)
        )

        self.ready_card.value_label.setText(
            str(ready)
        )

    # =============================================================
    # Count
    # =============================================================

    def update_count_label(self):

        count = len(
            self.filtered_models
        )

        if count == 1:
            text = "1 model"
        else:
            text = f"{count} models"

        self.model_count_label.setText(
            text
        )

    # =============================================================
    # Context Menu
    # =============================================================

    def show_model_menu(
        self,
        model
    ):

        menu = QMenu(
            self
        )

        view_action = menu.addAction(
            "View"
        )

        load_action = menu.addAction(
            "Load"
        )

        menu.addSeparator()

        rename_action = menu.addAction(
            "Rename"
        )

        version_action = menu.addAction(
            "Create Version"
        )

        menu.addSeparator()

        export_action = menu.addAction(
            "Export"
        )

        delete_action = menu.addAction(
            "Delete"
        )

        action = menu.exec(
            self.cursor().pos()
        )

        if action == view_action:
            self.view_model(
                model
            )

        elif action == load_action:
            self.load_model(
                model
            )

        elif action == rename_action:
            self.rename_model(
                model
            )

        elif action == version_action:
            self.create_version(
                model
            )

        elif action == export_action:
            self.export_model(
                model
            )

        elif action == delete_action:
            self.delete_model(
                model
            )

    # =============================================================
    # View
    # =============================================================

    def view_model(
        self,
        model
    ):

        QMessageBox.information(
            self,
            "Model Details",
            (
                f"Name: {model.name}\n"
                f"Type: {model.model_type}\n"
                f"Algorithm: {model.algorithm}\n"
                f"{model.metric_name}: "
                f"{self.format_metric(model)}\n"
                f"Version: {model.version}\n"
                f"Status: {model.status}\n"
                f"Created: {model.created_at}"
            )
        )

    # =============================================================
    # Load
    # =============================================================

    def load_model(
        self,
        model
    ):

        try:

            loaded_model = (
                self.registry.load_model(
                    model.name,
                    model.version
                )
            )

            QMessageBox.information(
                self,
                "Model Loaded",
                (
                    f"'{model.name}' "
                    f"{model.version} was loaded successfully."
                    f"\n\n"
                    f"Model type: "
                    f"{type(loaded_model).__name__}"
                )
            )

        except FileNotFoundError:

            QMessageBox.warning(
                self,
                "Model File Not Found",
                (
                    f"The model metadata exists, "
                    f"but the actual model file "
                    f"has not been created yet.\n\n"
                    f"Expected path:\n"
                    f"{model.model_path}"
                )
            )

        except Exception as error:

            QMessageBox.critical(
                self,
                "Load Error",
                str(error)
            )

    # =============================================================
    # Rename
    # =============================================================

    def rename_model(
        self,
        model
    ):

        new_name, ok = QInputDialog.getText(
            self,
            "Rename Model",
            "New model name:",
            text=model.name
        )

        if not ok:
            return

        new_name = new_name.strip()

        if not new_name:
            return

        try:

            self.registry.rename_model(
                model.name,
                new_name
            )

            self.refresh_models()

        except ValueError as error:

            QMessageBox.warning(
                self,
                "Rename Failed",
                str(error)
            )

        except Exception as error:

            QMessageBox.critical(
                self,
                "Rename Error",
                str(error)
            )

    # =============================================================
    # Create Version
    # =============================================================

    def create_version(
        self,
        model
    ):

        try:

            new_model = (
                self.registry.create_version(
                    model.name
                )
            )

            self.refresh_models()

            QMessageBox.information(
                self,
                "Version Created",
                (
                    f"Created {new_model.version} "
                    f"for '{new_model.name}'."
                )
            )

        except Exception as error:

            QMessageBox.critical(
                self,
                "Version Error",
                str(error)
            )

    # =============================================================
    # Export
    # =============================================================

    def export_model(
        self,
        model
    ):

        QMessageBox.information(
            self,
            "Export Model",
            (
                "Model export will be connected "
                "to the export system in the next stage."
            )
        )

    # =============================================================
    # Delete
    # =============================================================

    def delete_model(
        self,
        model
    ):

        answer = QMessageBox.question(
            self,
            "Delete Model",
            (
                f"Are you sure you want to delete "
                f"'{model.name}' {model.version}?"
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if answer != QMessageBox.Yes:
            return

        try:

            self.registry.delete_model(
                model.name,
                model.version
            )

            self.refresh_models()

        except Exception as error:

            QMessageBox.critical(
                self,
                "Delete Error",
                str(error)
            )
