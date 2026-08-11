from PySide6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    Qt,
)

from PySide6.QtWidgets import (
    QLabel,
    QTableView,
    QVBoxLayout,
    QHeaderView,
)

from app.widgets.card import Card


class ProjectModel(QAbstractTableModel):

    headers = [
        "Project",
        "Dataset",
        "Status",
        "Accuracy",
    ]

    projects = [
        [
            "Fraud Detection",
            "transactions.csv",
            "Running",
            "94.3%",
        ],
        [
            "Cancer Classifier",
            "cancer.csv",
            "Completed",
            "98.6%",
        ],
        [
            "Customer Churn",
            "customers.csv",
            "Training",
            "-",
        ],
        [
            "Sales Forecast",
            "sales.csv",
            "Completed",
            "92.1%",
        ],
    ]

    def rowCount(
        self,
        parent=QModelIndex(),
    ):
        return len(self.projects)

    def columnCount(
        self,
        parent=QModelIndex(),
    ):
        return len(self.headers)

    def data(
        self,
        index,
        role=Qt.DisplayRole,
    ):

        if not index.isValid():
            return None

        if role == Qt.DisplayRole:

            return self.projects[
                index.row()
            ][
                index.column()
            ]

        return None

    def headerData(
        self,
        section,
        orientation,
        role=Qt.DisplayRole,
    ):

        if (
            role == Qt.DisplayRole
            and orientation == Qt.Horizontal
        ):
            return self.headers[section]

        return None


class RecentProjects(Card):

    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)

        layout.setContentsMargins(
            22,
            20,
            22,
            20,
        )

        layout.setSpacing(15)

        # ---------------------------------------------------------
        # Header
        # ---------------------------------------------------------

        title = QLabel(
            "Recent Projects"
        )

        title.setObjectName(
            "SectionTitle"
        )

        layout.addWidget(title)

        # ---------------------------------------------------------
        # Table
        # ---------------------------------------------------------

        self.table = QTableView()

        self.table.setObjectName(
            "ProjectsTable"
        )

        self.model = ProjectModel()

        self.table.setModel(
            self.model
        )

        self.table.setSelectionBehavior(
            QTableView.SelectRows
        )

        self.table.setSelectionMode(
            QTableView.SingleSelection
        )

        self.table.setEditTriggers(
            QTableView.NoEditTriggers
        )

        self.table.verticalHeader().hide()

        header = self.table.horizontalHeader()

        header.setStretchLastSection(True)

        header.setSectionResizeMode(
            0,
            QHeaderView.Stretch,
        )

        header.setSectionResizeMode(
            1,
            QHeaderView.Stretch,
        )

        header.setSectionResizeMode(
            2,
            QHeaderView.ResizeToContents,
        )

        header.setSectionResizeMode(
            3,
            QHeaderView.ResizeToContents,
        )

        layout.addWidget(
            self.table
        )