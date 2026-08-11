from PySide6.QtWidgets import (
    QLabel,
    QVBoxLayout,
    QWidget,
)

from app.widgets.card import Card

class ActivityPanel(Card):

    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)

        layout.setContentsMargins(
            22,
            20,
            22,
            20,
        )

        layout.setSpacing(14)

        title = QLabel(
            "Recent Activity"
        )

        title.setObjectName(
            "SectionTitle"
        )

        layout.addWidget(title)

        activities = [
            (
                "Model training completed",
                "Fraud Detection",
                "5 min ago",
            ),
            (
                "Dataset uploaded",
                "transactions.csv",
                "32 min ago",
            ),
            (
                "New project created",
                "Customer Churn",
                "1 hour ago",
            ),
            (
                "Deployment started",
                "Sales Forecast",
                "2 hours ago",
            ),
        ]

        for title_text, project, time in activities:

            item = QWidget()

            item.setObjectName(
                "ActivityItem"
            )

            item_layout = QVBoxLayout(
                item
            )

            item_layout.setContentsMargins(
                0,
                4,
                0,
                4,
            )

            item_layout.setSpacing(2)

            title_label = QLabel(
                title_text
            )

            title_label.setObjectName(
                "ActivityTitle"
            )

            project_label = QLabel(
                f"{project}  •  {time}"
            )

            project_label.setObjectName(
                "ActivitySubtitle"
            )

            item_layout.addWidget(
                title_label
            )

            item_layout.addWidget(
                project_label
            )

            layout.addWidget(
                item
            )

        layout.addStretch()