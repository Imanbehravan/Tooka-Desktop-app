from PySide6.QtWidgets import (
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from app.widgets.card import Card


class QuickActions(Card):

    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)

        layout.setContentsMargins(
            22,
            20,
            22,
            20,
        )

        layout.setSpacing(10)

        title = QLabel(
            "Quick Actions"
        )

        title.setObjectName(
            "SectionTitle"
        )

        layout.addWidget(title)

        actions = [
            "+  New Project",
            "↥  Upload Dataset",
            "⚙  Train Model",
            "🚀  Deploy Model",
        ]

        for text in actions:

            button = QPushButton(
                text
            )

            button.setObjectName(
                "QuickActionButton"
            )

            button.setMinimumHeight(
                42
            )

            layout.addWidget(
                button
            )

        layout.addStretch()