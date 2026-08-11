from PySide6.QtWidgets import (
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
)

from app.widgets.card import Card


class WelcomeCard(Card):

    def __init__(self):
        super().__init__()

        self.setMinimumHeight(180)

        layout = QHBoxLayout(self)

        layout.setContentsMargins(
            28,
            24,
            28,
            24,
        )

        # ---------------------------------------------------------
        # Left
        # ---------------------------------------------------------

        left = QVBoxLayout()

        left.setSpacing(8)

        title = QLabel(
            "Welcome back 👋"
        )

        title.setObjectName(
            "WelcomeTitle"
        )

        description = QLabel(
            "Build, train and deploy machine "
            "learning models from one place."
        )

        description.setObjectName(
            "WelcomeDescription"
        )

        description.setWordWrap(True)

        left.addWidget(title)
        left.addWidget(description)

        left.addStretch()

        # ---------------------------------------------------------
        # Button
        # ---------------------------------------------------------

        new_project = QPushButton(
            "+  New Project"
        )

        new_project.setObjectName(
            "PrimaryButton"
        )

        new_project.setFixedHeight(42)

        left.addWidget(
            new_project,
            0,
        )

        layout.addLayout(left)

        # ---------------------------------------------------------
        # Right Decoration
        # ---------------------------------------------------------

        decoration = QWidget()

        decoration.setObjectName(
            "WelcomeDecoration"
        )

        decoration.setMinimumWidth(
            220
        )

        layout.addWidget(
            decoration
        )