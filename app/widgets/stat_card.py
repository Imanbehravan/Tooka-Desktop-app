from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
)

from app.widgets.card import Card

class StatCard(Card):

    def __init__(
        self,
        icon: str,
        title: str,
        value: str,
        subtitle: str,
    ):
        super().__init__()

        self.setMinimumHeight(125)

        layout = QHBoxLayout(self)

        layout.setContentsMargins(
            20,
            18,
            20,
            18,
        )

        layout.setSpacing(15)

        # ---------------------------------------------------------
        # Icon
        # ---------------------------------------------------------

        icon_label = QLabel(icon)

        icon_label.setObjectName(
            "StatIcon"
        )

        icon_label.setFixedSize(
            48,
            48,
        )

        icon_label.setAlignment(
            Qt.AlignCenter
        )

        layout.addWidget(icon_label)

        # ---------------------------------------------------------
        # Content
        # ---------------------------------------------------------

        content = QVBoxLayout()

        content.setSpacing(3)

        title_label = QLabel(title)

        title_label.setObjectName(
            "StatTitle"
        )

        value_label = QLabel(value)

        value_label.setObjectName(
            "StatValue"
        )

        subtitle_label = QLabel(subtitle)

        subtitle_label.setObjectName(
            "StatSubtitle"
        )

        content.addWidget(title_label)
        content.addWidget(value_label)
        content.addWidget(subtitle_label)

        layout.addLayout(content)