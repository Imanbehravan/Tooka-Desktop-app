from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QPushButton,
    QHBoxLayout,
    QLineEdit,
)


class Header(QWidget):

    def __init__(self):
        super().__init__()

        self.setObjectName("Header")
        self.setFixedHeight(70)

        self.build_ui()

    def build_ui(self):

        layout = QHBoxLayout(self)

        layout.setContentsMargins(20, 12, 20, 12)
        layout.setSpacing(15)

        # ---------------- Title ----------------

        title = QLabel("Tooka Desktop")
        title.setObjectName("HeaderTitle")

        layout.addWidget(title)

        layout.addSpacing(25)

        # ---------------- Search ----------------

        self.search = QLineEdit()

        self.search.setPlaceholderText("Search projects, datasets, models ...")

        self.search.setObjectName("SearchBox")

        self.search.setMinimumHeight(40)

        layout.addWidget(self.search, 1)

        # ---------------- Notification ----------------

        self.notification = QPushButton("🔔")
        self.notification.setObjectName("HeaderIcon")
        self.notification.setFixedSize(42, 42)

        layout.addWidget(self.notification)

        # ---------------- User ----------------

        self.user = QPushButton("👤  Mahdi")

        self.user.setObjectName("UserButton")

        self.user.setFixedHeight(42)

        layout.addWidget(self.user)