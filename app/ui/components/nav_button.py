from PySide6.QtCore import Qt
from PySide6.QtWidgets import QPushButton


class NavButton(QPushButton):

    def __init__(self, text: str, index: int):
        super().__init__(text)

        self.page_index = index

        self.setObjectName("NavButton")

        self.setCheckable(True)

        self.setCursor(
            Qt.PointingHandCursor
        )

        self.setMinimumHeight(46)