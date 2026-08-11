from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel


class LogoWidget(QLabel):

    def __init__(self):
        super().__init__()

        self.setObjectName("SidebarLogo")

        self.setAlignment(
            Qt.AlignCenter
        )

        self.setText("TOOKA")