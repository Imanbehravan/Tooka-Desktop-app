from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
)

from app.ui.sidebar import Sidebar
from app.ui.header import Header
from app.ui.page_stack import PageStack


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Tooka Desktop")
        self.resize(1400, 900)
        self.setMinimumSize(1200, 800)

        self.build_ui()
        self.connect_signals()

    def build_ui(self):

        # =========================================================
        # Central Widget
        # =========================================================

        central = QWidget()
        central.setObjectName("MainCentralWidget")

        self.setCentralWidget(central)

        # =========================================================
        # Root Layout
        # =========================================================

        root_layout = QHBoxLayout(central)

        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # =========================================================
        # Sidebar
        # =========================================================

        self.sidebar = Sidebar()

        root_layout.addWidget(self.sidebar)

        # =========================================================
        # Right Area
        # =========================================================

        right_widget = QWidget()
        right_widget.setObjectName("RightArea")

        right_layout = QVBoxLayout(right_widget)

        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        # =========================================================
        # Header
        # =========================================================

        self.header = Header()

        right_layout.addWidget(self.header)

        # =========================================================
        # Page Stack
        # =========================================================

        self.page_stack = PageStack()

        right_layout.addWidget(
            self.page_stack,
            1
        )

        # Add Right Area to Root Layout

        root_layout.addWidget(
            right_widget,
            1
        )

    # =============================================================
    # Signals
    # =============================================================

    def connect_signals(self):

        self.sidebar.page_changed.connect(
            self.page_stack.setCurrentIndex
        )