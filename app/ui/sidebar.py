from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
)

from app.ui.components.logo_widget import LogoWidget
from app.ui.components.nav_button import NavButton

class Sidebar(QWidget):
    """
    Main navigation sidebar for Tooka Desktop.
    """

    # Emits the index of the selected page
    page_changed = Signal(int)

    def __init__(self):
        super().__init__()

        self.setObjectName("Sidebar")
        self.setFixedWidth(250)

        self.buttons = []

        self.build_ui()
        self.connect_signals()

    # =============================================================
    # UI
    # =============================================================

    def build_ui(self):

        layout = QVBoxLayout(self)

        layout.setContentsMargins(
            15,
            20,
            15,
            20,
        )

        layout.setSpacing(6)

        # ---------------------------------------------------------
        # Logo
        # ---------------------------------------------------------

        self.logo = LogoWidget()

        layout.addWidget(self.logo)

        layout.addSpacing(25)

        # ---------------------------------------------------------
        # Navigation
        # ---------------------------------------------------------

        navigation_items = [
            "Dashboard",
            "Datasets",
            "Models",
            "AutoML",
            "Deploy",
            "Monitoring",
            "Settings",
        ]

        for index, text in enumerate(navigation_items):

            button = NavButton(
                text=text,
                index=index,
            )

            # Dashboard selected by default
            if index == 0:
                button.setChecked(True)

            layout.addWidget(button)

            self.buttons.append(button)

        # ---------------------------------------------------------
        # Bottom Space
        # ---------------------------------------------------------

        layout.addStretch()

    # =============================================================
    # Signals
    # =============================================================

    def connect_signals(self):

        for button in self.buttons:

            button.clicked.connect(
                lambda checked=False, index=button.page_index:
                self.handle_navigation(index)
            )

    # =============================================================
    # Navigation
    # =============================================================

    def handle_navigation(self, index: int):

        self.set_active_button(index)

        self.page_changed.emit(index)

    # =============================================================
    # Active Button
    # =============================================================

    def set_active_button(self, index: int):

        for button in self.buttons:

            button.setChecked(
                button.page_index == index
            )