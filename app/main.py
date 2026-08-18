import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from app.ui.main_window import MainWindow

def load_stylesheet(app: QApplication) -> None:
    """
    Load the global Tooka QSS stylesheet.
    """

    style_file = (
        Path(__file__).parent
        / "styles"
        / "dark.qss"
    )

    if not style_file.exists():
        print(
            f"Warning: stylesheet not found: {style_file}"
        )
        return

    try:
        with open(
            style_file,
            "r",
            encoding="utf-8",
        ) as file:
            stylesheet = file.read()

        app.setStyleSheet(stylesheet)

    except OSError as error:
        print(
            f"Warning: could not load stylesheet: {error}"
        )


def main() -> int:

    # =========================================================
    # Application
    # =========================================================

    app = QApplication(sys.argv)

    app.setApplicationName(
        "Tooka Desktop"
    )

    app.setOrganizationName(
        "Tooka AI"
    )

    app.setApplicationDisplayName(
        "Tooka Desktop"
    )

    # =========================================================
    # Global Style
    # =========================================================

    load_stylesheet(app)

    # =========================================================
    # Main Window
    # =========================================================

    window = MainWindow()

    window.show()

    # =========================================================
    # Event Loop
    # =========================================================

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())