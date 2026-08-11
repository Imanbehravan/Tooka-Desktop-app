from pathlib import Path

from PySide6.QtWidgets import QApplication


class ThemeManager:

    @staticmethod
    def load(app: QApplication):
        style_file = (
            Path(__file__).parent.parent
            / "styles"
            / "dark.qss"
        )

        if style_file.exists():
            with open(style_file, "r", encoding="utf-8") as f:
                app.setStyleSheet(f.read())