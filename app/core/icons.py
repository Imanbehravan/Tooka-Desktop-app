from pathlib import Path


class Icons:

    ROOT = (
        Path(__file__).parent.parent
        / "resources"
        / "icons"
    )

    DASHBOARD = ROOT / "dashboard.svg"

    DATASET = ROOT / "dataset.svg"

    MODEL = ROOT / "model.svg"

    AUTOML = ROOT / "automl.svg"

    SETTINGS = ROOT / "settings.svg"