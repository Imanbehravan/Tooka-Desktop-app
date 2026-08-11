from PySide6.QtWidgets import QStackedWidget

from app.pages.dashboard.dashboard import Dashboard
from app.pages.datasets.page import DatasetPage
from app.pages.models.page import ModelPage
from app.pages.automl.page import AutoMLPage
from app.pages.deploy.page import DeployPage
from app.pages.monitoring.page import MonitoringPage
from app.pages.settings.page import SettingsPage


class PageStack(QStackedWidget):

    def __init__(self):
        super().__init__()

        self.addWidget(Dashboard())
        self.addWidget(DatasetPage())
        self.addWidget(ModelPage())
        self.addWidget(AutoMLPage())
        self.addWidget(DeployPage())
        self.addWidget(MonitoringPage())
        self.addWidget(SettingsPage())