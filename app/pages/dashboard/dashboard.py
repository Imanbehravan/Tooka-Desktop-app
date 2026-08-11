from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
)

from app.widgets.stat_card import StatCard

from app.pages.dashboard.welcome_card import (
    WelcomeCard,
)

from app.pages.dashboard.recent_projects import (
    RecentProjects,
)

from app.pages.dashboard.activity_panel import (
    ActivityPanel,
)

from app.pages.dashboard.quick_actions import (
    QuickActions,
)


class Dashboard(QWidget):

    def __init__(self):
        super().__init__()

        self.setObjectName(
            "Dashboard"
        )

        self.build_ui()

    def build_ui(self):

        root = QVBoxLayout(self)

        root.setContentsMargins(
            30,
            28,
            30,
            30,
        )

        root.setSpacing(20)

        # =========================================================
        # Welcome
        # =========================================================

        root.addWidget(
            WelcomeCard()
        )

        # =========================================================
        # Statistics
        # =========================================================

        stats = QGridLayout()

        stats.setHorizontalSpacing(
            16
        )

        stats.setVerticalSpacing(
            16
        )

        stats.addWidget(
            StatCard(
                "▣",
                "Projects",
                "12",
                "+2 this week",
            ),
            0,
            0,
        )

        stats.addWidget(
            StatCard(
                "◈",
                "Models",
                "31",
                "+5 trained",
            ),
            0,
            1,
        )

        stats.addWidget(
            StatCard(
                "◎",
                "Best Accuracy",
                "98.6%",
                "+2.4% this month",
            ),
            0,
            2,
        )

        stats.addWidget(
            StatCard(
                "↗",
                "Deployments",
                "4",
                "All systems online",
            ),
            0,
            3,
        )

        root.addLayout(
            stats
        )

        # =========================================================
        # Bottom Content
        # =========================================================

        content = QHBoxLayout()

        content.setSpacing(
            20
        )

        # Recent Projects

        projects = RecentProjects()

        content.addWidget(
            projects,
            2,
        )

        # Right Column

        right = QVBoxLayout()

        right.setSpacing(
            20
        )

        right.addWidget(
            ActivityPanel(),
            1,
        )

        right.addWidget(
            QuickActions(),
            1,
        )

        content.addLayout(
            right,
            1,
        )

        root.addLayout(
            content,
            1,
        )