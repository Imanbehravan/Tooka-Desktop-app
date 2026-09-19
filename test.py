import sys

import pandas as pd
from sklearn.datasets import load_iris

from PySide6.QtWidgets import QApplication

from app.workers.automl_controller import AutoMLController


# =========================================================
# QApplication
# =========================================================

app = QApplication(sys.argv)


# =========================================================
# Dataset
# =========================================================

iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names,
)

df["target"] = iris.target


# =========================================================
# Controller
# =========================================================

controller = AutoMLController()


# =========================================================
# Signals
# =========================================================

controller.started.connect(
    lambda: print("\nAutoML started")
)

controller.status.connect(
    lambda message: print(
        f"STATUS: {message}"
    )
)

controller.progress.connect(
    lambda value: print(
        f"PROGRESS: {value}%"
    )
)


def on_error(message):

    print("\nERROR:")
    print(message)

    app.quit()


def on_finished(result):

    print(
        "\n================================"
    )

    print(
        "AUTOML FINISHED"
    )

    print(
        "================================"
    )

    print(
        "Best Model:",
        result["best_model"],
    )

    print(
        "Best Score:",
        result["best_score"],
    )

    print(
        "\nModel Results:"
    )

    for model_result in result["results"]:

        print(
            model_result["rank"],
            model_result["model"],
            "Accuracy:",
            model_result["accuracy"],
            "Best Params:",
            model_result["best_params"],
            "CV:",
            model_result["best_cv_score"],
        )

    print(
        "\n================================"
    )

    print(
        "RANDOM SEARCH TEST PASSED"
    )

    print(
        "================================"
    )

    app.quit()


controller.error.connect(
    on_error
)

controller.finished.connect(
    on_finished
)


# =========================================================
# Random Search Configuration
# =========================================================

tuning_params = {

    "KNN": {

        "param_grid": {

            "n_neighbors": [
                3,
                5,
                7,
                9,
            ],

            "weights": [
                "uniform",
                "distance",
            ],

            "p": [
                1,
                2,
            ],
        },

        "cv": 5,

        "scoring": "accuracy",

        "n_iter": 5,

        "random_state": 42,
    },

    "SVM": {

        "param_grid": {

            "C": [
                0.01,
                0.1,
                1,
                10,
                100,
            ],

            "kernel": [
                "linear",
                "rbf",
            ],

            "gamma": [
                "scale",
                "auto",
            ],
        },

        "cv": 5,

        "scoring": "accuracy",

        "n_iter": 5,

        "random_state": 42,
    },
}


# =========================================================
# Start AutoML
# =========================================================

controller.start(

    dataframe=df,

    target_column="target",

    model_names=[
        "KNN",
        "SVM",
    ],

    feature_selection="No Selection",

    hyperparameter_tuning=True,

    tuning_method="Random Search",

    tuning_params=tuning_params,

    scoring="accuracy",

    test_size=0.2,

    random_state=42,
)


# =========================================================
# Event Loop
# =========================================================

sys.exit(
    app.exec()
)