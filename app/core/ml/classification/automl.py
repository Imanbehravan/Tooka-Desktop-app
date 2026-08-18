from .training import train_classification
from .models import SUPPORTED_MODELS


def run_automl(
    X_train,
    y_train,
    X_test,
    y_test,
):
    """
    Train all supported classification models
    and return their results.
    """

    results = []

    for model_name in SUPPORTED_MODELS:

        try:
            result = train_classification(
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
            )

            results.append(result)

        except Exception as exc:
            results.append(
                {
                    "model_name": model_name,
                    "error": str(exc),
                }
            )

    successful_results = [
        result
        for result in results
        if "metrics" in result
    ]

    if not successful_results:
        raise RuntimeError(
            "All classification models failed."
        )

    best_result = max(
        successful_results,
        key=lambda result: result["metrics"]["accuracy"],
    )

    return {
        "results": results,
        "best_result": best_result,
    }