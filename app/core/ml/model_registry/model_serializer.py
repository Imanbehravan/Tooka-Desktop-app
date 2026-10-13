from pathlib import Path

import joblib

class ModelSerializer:
    """
    Handles saving and loading trained machine learning models.
    """


    @staticmethod
    def save(
        model,
        path: str | Path
    ) -> Path:
        """
        Serialize a trained model to disk.
        """

        model_path = Path(path)

        model_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        joblib.dump(
            model,
            model_path
        )

        return model_path

    @staticmethod
    def load(
        path: str | Path
    ):
        """
        Load a serialized model from disk.
        """

        model_path = Path(path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        return joblib.load(
            model_path
        )
