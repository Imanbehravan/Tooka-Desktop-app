from pathlib import Path
from typing import Optional

from .model_info import ModelInfo
from .storage import ModelStorage

class ModelRegistry:
    """
    Central registry for managing trained Tooka models.
    """


    def __init__(
        self,
        storage: Optional[ModelStorage] = None
    ):
        self.storage = storage or ModelStorage()

    # =============================================================
    # List
    # =============================================================

    def list_models(self) -> list[ModelInfo]:
        """
        Return all registered models.
        """

        data = self.storage.read_registry()

        return [
            ModelInfo.from_dict(item)
            for item in data
        ]

    # =============================================================
    # Get
    # =============================================================

    def get_model(
        self,
        name: str,
        version: Optional[str] = None
    ) -> Optional[ModelInfo]:

        models = self.list_models()

        for model in models:

            if model.name != name:
                continue

            if version is not None:
                if model.version != version:
                    continue

            return model

        return None

    # =============================================================
    # Register
    # =============================================================

    def register_model(
        self,
        model: ModelInfo
    ) -> ModelInfo:

        models = self.list_models()

        existing = self.get_model(
            model.name,
            model.version
        )

        if existing is not None:
            raise ValueError(
                f"Model '{model.name}' "
                f"version '{model.version}' "
                f"already exists."
            )

        if model.model_path is None:
            model.model_path = str(
                self.storage.get_model_file_path(
                    model.name,
                    model.version
                )
            )

        models.append(model)

        self._save_models(models)

        return model

    # =============================================================
    # Update
    # =============================================================

    def update_model(
        self,
        model: ModelInfo
    ) -> ModelInfo:

        models = self.list_models()

        found = False

        for index, existing in enumerate(models):

            if (
                existing.name == model.name
                and existing.version == model.version
            ):
                models[index] = model
                found = True
                break

        if not found:
            raise ValueError(
                f"Model '{model.name}' "
                f"version '{model.version}' "
                f"was not found."
            )

        self._save_models(models)

        return model

    # =============================================================
    # Rename
    # =============================================================

    def rename_model(
        self,
        name: str,
        new_name: str
    ) -> ModelInfo:

        new_name = new_name.strip()

        if not new_name:
            raise ValueError(
                "New model name cannot be empty."
            )

        if name == new_name:
            return self.get_model(name)

        models = self.list_models()

        matching_models = [
            model
            for model in models
            if model.name == name
        ]

        if not matching_models:
            raise ValueError(
                f"Model '{name}' was not found."
            )

        if any(
            model.name == new_name
            for model in models
        ):
            raise ValueError(
                f"Model '{new_name}' already exists."
            )

        # Rename filesystem directory first.
        self.storage.rename_model_files(
            name,
            new_name
        )

        for model in matching_models:

            model.name = new_name

            model.model_path = str(
                self.storage.get_model_file_path(
                    new_name,
                    model.version
                )
            )

        self._save_models(models)

        return matching_models[0]

    # =============================================================
    # Delete
    # =============================================================

    def delete_model(
        self,
        name: str,
        version: Optional[str] = None
    ):

        models = self.list_models()

        original_count = len(models)

        if version is None:

            models = [
                model
                for model in models
                if model.name != name
            ]

            if len(models) == original_count:
                raise ValueError(
                    "Model was not found."
                )

            self.storage.delete_model_files(
                name
            )

        else:

            models = [
                model
                for model in models
                if not (
                    model.name == name
                    and model.version == version
                )
            ]

            if len(models) == original_count:
                raise ValueError(
                    "Model was not found."
                )

            self.storage.delete_model_files(
                name,
                version
            )

        self._save_models(models)

    # =============================================================
    # Create Version
    # =============================================================

    def create_version(
        self,
        name: str
    ) -> ModelInfo:

        models = self.list_models()

        matching_models = [
            model
            for model in models
            if model.name == name
        ]

        if not matching_models:
            raise ValueError(
                f"Model '{name}' was not found."
            )

        latest_model = max(
            matching_models,
            key=self._version_number
        )

        next_version_number = (
            self._version_number(latest_model) + 1
        )

        new_version = ModelInfo(
            name=latest_model.name,
            model_type=latest_model.model_type,
            algorithm=latest_model.algorithm,
            metric_name=latest_model.metric_name,
            metric_value=latest_model.metric_value,
            version=f"v{next_version_number}",
            status="Ready",
            description=latest_model.description,
        )

        new_version.model_path = str(
            self.storage.get_model_file_path(
                new_version.name,
                new_version.version
            )
        )

        models.append(new_version)

        self._save_models(models)

        return new_version

    # =============================================================
    # Load Model
    # =============================================================

    def load_model(
        self,
        name: str,
        version: Optional[str] = None
    ):

        model_info = self.get_model(
            name,
            version
        )

        if model_info is None:
            raise ValueError(
                f"Model '{name}' was not found."
            )

        if not model_info.model_path:
            raise FileNotFoundError(
                "Model file path is not registered."
            )

        model_path = Path(
            model_info.model_path
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        import joblib

        return joblib.load(
            model_path
        )

    # =============================================================
    # Internal Save
    # =============================================================

    def _save_models(
        self,
        models: list[ModelInfo]
    ):

        self.storage.write_registry(
            [
                model.to_dict()
                for model in models
            ]
        )

    # =============================================================
    # Version Helper
    # =============================================================

    @staticmethod
    def _version_number(
        model: ModelInfo
    ) -> int:

        version = (
            model.version
            .lower()
            .strip()
        )

        if version.startswith("v"):
            version = version[1:]

        try:
            return int(version)
        except ValueError:
            return 0
