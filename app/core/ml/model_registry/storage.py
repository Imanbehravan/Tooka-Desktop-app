import json
import shutil
from pathlib import Path

from platformdirs import user_data_dir

class ModelStorage:
    """
    Handles persistent storage for Tooka model metadata
    and serialized model files.
    """


    APP_NAME = "Tooka Desktop"
    APP_AUTHOR = "Tooka AI"

    def __init__(self):
        self.base_dir = Path(
            user_data_dir(
                self.APP_NAME,
                self.APP_AUTHOR
            )
        )

        self.models_dir = self.base_dir / "models"

        self.registry_file = (
            self.models_dir / "registry.json"
        )

        self.files_dir = (
            self.models_dir / "files"
        )

        self._ensure_directories()

    # =============================================================
    # Directory Management
    # =============================================================

    def _ensure_directories(self):
        """
        Create required directories if they don't exist.
        """

        self.models_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.files_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        if not self.registry_file.exists():
            self._write_registry([])

    # =============================================================
    # Registry Read
    # =============================================================

    def read_registry(self) -> list[dict]:
        """
        Read model metadata from registry.json.
        """

        if not self.registry_file.exists():
            return []

        try:
            with self.registry_file.open(
                "r",
                encoding="utf-8"
            ) as file:

                data = json.load(file)

            if not isinstance(data, list):
                raise ValueError(
                    "Registry data must be a list."
                )

            return data

        except (
            json.JSONDecodeError,
            OSError,
            ValueError
        ):
            return []

    # =============================================================
    # Registry Write
    # =============================================================

    def write_registry(
        self,
        models: list[dict]
    ):
        """
        Persist model metadata to registry.json.
        """

        self._write_registry(models)

    def _write_registry(
        self,
        models: list[dict]
    ):
        """
        Internal registry writer using an atomic replacement.
        """

        self.registry_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        temporary_file = (
            self.registry_file.with_suffix(".tmp")
        )

        with temporary_file.open(
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                models,
                file,
                indent=4,
                ensure_ascii=False
            )

        temporary_file.replace(
            self.registry_file
        )

    # =============================================================
    # Model File Path
    # =============================================================

    def get_model_file_path(
        self,
        model_name: str,
        version: str
    ) -> Path:
        """
        Return the filesystem path for a model file.
        """

        safe_name = self._sanitize_name(
            model_name
        )

        version_dir = (
            self.files_dir /
            safe_name /
            version
        )

        version_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        return version_dir / "model.joblib"

    # =============================================================
    # Model Directory
    # =============================================================

    def get_model_directory(
        self,
        model_name: str
    ) -> Path:
        """
        Return the directory containing all versions
        of a model.
        """

        safe_name = self._sanitize_name(
            model_name
        )

        return self.files_dir / safe_name

    # =============================================================
    # Rename Model Files
    # =============================================================

    def rename_model_files(
        self,
        old_name: str,
        new_name: str
    ):
        """
        Rename the filesystem directory of a model.

        If the old model directory does not exist,
        no filesystem operation is required.
        """

        old_dir = self.get_model_directory(
            old_name
        )

        new_dir = self.get_model_directory(
            new_name
        )

        if not old_dir.exists():
            return

        if new_dir.exists():
            raise FileExistsError(
                f"Model directory already exists: {new_dir}"
            )

        old_dir.rename(
            new_dir
        )

    # =============================================================
    # Delete Model Files
    # =============================================================

    def delete_model_files(
        self,
        model_name: str,
        version: str | None = None
    ):
        """
        Delete model files.

        If version is provided, only that version is removed.
        Otherwise, all versions are removed.
        """

        model_dir = self.get_model_directory(
            model_name
        )

        if not model_dir.exists():
            return

        if version is None:

            shutil.rmtree(
                model_dir
            )

            return

        version_dir = (
            model_dir / version
        )

        if version_dir.exists():
            shutil.rmtree(
                version_dir
            )

        try:
            model_dir.rmdir()
        except OSError:
            pass

    # =============================================================
    # Helpers
    # =============================================================

    @staticmethod
    def _sanitize_name(
        name: str
    ) -> str:
        """
        Make a safe directory name from a model name.
        """

        invalid_chars = '<>:"/\\|?*'

        sanitized = "".join(
            "_"
            if char in invalid_chars
            else char
            for char in name
        )

        sanitized = sanitized.strip()

        return sanitized or "unnamed_model"
