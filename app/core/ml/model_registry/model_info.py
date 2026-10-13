from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

@dataclass
class ModelInfo:
    """
    Metadata describing a trained machine learning model.
    """


    name: str
    model_type: str
    algorithm: str

    metric_name: str
    metric_value: float

    version: str = "v1"
    status: str = "Ready"

    model_path: Optional[str] = None

    created_at: str = ""

    description: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def to_dict(self) -> dict:
        """
        Convert ModelInfo to a JSON-serializable dictionary.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelInfo":
        """
        Create ModelInfo from a dictionary.
        """
        return cls(
            name=data["name"],
            model_type=data["model_type"],
            algorithm=data["algorithm"],
            metric_name=data["metric_name"],
            metric_value=float(data["metric_value"]),
            version=data.get("version", "v1"),
            status=data.get("status", "Ready"),
            model_path=data.get("model_path"),
            created_at=data.get("created_at", ""),
            description=data.get("description", ""),
        )
