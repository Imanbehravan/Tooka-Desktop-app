from pathlib import Path

import pandas as pd


def load_dataset(path: str | Path) -> pd.DataFrame:
    """
    Load a CSV dataset for clustering.

    Parameters
    ----------
    path:
        Path to CSV file.

    Returns
    -------
    pandas.DataFrame
        Loaded dataset.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if not path.is_file():
        raise ValueError(f"Dataset path is not a file: {path}")

    if path.suffix.lower() != ".csv":
        raise ValueError("Only CSV datasets are supported.")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Dataset is empty.")

    return df