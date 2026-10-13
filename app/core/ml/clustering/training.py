from sklearn.decomposition import PCA

from .evaluation import evaluate_clustering
from .models import build_clustering_model


def train_clustering(
    dataframe,
    algorithm: str,
    params: dict | None = None,
    use_dim_reduction: bool = False,
):
    """
    Train a clustering model.

    Pipeline:

        DataFrame
            ↓
        optional PCA
            ↓
        clustering model
            ↓
        labels
            ↓
        evaluation
    """

    if dataframe is None:
        raise ValueError("Dataset cannot be None.")

    if dataframe.empty:
        raise ValueError("Dataset is empty.")

    params = params or {}

    X = dataframe.values

    # Same behavior as the original project:
    # PCA is applied before clustering.
    if use_dim_reduction:
        pca = PCA(n_components=2)
        X = pca.fit_transform(X)

    model = build_clustering_model(
        algorithm=algorithm,
        params=params,
    )

    labels = model.fit_predict(X)

    metrics = evaluate_clustering(
        X=X,
        labels=labels,
        algorithm=algorithm,
        model=model,
    )

    model_info = {
        "algorithm": algorithm,
        **params,
    }

    return {
        "model": model,
        "labels": labels,
        "data": X,
        "metrics": metrics,
        "model_info": model_info,
    }
