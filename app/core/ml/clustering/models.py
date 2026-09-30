from sklearn.cluster import (
    AgglomerativeClustering,
    DBSCAN,
    KMeans,
)


SUPPORTED_ALGORITHMS = (
    "KMeans",
    "DBSCAN",
    "Agglomerative",
)


def build_clustering_model(
    algorithm: str,
    params: dict | None = None,
):
    """
    Build a clustering model.

    Supported algorithms:
        - KMeans
        - DBSCAN
        - Agglomerative
    """

    params = params or {}

    if algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(
            f"Unsupported clustering algorithm: {algorithm}"
        )

    if algorithm == "KMeans":
        n_clusters = params.get("n_clusters", 3)

        if n_clusters is None:
            n_clusters = 3

        return KMeans(
            n_clusters=n_clusters,
            random_state=42,
        )

    if algorithm == "DBSCAN":
        eps = params.get("eps", 0.5)
        min_samples = params.get("min_samples", 5)

        if eps is None:
            eps = 0.5

        if min_samples is None:
            min_samples = 5

        return DBSCAN(
            eps=eps,
            min_samples=min_samples,
        )

    if algorithm == "Agglomerative":
        n_clusters = params.get("n_clusters", 3)
        linkage = params.get("linkage", "ward")

        if n_clusters is None:
            n_clusters = 3

        if linkage is None:
            linkage = "ward"

        return AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage,
        )

    raise ValueError(
        f"Unsupported clustering algorithm: {algorithm}"
    )