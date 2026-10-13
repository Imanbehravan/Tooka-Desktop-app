from sklearn.metrics import silhouette_score


def evaluate_clustering(
    X,
    labels,
    algorithm: str | None = None,
    model=None,
) -> dict:
    """
    Evaluate clustering results.

    Metrics:
        - inertia for KMeans
        - silhouette_score
        - n_clusters for DBSCAN
        - n_clusters for Agglomerative
    """

    # ---------------------------------------------------------
    # KMeans
    # ---------------------------------------------------------

    if algorithm == "KMeans":

        metric_value = (
            silhouette_score(X, labels)
            if len(set(labels)) > 1
            else None
        )

        metrics = {
            "inertia": model.inertia_,
            "silhouette_score": metric_value,
        }

        return metrics

    # ---------------------------------------------------------
    # DBSCAN
    # ---------------------------------------------------------

    if algorithm == "DBSCAN":

        n_clusters = (
            len(set(labels))
            - (1 if -1 in labels else 0)
        )

        metric_value = (
            silhouette_score(X, labels)
            if n_clusters > 1
            else None
        )

        metrics = {
            "n_clusters": n_clusters,
            "silhouette_score": metric_value,
        }

        return metrics

    # ---------------------------------------------------------
    # Agglomerative
    # ---------------------------------------------------------

    if algorithm == "Agglomerative":

        metric_value = (
            silhouette_score(X, labels)
            if len(set(labels)) > 1
            else None
        )

        metrics = {
            "n_clusters": len(set(labels)),
            "silhouette_score": metric_value,
        }

        return metrics

    # ---------------------------------------------------------
    # Unsupported algorithm
    # ---------------------------------------------------------

    raise ValueError(
        f"Unsupported algorithm: {algorithm}"
    )
