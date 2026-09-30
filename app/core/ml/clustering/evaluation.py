from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


def evaluate_clustering(
    X,
    labels,
    algorithm: str | None = None,
    model=None,
) -> dict:
    """
    Evaluate clustering results.

    Metrics:
        - silhouette_score
        - calinski_harabasz_score
        - davies_bouldin_score
        - inertia for KMeans
        - n_clusters
    """

    unique_labels = set(labels)

    # DBSCAN noise label (-1) is not a cluster.
    if -1 in unique_labels:
        n_clusters = len(unique_labels) - 1
    else:
        n_clusters = len(unique_labels)

    metrics = {
        "n_clusters": n_clusters,
        "silhouette_score": None,
        "calinski_harabasz_score": None,
        "davies_bouldin_score": None,
    }

    # Most clustering metrics require at least
    # two different clusters.
    if n_clusters >= 2:
        try:
            metrics["silhouette_score"] = silhouette_score(
                X,
                labels,
            )
        except Exception:
            pass

        try:
            metrics["calinski_harabasz_score"] = (
                calinski_harabasz_score(
                    X,
                    labels,
                )
            )
        except Exception:
            pass

        try:
            metrics["davies_bouldin_score"] = (
                davies_bouldin_score(
                    X,
                    labels,
                )
            )
        except Exception:
            pass

    # KMeans-specific metric.
    if algorithm == "KMeans" and model is not None:
        metrics["inertia"] = model.inertia_

    return metrics