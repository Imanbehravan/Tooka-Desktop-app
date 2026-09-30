import numpy as np

from sklearn.cluster import (
    AgglomerativeClustering,
    DBSCAN,
    KMeans,
)
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


def tune_kmeans(
    X,
    cluster_range=(2, 10),
    metric="silhouette",
):
    best_score = (
        -np.inf
        if metric != "davies"
        else np.inf
    )

    best_n_clusters = None
    scores = []

    for n_clusters in range(
        cluster_range[0],
        cluster_range[1] + 1,
    ):
        kmeans = KMeans(
            n_clusters=n_clusters,
            n_init=10,
            random_state=0,
        )

        cluster_labels = kmeans.fit_predict(X)

        if len(set(cluster_labels)) < 2:
            continue

        if metric == "silhouette":
            score = silhouette_score(
                X,
                cluster_labels,
            )

        elif metric == "calinski":
            score = calinski_harabasz_score(
                X,
                cluster_labels,
            )

        elif metric == "davies":
            score = -davies_bouldin_score(
                X,
                cluster_labels,
            )

        else:
            raise ValueError(
                f"Unsupported metric: {metric}"
            )

        scores.append(
            (n_clusters, score)
        )

        if (
            metric != "davies"
            and score > best_score
        ) or (
            metric == "davies"
            and score < best_score
        ):
            best_score = score
            best_n_clusters = n_clusters

    return (
        best_n_clusters,
        best_score,
        scores,
    )


def tune_dbscan(
    X,
    eps_values=None,
    min_samples_values=None,
    metric="silhouette",
):
    if eps_values is None:
        eps_values = np.linspace(
            0.1,
            3,
            20,
        )

    if min_samples_values is None:
        min_samples_values = [3, 4, 5, 6]

    best_score = -np.inf
    best_params = {}
    details = []

    for eps in eps_values:
        for min_samples in min_samples_values:

            dbscan = DBSCAN(
                eps=eps,
                min_samples=min_samples,
            )

            labels = dbscan.fit_predict(X)

            n_clusters = (
                len(set(labels))
                - (1 if -1 in labels else 0)
            )

            if n_clusters < 2:
                continue

            try:
                score = silhouette_score(
                    X,
                    labels,
                )

                details.append(
                    (
                        eps,
                        min_samples,
                        n_clusters,
                        score,
                    )
                )

                if score > best_score:
                    best_score = score

                    best_params = {
                        "eps": eps,
                        "min_samples": min_samples,
                        "n_clusters": n_clusters,
                    }

            except Exception:
                continue

    return (
        best_params,
        best_score,
        details,
    )


def tune_agglomerative(
    X,
    cluster_range=(2, 10),
    linkage_options=None,
    metric="silhouette",
):
    if linkage_options is None:
        linkage_options = [
            "ward",
            "complete",
            "average",
            "single",
        ]

    best_score = -np.inf
    best_params = {}
    details = []

    for n_clusters in range(
        cluster_range[0],
        cluster_range[1] + 1,
    ):
        for linkage in linkage_options:

            if (
                linkage == "ward"
                and X.shape[1] < 2
            ):
                continue

            model = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=linkage,
            )

            labels = model.fit_predict(X)

            if len(set(labels)) < 2:
                continue

            try:
                score = silhouette_score(
                    X,
                    labels,
                )

                details.append(
                    (
                        n_clusters,
                        linkage,
                        score,
                    )
                )

                if score > best_score:
                    best_score = score

                    best_params = {
                        "n_clusters": n_clusters,
                        "linkage": linkage,
                    }

            except Exception:
                continue

    return (
        best_params,
        best_score,
        details,
    )