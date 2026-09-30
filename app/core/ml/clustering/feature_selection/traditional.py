from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.cluster import FeatureAgglomeration


class NoFeatureSelection:
    def __init__(self, data):
        self.data = data

    def select_features(self):
        return self.data


class VarianceThresholdFeatureSelection:
    def __init__(self, data, threshold=0.0):
        self.data = data
        self.threshold = threshold
        self.selected_features = None

    def select_features(self):
        selector = VarianceThreshold(
            threshold=self.threshold
        )

        transformed_data = selector.fit_transform(self.data)

        self.selected_features = selector.get_support(
            indices=True
        )

        return transformed_data


class PCAFeatureSelection:
    def __init__(self, data, n_components=2):
        self.data = data
        self.n_components = n_components

    def select_features(self):
        pca = PCA(
            n_components=self.n_components
        )

        reduced_data = pca.fit_transform(self.data)

        return reduced_data


class FeatureAgglomerationSelection:
    def __init__(self, data, n_clusters=2):
        self.data = data
        self.n_clusters = n_clusters

    def select_features(self):
        agglo = FeatureAgglomeration(
            n_clusters=self.n_clusters
        )

        reduced_data = agglo.fit_transform(self.data)

        return reduced_data