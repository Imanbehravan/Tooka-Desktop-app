from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


SUPPORTED_MODELS = {
    "KNN",
    "SVM",
    "MLP",
    "Random Forest",
    "Naive Bayes",
    "Decision Tree",
}


def build_model(model_name: str, **params):
    """
    Create a classification model.
    """

    if model_name == "KNN":
        return KNeighborsClassifier(**params)

    if model_name == "SVM":
        return SVC(**params)

    if model_name == "MLP":
        return MLPClassifier(**params)

    if model_name == "Random Forest":
        return RandomForestClassifier(**params)

    if model_name == "Naive Bayes":
        return GaussianNB(**params)

    if model_name == "Decision Tree":
        return DecisionTreeClassifier(**params)

    raise ValueError(
        f"Unsupported classification model: {model_name}"
    )