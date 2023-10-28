from typing import Union

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# TODO: integrate tracking with mlflow


def get_model(
    model_name: str,
) -> Union[RandomForestClassifier, LogisticRegression, DecisionTreeClassifier]:
    if model_name == "Logistic Regression":
        model = LogisticRegression()
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier()
    elif model_name == "Random Forest":
        model = RandomForestClassifier()

    return model


def training() -> None:
    pass
