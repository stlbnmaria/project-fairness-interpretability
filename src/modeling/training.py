from typing import Union

import pandas as pd
from pygam import LogisticGAM
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def get_model(
    model_name: str = None,
) -> Union[RandomForestClassifier, LogisticRegression, DecisionTreeClassifier, LogisticGAM]:
    """Creates an ML model based on the model given as string.

    Parameters
    ----------
    model_name: str
            Name of the model we want to create.

    Returns
    -------
    model: Union[RandomForestClassifier, LogisticRegression, DecisionTreeClassifier, LogisticGAM]
        ML model we intended to create.
    """
    if model_name == "Logistic Regression":
        model = LogisticRegression()
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier()
    elif model_name == "Random Forest":
        model = RandomForestClassifier()
    elif model_name == "GAM":
        model = LogisticGAM()

    return model


def training(
    model_name: str = None, X_train: pd.DataFrame = None, y_train: pd.Series = None
) -> Union[RandomForestClassifier, LogisticRegression, DecisionTreeClassifier, LogisticGAM]:
    """Given the name of the ML model, creates the model, and fits it to X_train and y_train.

    Parameters
    ----------
    model_name: str
        Name of the model we want to create.
    X_train: pd.DataFrame
        Dataframe containing the training data without the target variable.
    y_train: pd.Series
        Series containing the values of the target variable for the training data.

    Returns
    -------
    model: Union[RandomForestClassifier, LogisticRegression, DecisionTreeClassifier, LogisticGAM]
        Model we wanted to create, fitted to X_train and y_train.
    """
    model = get_model(model_name)
    model.fit(X_train, y_train)
    return model
