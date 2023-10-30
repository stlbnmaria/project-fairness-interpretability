from typing import List, Union

import pandas as pd
from pygam import LogisticGAM
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# TODO: ADD PLTR


def get_model(
    model_name: str,
) -> Union[
    RandomForestClassifier,
    LogisticRegression,
    DecisionTreeClassifier,
    LogisticGAM,
    XGBClassifier,
    MLPClassifier,
]:
    """Creates an ML model based on the model given as string.

    Parameters
    ----------
    model_name: str
            Name of the model we want to create.

    Returns
    -------
    model: Union[
            RandomForestClassifier,
            LogisticRegression,
            DecisionTreeClassifier,
            LogisticGAM,
            XGBClassifier,
            MLPClassifier,
        ]
        ML model we intended to create.
    """
    # white box models
    if model_name == "Logistic Regression":
        model = LogisticRegression()
    elif model_name == "GAM":
        model = LogisticGAM()
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier()

    # black box models
    elif model_name == "Random Forest":
        model = RandomForestClassifier()
    elif model_name == "XGB":
        model = XGBClassifier()
    elif model_name == "ANN":
        model = MLPClassifier()

    return model


def training(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Union[List[dict], dict],
) -> Union[
    RandomForestClassifier,
    LogisticRegression,
    DecisionTreeClassifier,
    LogisticGAM,
    XGBClassifier,
    MLPClassifier,
]:
    """Given the name of the ML model, creates the model, and fits it to X_train and y_train.

    Parameters
    ----------
    model_name: str
        Name of the model we want to create.
    X_train: pd.DataFrame
        Dataframe containing the training data without the target variable.
    y_train: pd.Series
        Series containing the values of the target variable for the training data.
    params: Union[List[dict], dict]
        Parameters for model.

    Returns
    -------
    model: Union[RandomForestClassifier, LogisticRegression, DecisionTreeClassifier, LogisticGAM]
        Model we wanted to create, fitted to X_train and y_train.
    """
    # get instance of model by string
    model = get_model(model_name)

    # set params
    model.set_params(**params)

    # fit the model
    model.fit(X_train, y_train)
    return model
