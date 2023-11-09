from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
from pygam import LogisticGAM
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def categorical_partial_dependence(
    model: Union[
        RandomForestClassifier,
        LogisticRegression,
        DecisionTreeClassifier,
        LogisticGAM,
        XGBClassifier,
        MLPClassifier,
    ],
    X: pd.DataFrame,
    feature_names: List[str],
    figure_size: Tuple[int, int],
    y_label: str = "Partial Dependence",
) -> plt.figure:
    """Given a trained model, a dataframe and feature names, plots a partial dependence plot.

    Parameters
    ----------
    model: Union[
            RandomForestClassifier,
            LogisticRegression,
            DecisionTreeClassifier,
            LogisticGAM,
            XGBClassifier,
            MLPClassifier,
        ]
        Pre-trained model.
    X: pd.DataFrame
        Dataframe for which we want the Partial Dependence plot.
    feature_names: List[str]
        List of features of a category.
    figure_size: Tuple[int, int]
        Size of the partial dependence plot.
    y_label: str
        Label of the y axis of the partial dependence plot.

    Returns
    -------
    fig : plt.figure
        Partial dependence plot.

    """
    # initializing list of predictions for the PDP plot.
    average_preds = []
    # for each one hot encoded variable, compute the mean prediction
    #  as if all data points belonged to that class and adding it to the average preds.
    for element in feature_names:
        X_copy = X.copy()
        X_copy[element] = 1
        for second_element in feature_names:
            if element != second_element:
                X_copy[second_element] = 0
        probs = model.predict_proba(X_copy)
        mean = probs[:, 1].mean()
        average_preds += [mean]
    # creating the plot.
    fig = plt.figure(figsize=figure_size)
    plt.bar(feature_names, average_preds)
    plt.ylabel(y_label)
    plt.xticks(rotation="vertical")
    return fig


def ale_encoder(col: pd.Series) -> pd.DataFrame:
    """Given a pandas series, it one hot encodes it, returning a dataframe.

    Parameters
    ----------
    col: pd.Series
        Series to be one hot encoded.

    Returns
    -------
    feat_coded: pd.DataFrame
        Dataframe of the one hot encoded columns.
    """
    # Given the panda series, encoding it to obtain a dataframe.
    feat_coded = pd.get_dummies(col, dtype=int)
    return feat_coded
