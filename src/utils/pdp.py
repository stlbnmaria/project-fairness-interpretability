import matplotlib.pyplot as plt
from typing import List, Union, Tuple
from pygam import LogisticGAM
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

def get_underscore(cols : List) -> List[List]:
    """Allows us to separate the columns which have an _ in their name.

    Parameters
    ----------
    cols: list
        List of columns we want to separate.

    Returns
    -------
    underscore_columns: list
        List of columns with a _.
    non_underscore_columns: list
        List of columns without a _.
    """
    underscore_columns = []
    non_underscore_columns = []
    for col in cols:
        if "_" in col:
            underscore_columns +=[col]
        else:
            non_underscore_columns +=[col]
    return [underscore_columns, non_underscore_columns]

def ohe_filter(non_underscore_cols: List, cols: List) -> List[List]:
    """ Given columns with and without a _, returns a list of the columns which were and were not one hot encoded.

    Parameters
    ----------
    non_underscore_columns: list
        List of columns without a _.
    underscore_columns: list
        List of columns with a _.

    Returns
    -------
    non_ohe: list
        List of columns which were not one hot encoded.
    ohe: list
        List of columns which were one hot encoded.
    """
    prefix_dict = dict()
    for col in cols:
        prefix = col.split("_")[0]
        if prefix in prefix_dict:
            prefix_dict[prefix] += [col]
        else:
            prefix_dict[prefix] = [col]
    non_ohe = [[col] for col in non_underscore_cols]
    ohe = []
    for key, value in prefix_dict.items():
        if len(value) > 1:
            ohe += [value]
        else:
            non_ohe += [value]
    return [non_ohe, ohe]

def categorical_partial_dependence(model: Union[
        RandomForestClassifier, LogisticRegression, DecisionTreeClassifier, LogisticGAM
    ], X: pd.DataFrame, feature_names: List, figure_size: (int, int), y_label: str = "Partial Dependence"):
    """Given a trained model, a dataframe and feature names, plots a matplotlib partial dependence plot.

    Parameters
    ----------
    model: Union[RandomForestClassifier, LogisticRegression, DecisionTreeClassifier, LogisticGAM]
        Pre-trained model.
    X: pd.DataFrame
        Dataframe for which we want the Partial Dependence plot.
    figure_size: Tuple(int)
        Size of the partial dependence plot.
    y_label:
        Label of the y axis of the partial dependence plot.

    Returns
    -------
    None
    """
    average_preds = []
    for element in feature_names:
        X_copy = X.copy()
        X_copy[element] = 1
        for second_element in feature_names:
            if element != second_element:
                X_copy[second_element] = 0
        probs = model.predict_proba(X_copy)
        mean = probs[:,1].mean()
        average_preds += [mean]
    fig = plt.figure(figsize = figure_size)
    plt.bar(feature_names, average_preds)
    plt.ylabel(y_label)
    plt.xticks(rotation="vertical")
    plt.show()