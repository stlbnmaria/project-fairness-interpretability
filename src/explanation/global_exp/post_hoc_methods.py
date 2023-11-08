from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
from PyALE import ale
from pygam import LogisticGAM
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.modeling.create_data_split import split_data


def get_underscore(cols: List[str]) -> Tuple[List[str]]:
    """Allows us to separate the columns which have an _ in their name.

    Parameters
    ----------
    cols: List[str]
        List of columns we want to separate.

    Returns
    -------
    underscore_columns: List[str]
        List of columns with a _.
    non_underscore_columns: List[str]
        List of columns without a _.
    """
    # initializing lists of columns with and without _
    underscore_columns = []
    non_underscore_columns = []
    # checking for each column if they have a _ and assigning them to their respective list
    for col in cols:
        if "_" in col:
            underscore_columns += [col]
        else:
            non_underscore_columns += [col]
    return underscore_columns, non_underscore_columns


def ohe_filter(non_underscore_cols: List[str], cols: List[str]) -> Tuple[List[str]]:
    """Given columns with/without a _, returns list of columns which were/were not one hot encoded.

    Parameters
    ----------
    non_underscore_columns: List[str]
        List of columns without a _.
    underscore_columns: List[str]
        List of columns with a _.

    Returns
    -------
    non_ohe: List[str]
        List of columns which were not one hot encoded.
    ohe: List[str]
        List of columns which were one hot encoded.
    prefix_ohe: List[str]
        List of prefixes of one hot encoded columns.
    """
    # starting a dict with as keys the prefixes and as values the columns with those prefixes.
    prefix_dict = dict()
    # finding the prefixes for each column and adding them to the dictionary.
    for col in cols:
        prefix = col.split("_")[0]
        if prefix in prefix_dict:
            prefix_dict[prefix] += [col]
        else:
            prefix_dict[prefix] = [col]
    # adding all columns without underscore to non_ohe columns list.
    non_ohe = [[col] for col in non_underscore_cols]
    ohe = []
    prefix_ohe = []
    # checking for each prefix if there is more than one column
    # if so, OHE happened and so we assign them to the ohe list, and the prefix to the prefix list.
    for key, value in prefix_dict.items():
        if len(value) > 1:
            ohe += [value]
            prefix_ohe += [key]
        else:
            non_ohe += [value]
    return non_ohe, ohe, prefix_ohe


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


def ohe_ale(
    col: str,
    cat_cols: List[str],
    model_cols: List[str],
    df: pd.DataFrame,
    model: Union[
        RandomForestClassifier,
        LogisticRegression,
        DecisionTreeClassifier,
        LogisticGAM,
        XGBClassifier,
        MLPClassifier,
    ],
    train_size: float,
    test_size: float,
    figure_size: Tuple[int, int],
    random_state: int = 42,
) -> plt.figure:
    """Function which allows us to plot the ALE for one hot encoded variables.

    Parameters
    ----------
    col: str
        Column for which we want to plot the ALE.
    cat_cols: List[str]
        List of columns to be dummy encoded.
    model_cols: List[str]
        List of columns in the dataset used for training the model.
    df: pd.DataFrame
        Dataframe which will be split.
    model: Union[
            RandomForestClassifier,
            LogisticRegression,
            DecisionTreeClassifier,
            LogisticGAM,
            XGBClassifier,
            MLPClassifier,
        ]
        Model used for the ALE plot.
    train_size: float
        Percentage of whole data used for training.
    test_size: float
        Percentage for testing from val-test data.
    figure_size: Tuple[int, int]
        Size of the plot we want to create.
    random_state: int
        Random state of split for reproducibility.

    Returns
    -------
    ale_eff: plt.figure
        ALE plot.
    """
    # Finding the columns to be encoded first(we remove the column
    #  we want to plot due to package restrictions).
    cols_ohe = cat_cols.copy()
    cols_ohe.remove(col)
    # creating the dataset with the one hot encoding done(with the column
    #  we want to plot not one hot encoded).
    step_data = split_data(
        cols_ohe, df=df, train_size=train_size, test_size=test_size, random_state=random_state
    )
    # creating the ALE plot
    fig, ax = plt.subplots(figsize=figure_size)
    ale_eff = ale(
        X=step_data["test"][0],
        model=model,
        feature=[col],
        include_CI=False,
        encode_fun=ale_encoder,
        predictors=model_cols,
        fig=fig,
        ax=ax,
    )
    return ale_eff
