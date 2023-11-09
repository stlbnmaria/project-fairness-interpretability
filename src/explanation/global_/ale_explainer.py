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
