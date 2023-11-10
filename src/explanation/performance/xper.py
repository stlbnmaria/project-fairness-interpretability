from typing import Tuple, Union

import numpy as np
import pandas as pd
from pygam import LogisticGAM
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from XPER.compute.Performance import ModelPerformance


def xper_values(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model: Union[
        RandomForestClassifier,
        LogisticRegression,
        DecisionTreeClassifier,
        LogisticGAM,
        XGBClassifier,
        MLPClassifier,
    ],
) -> Tuple[np.ndarray, np.ndarray]:
    """Gives the performance metric score as well as the XPER values for each feature.

    Parameters
    ----------
    X_train: pd.DataFrame
        Corresponds to the dataframe with all the non target columns of the train dataset.
    y_train: pd.Series
        Corresponds to the target column of the train dataset.
    X_test: pd.DataFrame
        Corresponds to the dataframe with all the non target columns of the test dataset.
    y_test: pd.Series
        Corresponds to the target column of the test dataset.
    model: Union[
            RandomForestClassifier,
            LogisticRegression,
            DecisionTreeClassifier,
            LogisticGAM,
            XGBClassifier,
            MLPClassifier,
        ]
        Pre-trained model.

    Returns
    -------
    PM_round: float
        The performance metric value rounded to 3 decimal places.
    XPER_values: tuple
        The XPER values for each feature as well as XPER value for each feature and individual
    """
    XPER = ModelPerformance(X_train, y_train, X_test, y_test, model, sample_size=200)
    PM = XPER.evaluate(["AUC"])
    PM_round = round(PM, 3)
    XPER_values = XPER.calculate_XPER_values(["AUC"], N_coalition_sampled=1000)
    return PM_round, XPER_values


def xper_contribution(
    XPER_values: Tuple[np.ndarray, np.ndarray], X_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    """Gives the percentage contribution to AUC for each feature and the labels.

    Parameters
    ----------
    XPER_values: Tuple[np.ndarray, np.ndarray]
        Corresponds to the XPER values.
    X_test: pd.DataFrame
        Corresponds to the dataframe with all the non target columns of the test dataset.

    Returns
    -------
    final_selection_values: np.ndarray
        Array with the percentage contribution to AUC for each feature.
    final_selection_labels: np.ndarray
        Array with all the labels.
    """
    Contribution = XPER_values[0].copy()

    ecart_contrib = sum(Contribution) - Contribution[0]

    Contribution = (Contribution / ecart_contrib) * 100
    selection = pd.DataFrame(Contribution[1:], index=X_test.columns, columns=["Metric"])
    labels = list(X_test.columns)
    selection["Labels"] = labels

    selection["absolute"] = abs(selection["Metric"])

    testons = selection.sort_values(by="absolute", ascending=False)[:42].index

    final_selection_values = selection.loc[testons, "Metric"].to_numpy()
    final_selection_labels = selection.loc[testons, "Labels"]
    return final_selection_values, final_selection_labels


def absolute_contribution(
    PM: float, final_selection_values: np.ndarray, final_selection_labels: np.ndarray
) -> dict:
    """Gives a dictionary with the index as keys and absolute contributions to AUC as values.

    Parameters
    ----------
    PM: float
        Value of the performance metric
    final_selection_values: np.ndarray
        Array with the percentage contribution to AUC for each feature.
    final_selection_labels: np.ndarray
        Array with all the labels.

    Returns
    -------
    contribution_dict: dict
        dictionary with the index as keys and absolute contribution as values.
    """
    contribution_dict = {}
    benchmark = 0.5
    over_performance = PM - benchmark
    for percentage, index in zip(final_selection_values, final_selection_labels):
        contribution = (percentage / 100) * over_performance
        contribution_dict[index] = contribution
    return contribution_dict
