from typing import Tuple, Union

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
) -> Tuple[float, tuple]:
    """Givesthe performance metric score as well as the XPER values for each feature.

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
    XPER = ModelPerformance(X_train, y_train, X_test, y_test, model)
    PM = XPER.evaluate(["AUC"])
    PM_round = round(PM, 3)
    XPER_values = XPER.calculate_XPER_values(["AUC"])
    return PM_round, XPER_values
