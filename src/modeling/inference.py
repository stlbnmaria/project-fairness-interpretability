from typing import Union

import numpy as np
import pandas as pd
from pygam import LogisticGAM
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


def inference(
    model: Union[RandomForestClassifier, LogisticRegression, DecisionTreeClassifier, LogisticGAM],
    X: pd.DataFrame,
) -> np.ndarray:
    """Given X and a model, returns an array containing the predictions.

    Parameters
    ----------
    model: Union[RandomForestClassifier, LogisticRegression, DecisionTreeClassifier, LogisticGAM]
        Model which will be used to make the predictions.
    X: pd.DataFrame
        Data for which the model will predict the target variable label or probability.

    Returns
    -------
    preds: np. ndarray
        Array containing the predicted probabilities.
    """
    preds = model.predict_proba(X)
    return preds
