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
    predict_proba: bool = True,
) -> np.ndarray:
    """Given X and a model, returns an array containing the predictions.

    Parameters
    ----------
    model: Union[RandomForestClassifier, LogisticRegression, DecisionTreeClassifier, LogisticGAM]
        Model which will be used to make the predictions.
    X: pd.DataFrame
        Data for which the model will predict the target variable label or probability.
    predict_proba: bool
        Boolean which represents whether the output should be the predicted labels or probabilities.

    Returns
    -------
    preds: np. ndarray
        Array containing the predicted labels/probabilities.
    """
    if predict_proba:
        preds = model.predict_proba(X)
    else:
        preds = model.predict(X)
    return preds
