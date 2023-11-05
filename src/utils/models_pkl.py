import pickle
from pathlib import Path
from typing import Union

from pygam import LogisticGAM
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def save_pickle(
    path: Path,
    obj: Union[
        RandomForestClassifier,
        LogisticRegression,
        DecisionTreeClassifier,
        LogisticGAM,
        XGBClassifier,
        MLPClassifier,
    ],
) -> None:
    """Given a path and an object, stores the object as a pickle file in the specified path.

    Parameters
    ----------
    path : Path
           Represents the path where the pickle object will be stored.

    obj : Union[
            RandomForestClassifier,
            LogisticRegression,
            DecisionTreeClassifier,
            LogisticGAM,
            XGBClassifier,
            MLPClassifier,
        ]
          Represents the model that will be stored.

    Returns
    -------
    None : None
    """
    with open(path, "wb") as f:
        pickle.dump(obj, f)
