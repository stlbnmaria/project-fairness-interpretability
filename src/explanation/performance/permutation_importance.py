from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from eli5.sklearn import PermutationImportance
from pygam import LogisticGAM
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from config.config_modeling import RANDOM_STATE


def pi(
    X_train: pd.DataFrame,
    X_test: pd.Series,
    y_test: pd.Series,
    model: Union[
        RandomForestClassifier,
        LogisticRegression,
        DecisionTreeClassifier,
        LogisticGAM,
        XGBClassifier,
        MLPClassifier,
    ],
) -> Tuple[List[str], List[float], PermutationImportance]:
    """Calculate permutation importance (PI) for a given model using AUC as the scoring metric.

    Parameters
    ----------
    X_train: pd.DataFrame
        Corresponds to the dataframe with all the non target columns of the train dataset.
    X_test: pd.DataFrame
        Corresponds to the non target columns of the test dataset for which PI is calculated.
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
        Pre-trained model for which permutation importance is calculated.

    Returns
    -------
    sorted_features: List[str]
        A list of feature names sorted by their contribution to AUC.
    sorted_contributions: List[float]
        A list of percentage contributions of each feature to AUC.
    perm_importance: PermutationImportance
        The PermutationImportance object containing detailed results.
    """
    perm_importance = PermutationImportance(
        model, scoring="roc_auc", random_state=RANDOM_STATE, n_iter=30
    )
    perm_importance.fit(X_test, y_test)

    importances = perm_importance.feature_importances_
    percentage_contributions = (importances / importances.sum()) * 100

    feature_names = list(X_train.columns)

    # Sort feature importances in descending order
    sorted_indices = np.argsort(percentage_contributions)[::-1]
    sorted_features = [feature_names[i] for i in sorted_indices]
    sorted_contributions = [percentage_contributions[i] for i in sorted_indices]
    return sorted_features, sorted_contributions, perm_importance
