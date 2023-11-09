from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pygam import LogisticGAM
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.fairness.statistical_parity import _get_group_value
from src.modeling.create_data_split import split_data


def get_chi2_statistic(
    predictions: np.ndarray or pd.Series,
    positive_outcome: int,
    attribute_values: np.ndarray or pd.Series,
    protected_group: str,
) -> float:
    """Calculate the chi-squared statistic value.

    Parameters
    ----------
    predictions : array_like
        The list of predicted outcomes.
    positive_outcome : str
        The value of the positive outcome.
    attribute_values : array_like
        The list of sensitive attribute's values.
    protected_group : str
        The protected value of a sensitive attribute.

    Returns
    -------
    float
        The chi-squared statistic value.
    """
    # Population size.
    n_total = len(attribute_values)

    # Protected and unprotected groups size.
    n_protected = sum(attribute_values == protected_group)
    n_unprotected = sum(attribute_values != protected_group)

    # Positive and negative outcomes number.
    n_positive = sum(predictions == positive_outcome)
    n_negative = sum(predictions != positive_outcome)

    # For each group the number of positive and negative outcomes.
    n_positive_protected = sum(predictions[attribute_values == protected_group] == positive_outcome)
    n_positive_unprotected = sum(
        predictions[attribute_values != protected_group] == positive_outcome
    )
    n_negative_protected = sum(predictions[attribute_values == protected_group] != positive_outcome)
    n_negative_unprotected = sum(
        predictions[attribute_values != protected_group] != positive_outcome
    )

    # Calculate chi-squared statistic value.
    statistic = (
        _get_group_value(n_positive_protected, n_positive, n_protected, n_total)
        + _get_group_value(n_positive_unprotected, n_positive, n_unprotected, n_total)
        + _get_group_value(n_negative_protected, n_negative, n_protected, n_total)
        + _get_group_value(n_negative_unprotected, n_negative, n_unprotected, n_total)
    )

    return statistic


def remove_sensitive_variable(ohe_list: List, sensitive_variable: str) -> List:
    ohe_wo_sens = []
    for element in ohe_list:
        if not element[0].startswith(sensitive_variable):
            ohe_wo_sens += [element]
    return ohe_wo_sens


def fpdp_dataset(
    df: pd.DataFrame,
    f_col: str,
    cat_cols: list,
    train_size: float = 0.6,
    test_size: float = 0.5,
    random_state: int = 42,
) -> dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """Returns the not encoded values of the f_col on the test set.

    Parameters
    ----------
    df : pd.DataFrame
            Dataframe to split.
    f_col : str
            Column to not be one hot encoded.
    cat_cols : list
            List of columns to be dummy encoded.
    train_size : float
            Percentage of whole data used for training.
    test_size : float
            Percentage for testing from val-test data.
    random_stat : int
            Random state of split for reproducibility.

    Returns
    -------
    data : pd.Series
            Series of the values of the f_col for the test dataset.
    """
    # selecting the columns to one hot encoding
    cols_ohe = cat_cols.copy()
    cols_ohe.remove(f_col)
    # spliting the data
    fpdp_data = split_data(
        cols_ohe, df=df, train_size=train_size, test_size=test_size, random_state=random_state
    )
    # selecting the test set
    fpdp_df = fpdp_data["test"][0]
    # selecting the values of the f_col within the test set.
    fpdp_df = fpdp_df[f_col]
    return fpdp_df


def fairness_pdp_ohe(
    model: Union[
        RandomForestClassifier,
        LogisticRegression,
        DecisionTreeClassifier,
        LogisticGAM,
        XGBClassifier,
        MLPClassifier,
    ],
    feature_names: List[str],
    protected_group: str,
    X: pd.DataFrame,
    sensitive_values: pd.Series,
    fig_size: Tuple[int, int],
    p_value_threshold: float = 0.05,
    y_label: str = "P-value",
) -> plt.figure:
    """Creates the FPDP for one hot encoded variables.

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
    feature_names: List[str]
        List of features of a category.
    protected_group: str
        The protected value of a sensitive attribute.
    X: pd.DataFrame
        Dataframe for which we want the Partial Dependence plot.
    sensitive_values: pd.Series
        Values of the sensitive attribute.
    fig_size: Tuple[int, int]
        Size of the partial dependence plot.
    p_value_threshold: float
        Value of the p_value threshold to be displayed within the FPDP.
    y_label: str
        Label of the y axis of the partial dependence plot.

    Returns
    -------
    fig: plt.figure
        FPDP plot.
    """
    # initializing the p values list
    p_values = []
    # for each one hot encoded variable, compute the p_value of the chi2 statistic
    #  when the predictions where done on X_test setting the value of the OHE variable
    #  to 1 and the others to 0.
    for element in feature_names:
        X_copy = X.copy()
        X_copy[element] = 1
        for second_element in feature_names:
            if element != second_element:
                X_copy[second_element] = 0
        preds = model.predict(X_copy)
        chi2 = get_chi2_statistic(preds, 1, sensitive_values, protected_group)
        d_fred = len(sensitive_values.unique().tolist()) - 1
        p_value = stats.chi2.sf(chi2, d_fred)
        p_values += [p_value]
    # creating the plot.
    fig = plt.figure(figsize=fig_size)
    plt.bar(feature_names, p_values)
    plt.axhline(y=p_value_threshold, color="r", linestyle="-")
    plt.ylabel(y_label)
    plt.xticks(rotation="vertical")
    return fig


def fairness_pdp_cat(
    model: Union[
        RandomForestClassifier,
        LogisticRegression,
        DecisionTreeClassifier,
        LogisticGAM,
        XGBClassifier,
        MLPClassifier,
    ],
    feature: str,
    protected_group: str,
    X: pd.DataFrame,
    sensitive_values: pd.Series,
    fig_size: Tuple[int, int],
    x_label: str,
    p_value_threshold: float = 0.05,
    y_label: str = "P-value",
) -> plt.figure:
    """Gives the FPDP plot for categorical, non one hot encoded columns.

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
    feature: str
        Feature to be taken into account for FPDP.
    protected_group: str
        The protected value of a sensitive attribute.
    X: pd.DataFrame
        Dataframe for which we want the Partial Dependence plot.
    sensitive_values: pd.Series
        Values of the sensitive attribute.
    fig_size: Tuple[int, int]
        Size of the partial dependence plot.
    x_label: str
        Label of the x axis of the partial dependence plot.
    p_value_threshold: float
        Value of the p_value threshold to be displayed within the FPDP.
    y_label: str
        Label of the y axis of the partial dependence plot.

    Returns
    -------
    fig: plt.figure
        FPDP plot.
    """
    # intializing the p values list
    p_values = []
    # obtaining a list of all the different categories.
    unique_vals = np.sort(X[feature].unique().tolist())
    # for each of the categories, compute the p-value of the chi2 statistic obtained
    #  when the predictions where done on X_test setting the value of the categorical variable
    #  to the category.
    for element in unique_vals:
        X_copy = X.copy()
        X_copy[feature] = element
        preds = model.predict(X_copy)
        chi2 = get_chi2_statistic(preds, 1, sensitive_values, protected_group)
        d_fred = len(sensitive_values.unique().tolist()) - 1
        p_value = stats.chi2.sf(chi2, d_fred)
        p_values += [p_value]
    # creating the plot.
    fig = plt.figure(figsize=fig_size)
    unique_vals_plot = [str(x) for x in unique_vals]
    plt.bar(unique_vals_plot, p_values)
    plt.axhline(y=p_value_threshold, color="r", linestyle="-")
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.xticks(rotation="vertical")
    return fig


def fairness_pdp_num(
    model: Union[
        RandomForestClassifier,
        LogisticRegression,
        DecisionTreeClassifier,
        LogisticGAM,
        XGBClassifier,
        MLPClassifier,
    ],
    feature: str,
    step: float,
    protected_group: str,
    X: pd.DataFrame,
    sensitive_values: pd.Series,
    fig_size: Tuple[int, int],
    p_value_threshold: float = 0.05,
    y_label: str = "P-value",
) -> plt.figure():
    """Returns the FPDP for a numerical feature.

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
    feature: str
        Feature to be taken into account for FPDP.
    step: float
        Value by which we increase the value of the feature when computing the p-values.
    protected_group: str
        The protected value of a sensitive attribute.
    X: pd.DataFrame
        Dataframe for which we want the Partial Dependence plot.
    sensitive_values: pd.Series
        Values of the sensitive attribute.
    fig_size: Tuple[int, int]
        Size of the partial dependence plot.
    p_value_threshold: float
        Value of the p_value threshold to be displayed within the FPDP.
    y_label: str
        Label of the y axis of the partial dependence plot.

    Returns
    -------
    fig: plt.figure
        FPDP plot.
    """
    # initializing the p values and the values of the feature
    p_values = []
    i_vals = []
    # initializing the i value at the min value found and finding the max for the upper boundary.
    i = X[feature].min()
    max_val = X[feature].max()
    # looping over the values until i reaches the maximum value, and computing the p values.
    while i <= max_val:
        X_copy = X.copy()
        X_copy[feature] = i
        preds = model.predict(X_copy)
        chi2 = get_chi2_statistic(preds, 1, sensitive_values, protected_group)
        d_fred = len(sensitive_values.unique().tolist()) - 1
        p_value = stats.chi2.sf(chi2, d_fred)
        p_values += [p_value]
        i_vals += [i]
        i += step
    # creating the plot.
    fig = plt.figure(figsize=fig_size)
    plt.plot(i_vals, p_values)
    plt.axhline(y=p_value_threshold, color="r", linestyle="-")
    plt.ylabel(y_label)
    return fig
