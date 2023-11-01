import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def auc_score(y: pd.Series, pred_probs: np.ndarray) -> float:
    """Computes the AUC metric, given the true y values and the predicted probabilities(pred_probs).

    Parameters
    ----------
    y: pd.Series
        True y_labels.
    pred_probs: np.ndarray
        Predicted probabilities of y given by a model.

    Returns
    -------
    AUC: float
        Area under the curve metric value.
    """
    AUC = roc_auc_score(y, pred_probs)
    return AUC


def accuracy(y: pd.Series, preds: np.ndarray) -> float:
    """Computes the Accuracy metric, given the true y values and the predicted labels(preds).

    Parameters
    ----------
    y: pd.Series
        True y_labels.
    preds: np.ndarray
        Predicted labels of y given by a model.

    Returns
    -------
    Acc: float
       Accuracy metric value.
    """
    Acc = accuracy_score(y, preds)
    return Acc


def precision(y: pd.Series, preds: np.ndarray) -> float:
    """Computes the Precision metric, given the true y values and the predicted labels(preds).

    Parameters
    ----------
    y: pd.Series
        True y_labels.
    preds: np.ndarray
        Predicted labels of y given by a model.

    Returns
    -------
    precision: float
       Precision metric value.
    """
    precision = precision_score(y, preds)
    return precision


def recall(y: pd.Series, preds: np.ndarray) -> float:
    """Computes the Recall metric, given the true y values and the predicted labels(preds).

    Parameters
    ----------
    y: pd.Series
        True y_labels.
    preds: np.ndarray
        Predicted labels of y given by a model.

    Returns
    -------
    recall: float
       Recall metric value.
    """
    recall = recall_score(y, preds)
    return recall


def f1(y: pd.Series, preds: np.ndarray) -> float:
    """Computes the F1 score, given the true y values and the predicted labels(preds).

    Parameters
    ----------
    y: pd.Series
        True y_labels.
    preds: np.ndarray
        Predicted labels of y given by a model.

    Returns
    -------
    f1: float
       F1 score value.
    """
    f1 = f1_score(y, preds)
    return f1


def eval_metrics(y: pd.Series, preds: np.ndarray, pred_probs: np.ndarray) -> dict[str, float]:
    """Given the true y, predicted labels and probs, returns a dict with the different metrics.

    Parameters
    ----------
    y: pd.Series
        True y_labels.
    preds: np.ndarray
        Predicted labels of y given by a model.
    pred_probs: np.ndarray
        Predicted probabilities of y given by a model.

    Returns
    -------
    metrics: dict[str,float]
        Dictionary containing the different metrics.
    """
    metrics = dict()

    # AUC
    metrics["AUC"] = auc_score(y, pred_probs)

    # Accuracy
    metrics["Acc"] = accuracy(y, preds)

    # Precision
    metrics["Precision"] = precision(y, preds)

    # Recall
    metrics["Recall"] = precision(y, preds)

    # F1_score
    metrics["F1_score"] = f1(y, preds)

    return metrics
