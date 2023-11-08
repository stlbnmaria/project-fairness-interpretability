"""The base class for the local explainers."""

from abc import ABC, abstractmethod

import pandas as pd


class BaseLocalExplainer(ABC):
    """The base local explainer class.

    All classes implementing local explanation approaches must inherit
    from it.
    """

    def __init__(self, prediction_function: object, dataset: pd.DataFrame) -> None:
        """The constructor of the class.

        Parameters
        ----------
        prediction_function : object
            The function, which accepts pd.DataFrame and
            gives back np.ndarray with predictions.
        dataset : pd.DataFrame
            The background data.

        Returns
        -------
        Nothing.
        """
        self.prediction_function = prediction_function
        self.dataset = dataset

    @abstractmethod
    def get_global_explanation(self, x: pd.DataFrame, normalize: bool = False) -> dict[str, float]:
        """Get features global explanation.

        Must be implemented by all child classes.
        Parameters
        ----------
        x : pd.DataFrame
            Input data to be explained.
        normalize : bool
            The flag, signifying if to normalize calculated global explanations.

        Returns
        -------
        dict[str, float]
            The dictionary with feature names and related global explanations.
        """
        raise NotImplementedError
