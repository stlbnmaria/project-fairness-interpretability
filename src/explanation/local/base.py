from abc import ABC, abstractmethod

import pandas as pd


class BaseLocalExplainer(ABC):
    def __init__(self, prediction_function: object, dataset: pd.DataFrame) -> None:
        """The base local explainer class.

        All classes implementing local explanation approaches must inherit
        from it.
        :param prediction_function: The function, which accepts pd.DataFrame
            and gives back np.ndarray with predictions.
        :param dataset: The pd.DataFrame with the background data.
        """
        self.prediction_function = prediction_function
        self.dataset = dataset

    @abstractmethod
    def get_global_explanation(self, x: pd.DataFrame, normalize: bool = False) -> dict[str, float]:
        """Get features global explanation.

        Must be implemented by all child classes.
        """
        raise NotImplementedError
