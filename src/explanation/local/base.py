import pandas as pd


class BaseLocalExplainer:
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
