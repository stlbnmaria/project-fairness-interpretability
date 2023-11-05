from collections import defaultdict

import numpy as np
import pandas as pd
from lime.explanation import Explanation
from lime.lime_tabular import LimeTabularExplainer
from tqdm import tqdm

from src.explanation.local.base import BaseLocalExplainer


class LimeExplainer(BaseLocalExplainer):
    def __init__(self, prediction_function: object, dataset: pd.DataFrame) -> None:
        """The Lime Explainer class.

        Performs local model explanation using LIME approach.
        :param prediction_function: The function, which accepts pd.DataFrame
            and gives back np.ndarray with predictions.
        :param dataset: The pd.DataFrame with training data to initialize LIME explainer.
        """
        super().__init__(prediction_function, dataset)
        self._explainer = self._init_explainer()

    def _init_explainer(self) -> LimeTabularExplainer:
        """Init LIME Tabular Explainer.

        :return: LIME Tabular Explainer.
        """
        explainer = LimeTabularExplainer(
            training_data=self.dataset.values,
            feature_names=self.dataset.columns,
            discretize_continuous=False,
        )
        return explainer

    def get_lime_explanation(self, x: pd.Series, **kwargs: dict) -> Explanation:
        """Get LIME explanation for the given data row.

        :param x: Input data raw to be explained.
        :return: LIME explanations.
        """
        lime_explanation = self._explainer.explain_instance(
            data_row=x, predict_fn=self.prediction_function, num_features=len(x), **kwargs
        )
        return lime_explanation

    def get_global_explanation(self, x: pd.DataFrame, normalize: bool = False) -> dict[str, float]:
        """Get global LIME explanations.

        Calculate mean absolute values of LIME values for each feature.

        :param x: Input data to be explained.
        :param normalize: The flag, whether to normalize calculated global explanations.
        :return: Dict with feature names and related global explanations.
        """
        global_explanation = defaultdict(list)
        for i in tqdm(range(len(x))):
            sample_explanation = self.get_lime_explanation(x.iloc[i]).as_list()

            for k, v in sample_explanation:
                global_explanation[k].append(np.abs(v))

        global_explanation = {k: np.mean(v) for k, v in global_explanation.items()}

        if normalize:
            divider = sum(global_explanation.values())
            global_explanation = {k: v / divider for k, v in global_explanation.items()}

        return global_explanation
