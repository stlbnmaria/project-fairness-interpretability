import numpy as np
import pandas as pd
from shap import Explanation, KernelExplainer

from src.explanation.local.base import BaseLocalExplainer


class ShapExplainer(BaseLocalExplainer):
    """The SHAP Explainer class.

    Performs local model explanation using SHapley Additive exPLanations.
    """

    def __init__(self, prediction_function: object, dataset: pd.DataFrame) -> None:
        """The constructor of the class.

        Parameters
        ----------
        prediction_function : object
            The function, which accepts pd.DataFrame
            and gives back np.ndarray with predictions.
        dataset : pd.DataFrame
            The background data.

        Returns
        -------
        Nothing.
        """
        super().__init__(prediction_function, dataset)
        self._explainer = self._init_explainer()

    def _get_background(self) -> pd.DataFrame:
        """Calculate background sample.

        This sample features are used to emulate missing values for the Kernel Explainer.

        Returns
        ----------
        pd.DataFrame
            Background sample.
        """
        numerical_features = self.dataset.select_dtypes(float).columns

        median = self.dataset.median(numeric_only=True)
        background = self.dataset.mode(dropna=False).head(1)

        # Use median of the numerical features.
        for feature in numerical_features:
            background[feature] = median[feature]

        return background.astype(self.dataset.dtypes)

    def _init_explainer(self) -> KernelExplainer:
        """Init Kernel Explainer.

        Returns
        -------
        KernelExplainer
            Shap Kernel Explainer.
        """
        explainer = KernelExplainer(
            model=self.prediction_function,
            data=self._get_background(),
            link="logit",
            keep_index=True,
        )
        return explainer

    def get_shap_explanation(self, x: pd.DataFrame) -> Explanation:
        """Get shap explanations for the given data.

        Parameters
        ----------
        x : pd.DataFrame
            Input data to be explained.

        Returns
        -------
        Explanation
            Shap explanations.
        """
        return self._explainer(x)

    def get_global_explanation(self, x: pd.DataFrame, normalize: bool = False) -> dict[str, float]:
        """Get global shap explanations.

        Calculate mean absolute values of shap values for each feature.
        Parameters
        ----------
        x : pd.DataFrame
            Input data to be explained.
        normalize : bool
            The flag, whether to normalize calculated global explanations.

        Returns
        -------
        dict[str, float]
            The dictionary with feature names and related global explanations.
        """
        shap_explanation = self.get_shap_explanation(x)

        global_explanation = {
            feature: np.abs(shap_explanation[:, feature].values).mean()
            for feature in self.dataset.columns
        }

        if normalize:
            divider = sum(global_explanation.values())
            global_explanation = {k: v / divider for k, v in global_explanation.items()}

        return global_explanation
