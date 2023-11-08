"""The implementation of the (conditional) statistical parity test.

The model fairness is tested using the Standard Pearson chi-squared independence test.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class TestResult:
    """The representation of the chi-squared test result.

    Parameters
    ----------
    passed : bool
        The flag, signifying if the statistical test has passed.
    statistic_value : float
        The value of a calculated statistic.
    quantile_value : float
        The value of a distribution's quantile.
    quantile_number: float
        The number of the quantile.
    degrees_of_freedom: int
        The number of distribution's degrees of freedom.
    """

    passed: bool
    statistic_value: float
    quantile_value: float
    quantile_number: float
    degrees_of_freedom: int

    def __str__(self) -> str:
        """Test result representation.

        Returns
        -------
        str
            The test result summary.
        """
        return (
            f"The chi^2 fairness test of statistical parity.\n"
            f"The null hypothesis - the prediction and the sensitive attribute are independent.\n"
            f"Result: \n"
            f"Passed: {self.passed}\n"
            f"Statistic value: {self.statistic_value}\n"
            f"Quantile value: {self.quantile_value}\n"
            f"Parameters:\n"
            f"Quantile number: {self.quantile_number}\n"
            f"DoF: {self.degrees_of_freedom}"
        )


def _get_group_value(n_outcome_group: int, n_outcome: int, n_group: int, n_total: int) -> float:
    """Calculate the term of the chi-squared statistic for the given group.

    Parameters
    ----------
    n_outcome_group : int
        The number of records with the 'group' and 'outcome' values.
    n_outcome : int
        The number of records with the 'outcome' values.
    n_group : int
        The number of records with the 'group' values.
    n_total : int
        The total size of a subpopulation.

    Returns
    -------
    float
        The term for the given group.
    """
    expectation = n_group * n_outcome / n_total
    res = (n_outcome_group - expectation) ** 2 / expectation
    return res


def _get_chi2_statistic(
    predictions: np.ndarray | pd.Series,
    positive_outcome: str,
    attribute_values: np.ndarray | pd.Series,
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
    protected_group : The protected value of a sensitive attribute.

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


def statistical_parity_test(
    dataset_list: list[pd.DataFrame],
    sensitive_attribute: str,
    protected_group: str,
    target_column: str,
    positive_outcome: str,
    p_quantile: float = 0.95,
) -> TestResult:
    """Perform statistical parity test.

    If multiple datasets are passed the "conditional" statistical parity test is performed.
    Parameters
    ----------
    dataset_list : array_like
        The list of population subgroups. Single element list is considered as
        a whole population. Each dataset must contain 'target' (ground truth or prediction) column.
    sensitive_attribute : str
        The attribute which is tested for a model's fairness.
        F.e. the 'sex' or the 'race'.
    protected_group : str
        The sensitive attribute's protected value.
    target_column : str
        The target column.
    positive_outcome : str
        The positive outcome value.
    p_quantile : float
        The p quantile of the chi-squared distribution.

    Returns
    -------
    float
        The 'TestResult' object encapsulating all information about
        the statistical parity test performed.
    """
    # Calculate test statistic.
    chi2_statistic = 0
    for dataset in dataset_list:
        chi2_statistic += _get_chi2_statistic(
            dataset[target_column], positive_outcome, dataset[sensitive_attribute], protected_group
        )

    # Calculate threshold value.
    degrees_of_freedom = len(dataset_list)
    quantile_value = stats.chi2.ppf(p_quantile, degrees_of_freedom)

    # Form the test result.
    test_result = TestResult(
        passed=(chi2_statistic <= quantile_value),
        statistic_value=chi2_statistic,
        quantile_value=quantile_value,
        quantile_number=p_quantile,
        degrees_of_freedom=degrees_of_freedom,
    )
    return test_result
