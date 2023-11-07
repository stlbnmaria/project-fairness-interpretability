from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class TestResult:
    """Representation of the chi^2 test result."""

    passed: bool
    statistic_value: float
    quantile_value: float
    quantile_number: float
    degrees_of_freedom: int

    def __str__(self) -> str:
        """Test result representation."""
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
    """Calculate the term of the chi^2 statistic for the given group.

    :param n_outcome_group: The number of records with the 'group' and 'outcome' values.
    :param n_outcome: The number of records with the 'outcome' values.
    :param n_group: The number of records with the 'group' values.
    :param n_total: The total size of a sub-/population.
    :return: The term for the given group.
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
    """Calculate the chi^2 statistic value.

    :param predictions: The list of predicted outcomes.
    :param positive_outcome: The value of the positive outcome.
    :param attribute_values: The list of sensitive attribute's values.
    :param protected_group: The protected value of a sensitive attribute.
    :return: The chi^2 statistic value.
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
    :param dataset_list: The list of population subgroups. Single element list is considered as
        a whole population. Each dataset must contain 'target' (ground truth or prediction) column.
    :param sensitive_attribute: The attribute which is tested for a model's fairness.
        F.e. the 'sex' or the 'race'.
    :param protected_group: The sensitive attribute's protected value.
    :param target_column: The target column.
    :param positive_outcome: The positive outcome value.
    :param p_quantile: The p quantile of the chi^2 distribution.
    :return: The 'TestResult' object encapsulating all information about
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
