from typing import Iterable, Optional

import seaborn as sns


def plot_bar(*args: list, **kwargs: dict) -> None:
    """Plot the bar-plot with feature name vs feature importance value.

    Parameters
    ----------
    *args : list
        Non-keyword arguments.
    *kwargs : dict
        Keyword arguments.

    Returns
    -------
    Nothing.
    """
    sns.barplot(*args, **kwargs)


def plot_scatter(
    feature_values: Iterable,
    feature_importance_values: Iterable,
    feature_name: Optional[str] = None,
    title: str = "",
    **kwargs: dict,
) -> None:
    """Plot the scatter plot with feature values vs related feature importance values.

    Parameters
    ----------
    feature_values : array_like
        Feature values.
    feature_importance_values : array_like
        Feature importance values.
    feature_name : str
        The name of a feature.
    title : str
        The title of the diagram.
    **kwargs : dict
        Keyword arguments.

    Returns
    -------
    Nothing.
    """
    ax = sns.scatterplot(x=feature_values, y=feature_importance_values, **kwargs)

    if feature_name is not None:
        ax.set(
            xlabel=f"{feature_name}",
            ylabel=f"Attribution value for {feature_name}",
            title=f"{title}",
        )
