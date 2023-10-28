from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

# TODO: add bool for dummy encoding


def split_data(
    data_path: Path = None,
    df: pd.DataFrame = None,
    train_size: float = 0.6,
    test_size: float = 0.5,
    random_state: int = 42,
) -> dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """Create a train-test-val split for modeling / anylsis.

    Parameters
    -------
    data_path : Path
            Path to the data.
    df : pd.DataFrame
            Dataframe to split.
    train_size : float
            Percentage of whole data used for training.
    test_size : float
            Percentage for testing from val-test data.
    random_stat : int
            Random state of split for reproducibility.

    Returns
    -------
    data : dict[str, Tuple[pd.DataFrame, pd.Series]]
            Dict with X,y per train, val, test.
    """
    # check that at least one has a value
    assert not (
        (data_path is None) and (df is None)
    ), "You need to pass either a path to data or a dataframe."

    # load data from path
    if data_path:
        df = pd.read_csv(data_path)

    # create split betweeen train and val-test
    X_train, X_val_test, y_train, y_val_test = train_test_split(
        df.drop(columns="Citation"),
        df["Citation"],
        train_size=train_size,
        random_state=random_state,
    )

    # split of test and val data
    X_val, X_test, y_val, y_test = train_test_split(
        X_val_test,
        y_val_test,
        test_size=test_size,
        random_state=random_state,
    )

    # create dict for phases
    data = {"train": (X_train, y_train), "val": (X_val, y_val), "test": (X_test, y_test)}

    return data
