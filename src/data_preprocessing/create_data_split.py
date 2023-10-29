from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

def dummy_encoding(data_path: Path = None, df: pd.DataFrame = None, cols: list = ["Color", "Gender", "Make", "Race", "VehicleType"], catboost: bool = False) -> pd.DataFrame:
    """Encode the categorical variables within the dataframe.

    Parameters
    -------
    data_path : Path
            Path to the data.
    df : pd.DataFrame
            Dataframe to split.
    cols : list
            List with the categorical variables we wish to dummy encode.
    catboost : Bool
            Represents whether or not we are preparing the data to be used in a catboost model.

    Returns
    -------
    one_hot_df : pd.DataFrame
            Dataframe with its categorical variables one-hot-encoded(returns the original df if catboost=True).
    """

    # check that at least one has a value
    assert not (
        (data_path is None) and (df is None)
    ), "You need to pass either a path to data or a dataframe."
    # load data from path
    if data_path:
        df = pd.read_csv(data_path)
    if catboost:
        return df
    else:
        one_hot_df = pd.get_dummies(df, columns = ["Color", "Gender", "Make", "Race", "VehicleType"], dtype=int)
        return one_hot_df




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
