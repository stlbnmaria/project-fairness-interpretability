from pathlib import Path
from typing import List, Tuple

import pandas as pd
from scipy.io.arff import loadarff
from scipy.io.arff._arffread import MetaData

from config.config_data import DATA_PATH, DROP_COLS, OUT_PATH


def load_data(path_: Path) -> Tuple[pd.DataFrame, MetaData]:
    """Loads the .arff file (incl. metadata) and converts to utf-8.

    Parameters
    -------
    path_ : Path
            Path of the data.

    Returns
    -------
    data : pd.DataFrame
            Data as a dataframe.
    meta : scipy.io.arff._arffread.Metadata
            Metadata of the dataset.
    """
    # load df and metadata from .arff
    data, meta = loadarff(path_)
    data = pd.DataFrame(data)

    # remove b string from data
    str_df = data.select_dtypes([object])
    str_df = str_df.reset_index().melt(id_vars="index").set_index("index")
    str_df["value"] = str_df["value"].str.decode("utf-8")

    # rename the 'value' column to avoid conflicts and perform pivot
    str_df = str_df.rename(columns={"value": "decoded_value"})
    str_df = pd.pivot_table(
        str_df, columns="variable", values="decoded_value", index="index", aggfunc=lambda x: x
    )

    # reset both the column and index names to None
    str_df = str_df.rename_axis(index=None, columns=None)

    # merge str and non-str columns
    data = pd.concat([str_df, data.select_dtypes(exclude=[object])], axis=1)

    return data, meta


def change_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Change yes/no values in columns to 0/1.

    Parameters
    -------
    df : pd.DataFrame
            Data to transform.

    Returns
    -------
    df : pd.DataFrame
            Transformed data.
    """
    for col in df.columns:
        # only change columns that have no missing values to 0 / 1
        if set(df[col].unique().tolist()) - set(["No", "Yes"]) == set():
            df[col] = df[col].map(dict(Yes=1, No=0))

    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Creates new features based and drops the old ones.

        - If the Vehicle's State is Maryland (MD)
        - If the Driver's State is Maryland (MD)
        - If the Driver License's State is Maryland (MD)

    Parameters
    -------
    df : pd.DataFrame
            Data to transform.

    Returns
    -------
    df : pd.DataFrame
            Transformed data.
    """
    # Vehicle's State
    df["State_MD"] = (df["State"] == "MD").astype(int)
    df.loc[df["State"] == "?", "State_MD"] = -1

    # Driver's State
    df["Driver_State_MD"] = (df["Driver.State"] == "MD").astype(int)
    df.loc[df["Driver.State"] == "?", "Driver_State_MD"] = -1

    # Driver License's State
    df["DL_State_MD"] = (df["DL.State"] == "MD").astype(int)
    df.loc[df["DL.State"] == "?", "DL_State_MD"] = -1

    df = df.drop(columns=["State", "Driver.State", "DL.State"])
    return df


def transform_label(df: pd.DataFrame) -> pd.DataFrame:
    """Drops rows that are equal to SERO and changes label to "Citation" with 0/1 - values.

    Parameters
    -------
    df : pd.DataFrame
            Data to transform.

    Returns
    -------
    df : pd.DataFrame
            Transformed data.
    """
    df = df[df["Violation.Type"] != "SERO"].copy()
    df["Citation"] = df.loc[:, "Violation.Type"].apply(lambda x: 1 if x == "Citation" else 0)
    df = df.drop(columns=["Violation.Type"])
    return df


def drop_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Drop columns from dataframe.

    Parameters
    -------
    df : pd.DataFrame
            Data to transform.
    cols : List
            List of columns to drop.

    Returns
    -------
    df : pd.DataFrame
            Transformed data.
    """
    df = df.drop(columns=cols)

    return df


def filter_na(df: pd.DataFrame) -> pd.DataFrame:
    """Filter na-values in full df.

        Since analysis showed that values are missing at random
        across groups, make less than 1% of instances and instance normally has
        multiple missing feature values, this operation is valid.

    Parameters
    -------
    df : pd.DataFrame
            Data to transform.

    Returns
    -------
    df : pd.DataFrame
            Transformed data.
    """
    # filter out -1 in integer columns
    cols = df.select_dtypes([int]).columns
    for col in cols:
        df = df[df[col] != -1].copy()

    # filter out ? in string columns
    cols = df.select_dtypes([object]).columns
    for col in cols:
        df = df[df[col] != "?"].copy()

    return df


def preprocessor(data_path: Path, cols: List[str]) -> pd.DataFrame:
    """Load data and perform preprocessing steps.

    Parameters
    -------
    path_ : Path
            Path of the data.
    cols : List
            List of columns to drop.

    Returns
    -------
    data : pd.DataFrame
            Processed data.
    """
    data, _ = load_data(data_path)
    data = change_to_numeric(data)
    data = feature_engineering(data)
    data = drop_cols(data, cols)
    data = filter_na(data)

    return data


if __name__ == "__main__":
    data = preprocessor(DATA_PATH, DROP_COLS)
    data.to_csv(OUT_PATH, index=False)
