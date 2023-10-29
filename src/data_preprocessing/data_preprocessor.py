import re
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import yaml
from fuzzywuzzy import fuzz, process
from scipy.io.arff import loadarff
from scipy.io.arff._arffread import MetaData

from config.config_data import (
    DATA_PATH,
    DROP_COLS,
    M_DICT_H_PATH,
    M_DICT_PATH,
    N_CATEGORIES,
    OUT_PATH,
)


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


def convert_float_to_int(df: pd.DataFrame, column_name: str = "Year") -> pd.DataFrame:
    """If possible to convert float to int converts to int.

    Args:
        df (pd.DataFrame): df of which column should be converted
        column_name (str): column name that should be converted to int
    """
    if df[column_name].dropna().apply(lambda x: x.is_integer()).all():
        df[column_name] = df[column_name].fillna(-1).astype(int)
    else:
        print("Can't be converted to int")

    return df


def read_yaml(path: Path) -> dict:
    """Reads yaml file from given path and returns as dict.

    Args:
        path (Path): the path of the respective yaml file

    Returns:
        dict: the yaml file reconverted to a dict
    """
    with open(path, "r") as yaml_file:
        make_match_dictionary = yaml.load(yaml_file, Loader=yaml.FullLoader)

    return make_match_dictionary


def clean_string(s: str) -> str:
    """Cleans the string by converting to lowercase and removing alphabetical values.

    Args:
        s (string): the string that should be changes

    Returns:
        string: converted string t
    """
    # Remove non-alphanumeric characters and convert to lowercase
    s = re.sub(r"[^a-zA-Z0-9\s]", "", s)
    s = s.lower()
    return s


# Function to find the best match
def replace_with_best_match(
    df: pd.DataFrame,
    choices: dict,
    threshold: int = 50,
    column_name: str = "Make",
) -> pd.DataFrame:
    """Finds best match between value of the dataframe & of dictionary.

    Args:
        df (pd.DataFrame): The input DataFrame where one column should
        be replaced with its best match from the choices dictionary.

        column_name (str): The name of column that should be replaced with it's best match
        choices (dict):  A dicitionary in which the values
        of a row are similar words or choices for matching with the respective value of the column
        threshold (int, optional): The threshold set for
        the similarity score between the column value and the choices. Defaults to 50.
    """

    def get_best_match(value: str) -> str:
        """Finds the best match for a given value within the choices.

        Args:
            value (str): The value to find the best match for.

        Returns:
            str: The best match for the input value.
        """
        if not value or len(value) < 3:  # Skip empty strings and very short strings
            return value, 0

        # Clean the input value
        value = clean_string(value)
        # Clean the choices
        cleaned_choices = [clean_string(choice) for choice in choices]

        # Use fuzz.token_set_ratio for better token matching
        best_match, score = process.extractOne(value, cleaned_choices, scorer=fuzz.token_set_ratio)

        if score >= threshold:
            best_match = best_match
        else:
            best_match = value

        return best_match

    df[column_name] = df[column_name].apply(lambda x: get_best_match(x))

    return df


def replace_with_hard(data: pd.DataFrame, dict_hard: dict, column: str = "Make") -> pd.DataFrame:
    data[column] = data[column].replace(dict_hard)
    return data


def categorize_top_n(
    df: pd.DataFrame, column_name: str = "Make", n: int = N_CATEGORIES
) -> pd.DataFrame:
    """Keep top n classes & missing values and set rest of categories as other.

    Args:
        df (pd.DataFrame): The df used
        column_name (str): The name of the column that should be recategorized
        n (_type_): _description_

    Returns:
        pd.DataFrame: returns dataframe with limited number of classes
    """
    # get the value counts for the specified column
    value_counts = df[column_name].value_counts()

    # get top n categories
    top_n_categories = value_counts.index[:n].tolist()
    top_n_categories.append("?")

    # repalce categories not in top n with 'Other'
    df[column_name] = df[column_name].apply(lambda x: x if x in top_n_categories else "Other")

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
    data = transform_label(data)
    data = convert_float_to_int(data)
    make_dict = read_yaml(M_DICT_PATH)
    data = replace_with_best_match(data, make_dict)
    make_dict_hard = read_yaml(M_DICT_H_PATH)
    data = replace_with_hard(data, make_dict_hard)
    data = categorize_top_n(data)
    data = drop_cols(data, cols)
    data = filter_na(data)

    return data


if __name__ == "__main__":
    data = preprocessor(DATA_PATH, DROP_COLS)
    data.to_csv(OUT_PATH, index=False)
