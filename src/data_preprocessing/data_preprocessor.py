import datetime
import re
import string
from pathlib import Path
from typing import List, Tuple

import nltk
import pandas as pd
import spacy
import yaml
from fuzzywuzzy import fuzz, process
from nltk.corpus import stopwords
from scipy.io.arff import loadarff
from scipy.io.arff._arffread import MetaData
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from config.config_data import (
    DATA_PATH,
    DICT_H_PATH,
    DICT_PATH,
    DROP_COLS,
    N_CATEGORIES,
    N_TOPICS,
    OUT_PATH,
)

nltk.download("stopwords")
nlp = spacy.load("en_core_web_sm")


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

    Parameters
    -------
    df : pd.DataFrame
            Data to transform.
    column_name : str
            Column that should be converted to int.

    Returns
    -------
    df : pd.DataFrame
            Transformed data.
    """
    assert (
        df[column_name].dropna().apply(lambda x: x.is_integer()).all()
    ), "Can't be converted to int"
    df[column_name] = df[column_name].fillna(-1).astype(int)

    # get today's year and filter by -1 (na), above 1990 or below/equal today's year
    year = int(datetime.date.today().strftime("%Y"))
    df = df[(df[column_name] == -1) | ((df[column_name] > 1990) & (df[column_name] <= year))]

    return df


def read_yaml(path: Path) -> dict:
    """Reads yaml file from given path and returns as dict.

    Parameters
    -------
    path : Path
            Path of the respective yaml file.

    Returns
    -------
    make_match_dictionary : dict
            Yaml file loaded as dict.
    """
    with open(path, "r") as yaml_file:
        dictionary = yaml.load(yaml_file, Loader=yaml.FullLoader)

    return dictionary


def clean_string(s: str) -> str:
    """Cleans the string by converting to lowercase and removing alphabetical values.

    Parameters
    -------
    s : str
        String that should be changd.

    Returns
    -------
    s : str
        Converted string.
    """
    # Remove non-alphanumeric characters and convert to lowercase
    s = re.sub(r"[^a-zA-Z0-9\s]", "", s)
    s = s.lower()
    return s


def get_best_match(value: str, choices: dict, threshold: int = 50) -> str:
    """Finds the best match for a given value within the choices.

    Parameters
    -------
    value : str
        The value to find the best match for.
    choices : dict
        A dicitionary in which the values
    choices : dict
        A dicitionary in which the values of a row are similar words or
        choices for matching with the respective value of the column.
    threshold : int
        The threshold set for the similarity score between the
        column value and the choices. Defaults to 50.

    Returns
    -------
    best_match : str
        The best match for the input value.
    """
    if not value or len(value) < 3:  # Skip empty strings and very short strings
        return value

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


def replace_with_best_match(
    df: pd.DataFrame, column_name: str, choices: dict, threshold: int = 50
) -> pd.DataFrame:
    """Finds best match between value of the dataframe & of dictionary.

    Parameters
    -------
    df : pd.DataFrame
        The input DataFrame where one column should.
    column_name : str
        The name of column that should be replaced with it's best match.
    choices : dict
        A dicitionary in which the values of a row are similar words or
        choices for matching with the respective value of the column.
    threshold : int
        The threshold set for the similarity score between the
        column value and the choices. Defaults to 50.

    Returns
    -------
    data : pd.DataFrame
        Transofrmed dataframe.
    """
    df[column_name] = df[column_name].apply(lambda x: get_best_match(x, choices, threshold))

    return df


def replace_with_hard(data: pd.DataFrame, dict_hard: dict, column: str) -> pd.DataFrame:
    """Replaces existing categories based on a hard encoded dictionary.

    Parameters
    -------
    data : pd.DataFrame
        Dataframe to transform.
    dict_hard : dict
        Dict that will be used to change the values of a column.
    column : str
        The column that will be changed.

    Returns
    -------
    data : pd.DataFrame
        Transofrmed dataframe.
    """
    inv_map = {val: k for k, v in dict_hard.items() for val in v}
    data[column] = data[column].replace(inv_map)
    return data


def categorize_top_n(df: pd.DataFrame, column_name: str, n: int = 10) -> pd.DataFrame:
    """Keep top n classes & missing values and set rest of categories as other.

    Parameters
    -------
    df : pd.DataFrame
        Dataframe to transform.
    column_name : str
        Name of the column that should be grouped.
    n : int
        Top n categories that shoul be kept explicitly.

    Returns
    -------
    df : pd.DataFrame
        Dataframe with limited number of classes.
    """
    # get the value counts for the specified column
    value_counts = df[column_name].value_counts()

    # get top n categories
    top_n_categories = value_counts.index[:n].tolist()
    top_n_categories.append("?")

    # repalce categories not in top n with 'other'
    df[column_name] = df[column_name].apply(lambda x: x if x in top_n_categories else "other")

    return df


def preprocess_text(df: pd.DataFrame, column_name: str = "Description") -> pd.DataFrame:
    """Reformats text so that LDA can be applied in the next step.

    Parameters
    -------
    df : pd.DataFrame
            Data to transform.
    column_name : (str, optional)
            Name of column to be transformed.
            Defaults to "Description".

    Returns
    -------
    df : pd.DataFrame
            Transformed data.
    """
    # making text lower case
    lowercase_text = df[column_name].apply(lambda x: x.lower())

    # tokenize text
    tokenized_text = lowercase_text.apply(lambda x: nltk.word_tokenize(x))

    # removing stopwords
    stop_words = set(stopwords.words("english"))
    clean_text = tokenized_text.apply(
        lambda tokens: [
            word for word in tokens if word not in stop_words and word not in string.punctuation
        ]
    )

    # lemmatize words
    lemmatized_text = clean_text.apply(
        lambda tokens: [token.lemma_ for token in nlp(" ".join(tokens))]
    )

    df["description_clean"] = lemmatized_text.apply(lambda lem_tokens: " ".join(lem_tokens))

    return df


def create_n_topics(
    df: pd.DataFrame,
    column_name: str = "description_clean",
    n_topics: int = 10,
    max_features: int = 1000,
) -> pd.DataFrame:
    """Applies LDA to a text column of DF and adds LDA topic distributions as new features.

    Parameters
    ----------
    df : pd.DataFrame
        Data to transform.

    column_name : str, optional
        Name of the column to be transformed. Defaults to "Description".

    num_topics : int, optional
        Number of topics for LDA. Defaults to 10.

    max_features : int, optional
        Maximum number of features for CountVectorizer. Defaults to 1000.

    Returns
    -------
    df : pd.DataFrame
        Transformed data with added LDA topic features.
    """
    # Create a CountVectorizer
    vectorizer = CountVectorizer(max_features=max_features, stop_words="english")
    X = vectorizer.fit_transform(df[column_name])

    # create an LDA model
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)

    lda.fit(X)

    topic_distributions = lda.transform(X)

    for i in range(n_topics):
        df[f"Topic_{i+1}"] = topic_distributions[:, i]

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
        df = df[~df[col].isin(["U", "?"])].copy()

    return df


def preprocessor(data_path: Path, n_topics: int, cols: List[str]) -> pd.DataFrame:
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
    # load the data
    data, _ = load_data(data_path)

    # convert yes/no to 0/1 and year to int
    data = change_to_numeric(data)
    data = convert_float_to_int(data)

    # perform feature engineering on state columns
    data = feature_engineering(data)

    # preform refactroing and grouping on certain columns
    make_dict = read_yaml(DICT_PATH)["Make"]
    data = replace_with_best_match(data, "Make", make_dict)
    dict_hard = read_yaml(DICT_H_PATH)
    for key in dict_hard.keys():
        data = replace_with_hard(data, dict_hard[key], column=key)
        data = categorize_top_n(data, column_name=key, n=N_CATEGORIES)

    # transform native american to other category and only keep top 4 explicit
    data = categorize_top_n(data, column_name="Race", n=4)

    # transform describe column so that NLP can be applied
    data = preprocess_text(data)

    # extracts n new topics from descibe column
    data = create_n_topics(data, n_topics=n_topics)

    # drop unwished cols
    data = drop_cols(data, cols)

    # transform label to 0/1 for citation
    data = transform_label(data)

    # filter na
    data = filter_na(data)

    return data


if __name__ == "__main__":
    data = preprocessor(DATA_PATH, N_TOPICS, DROP_COLS)
    data.to_csv(OUT_PATH, index=False)
