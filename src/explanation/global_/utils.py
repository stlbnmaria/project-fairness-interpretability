from typing import List, Tuple


def get_underscore(cols: List[str]) -> Tuple[List[str]]:
    """Allows us to separate the columns which have an _ in their name.

    Parameters
    ----------
    cols: List[str]
        List of columns we want to separate.

    Returns
    -------
    underscore_columns: List[str]
        List of columns with a _.
    non_underscore_columns: List[str]
        List of columns without a _.
    """
    # initializing lists of columns with and without _
    underscore_columns = []
    non_underscore_columns = []
    # checking for each column if they have a _ and assigning them to their respective list
    for col in cols:
        if "_" in col:
            underscore_columns += [col]
        else:
            non_underscore_columns += [col]
    return underscore_columns, non_underscore_columns


def ohe_filter(non_underscore_cols: List[str], cols: List[str]) -> Tuple[List[str]]:
    """Given columns with/without a _, returns list of columns which were/were not one hot encoded.

    Parameters
    ----------
    non_underscore_columns: List[str]
        List of columns without a _.
    underscore_columns: List[str]
        List of columns with a _.

    Returns
    -------
    non_ohe: List[str]
        List of columns which were not one hot encoded.
    ohe: List[str]
        List of columns which were one hot encoded.
    prefix_ohe: List[str]
        List of prefixes of one hot encoded columns.
    """
    # starting a dict with as keys the prefixes and as values the columns with those prefixes.
    prefix_dict = dict()
    # finding the prefixes for each column and adding them to the dictionary.
    for col in cols:
        prefix = col.split("_")[0]
        if prefix in prefix_dict:
            prefix_dict[prefix] += [col]
        else:
            prefix_dict[prefix] = [col]
    # adding all columns without underscore to non_ohe columns list.
    non_ohe = [[col] for col in non_underscore_cols]
    ohe = []
    prefix_ohe = []
    # checking for each prefix if there is more than one column
    # if so, OHE happened and so we assign them to the ohe list, and the prefix to the prefix list.
    for key, value in prefix_dict.items():
        if len(value) > 1:
            ohe += [value]
            prefix_ohe += [key]
        else:
            non_ohe += [value]
    return non_ohe, ohe, prefix_ohe
