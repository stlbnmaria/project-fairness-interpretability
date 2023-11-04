import argparse
from pathlib import Path
from typing import List

import pandas as pd
from create_data_split import split_data
from data_preprocessor import preprocessor

from config.config_data import DATA_PATH, DROP_COLS
from config.config_modeling import CAT_COLS, RANDOM_STATE, TEST_FROM_VAL, TRAIN_SIZE


def main(
    data_path: Path,
    drop_cols: List,
    cat_cols: List,
    train_size: float,
    test_size: float,
    random_state: int,
) -> None:
    data = preprocessor(data_path, drop_cols)
    data.to_csv(Path(data_path.parent / "data.csv"), index=False)

    print("------Data preprocessing Done-----")

    data = split_data(
        cols=cat_cols,
        data_path=Path(data_path.parent / "data.csv"),
        train_size=train_size,
        test_size=test_size,
        random_state=random_state,
    )

    for phase, (X, y) in data.items():
        df_X = pd.DataFrame(X)
        df_y = pd.DataFrame(y)
        csv_file_name_X = f"{phase}_X.csv"
        df_X.to_csv(Path(data_path.parent / csv_file_name_X), index=False)
        csv_file_name_Y = f"{phase}_Y.csv"
        df_y.to_csv(Path(data_path.parent / csv_file_name_Y), index=False)

    print("------Data split Done-----")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", default=DATA_PATH, help="Path to data")
    parser.add_argument("--drop_cols", default=DROP_COLS, help="Columns to drop")
    parser.add_argument(
        "--cat_cols", default=CAT_COLS, help="Categorical columns for dummy encoding"
    )
    parser.add_argument("--train_size", default=TRAIN_SIZE, help="Share of data for train")
    parser.add_argument(
        "--test_size", default=TEST_FROM_VAL, help="Share of val-test data for test"
    )
    parser.add_argument("--random_state", default=RANDOM_STATE, help="Random state of data split")

    args = parser.parse_args()

    main(
        data_path=args.data_path,
        drop_cols=args.drop_cols,
        cat_cols=args.cat_cols,
        train_size=args.train_size,
        test_size=args.test_size,
        random_state=args.random_state,
    )
