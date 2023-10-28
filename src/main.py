import argparse
from pathlib import Path

from config.config_data import OUT_PATH
from config.config_modeling import RANDOM_STATE, TEST_FROM_VAL, TRAIN_SIZE
from data_preprocessing.create_data_split import split_data


def main(data_path: Path, train_size: float, test_size: float, random_state: int) -> None:
    split_data(
        data_path=data_path,
        train_size=train_size,
        test_size=test_size,
        random_state=random_state,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", default=OUT_PATH, help="Path to data")
    parser.add_argument("--train_size", default=TRAIN_SIZE, help="Share of data for train")
    parser.add_argument(
        "--test_size", default=TEST_FROM_VAL, help="Share of val-test data for test"
    )
    parser.add_argument("--random_state", default=RANDOM_STATE, help="Random state of data split")

    args = parser.parse_args()

    main(
        data_path=args.data_path,
        train_size=args.train_size,
        test_size=args.test_size,
        random_state=args.random_state,
    )
