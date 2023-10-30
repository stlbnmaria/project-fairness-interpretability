import argparse
from pathlib import Path
from typing import List, Tuple, Union

import mlflow
from create_data_split import split_data
from evaluation import eval_metrics
from inference import inference
from training import training

from config.config_data import OUT_PATH
from config.config_modeling import (
    CAT_COLS,
    EXPERIMENT,
    MODELS_TO_COMPARE,
    RANDOM_STATE,
    TEST_FROM_VAL,
    THRESHOLD,
    TRAIN_SIZE,
)


def main(
    data_path: Path,
    experiment: str,
    run_name: str,
    cat_cols: List,
    train_size: float,
    test_size: float,
    random_state: int,
    model_name: str,
    params: dict[str, Union[float, int, bool, Tuple[int]]],
    threshold: float,
) -> None:
    # validate that mlflow runs locally
    print(f"tracking URI: '{mlflow.get_tracking_uri()}'")

    # set experiment
    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name=run_name):
        # dummy encoding and data split
        data = split_data(
            cols=cat_cols,
            data_path=data_path,
            train_size=train_size,
            test_size=test_size,
            random_state=random_state,
        )

        # training
        model = training(
            model_name=model_name, X_train=data["train"][0], y_train=data["train"][1], params=params
        )
        mlflow.log_params(params)

        # predictions and evalutation
        for state in ["train", "val", "test"]:
            preds_prob = inference(model=model, X=data[state][0])
            preds = preds_prob > threshold
            metrics = eval_metrics(data[state][1], preds, preds_prob)
            metrics = {state + "_" + k: v for k, v in metrics.items()}
            mlflow.log_metrics(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", default=OUT_PATH, help="Path to data")
    parser.add_argument("--experiment", default=EXPERIMENT, help="Experiment name for Mlflow")
    parser.add_argument(
        "--cat_cols", default=CAT_COLS, help="Categorical columns for dummy encoding"
    )
    parser.add_argument("--train_size", default=TRAIN_SIZE, help="Share of data for train")
    parser.add_argument(
        "--test_size", default=TEST_FROM_VAL, help="Share of val-test data for test"
    )
    parser.add_argument("--random_state", default=RANDOM_STATE, help="Random state of data split")
    parser.add_argument(
        "--threshold", default=THRESHOLD, help="Threshold for probability predictions"
    )

    args = parser.parse_args()

    for model_config in MODELS_TO_COMPARE:
        main(
            data_path=args.data_path,
            experiment=args.experiment,
            run_name=model_config["RUN_NAME"],
            cat_cols=args.cat_cols,
            train_size=args.train_size,
            test_size=args.test_size,
            random_state=args.random_state,
            model_name=model_config["MODEL_NAME"],
            params=model_config["PARAMS"],
            threshold=args.threshold,
        )
