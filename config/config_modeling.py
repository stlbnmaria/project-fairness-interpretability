# data processing
CAT_COLS = ["Color", "Gender", "Make", "Race", "VehicleType"]
TRAIN_SIZE = 0.6
TEST_FROM_VAL = 0.5
RANDOM_STATE = 42

# modeling
EXPERIMENT = "Tuning"
MODELS_TO_COMPARE = [
    {
        "RUN_NAME": "Tuning Run 1.",
        "MODEL_NAME": "Random Forest",
        "PARAMS": {"max_depth": [30, 50], "ccp_alpha": [0.0001, 0.001], "max_samples": [0.7, 0.8]},
    },
    {
        "RUN_NAME": "Tuning Run 1.",
        "MODEL_NAME": "Logistic Regression",
        "PARAMS": {"C": [0.9, 1.0], "penalty": ["l2"], "solver": ["lbfgs"]},
    },
    {
        "RUN_NAME": "Tuning Run 1.",
        "MODEL_NAME": "Decision Tree",
        "PARAMS": {"max_depth": [10, 20], "min_samples_split": [2]},
    },
    {
        "RUN_NAME": "Tuning Run 1.",
        "MODEL_NAME": "GAM",
        "PARAMS": {"max_iter": [100], "tol": [0.0001]},
    },
    {
        "RUN_NAME": "Tuning Run 1.",
        "MODEL_NAME": "XGB",
        "PARAMS": {"max_depth": [20, 30], "n_estimators": [100]},
    },
    {
        "RUN_NAME": "Tuning Run 1.",
        "MODEL_NAME": "ANN",
        "PARAMS": {
            "hidden_layer_sizes": [
                (100,),
                (
                    80,
                    40,
                ),
            ],
            "activation": ["relu"],
        },
    },
]

# inference
THRESHOLD = 0.5
