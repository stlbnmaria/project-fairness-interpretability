from pathlib import Path

# data processing
CAT_COLS = ["Color", "Gender", "Make", "Race", "VehicleType"]
TRAIN_SIZE = 0.6
TEST_FROM_VAL = 0.5
RANDOM_STATE = 42

# modeling
EXPERIMENT = "Final Models"
MODELS_PATH = Path("models/")
MODELS_TO_COMPARE = [
    {
        "RUN_NAME": "Random Forest - threshold",
        "MODEL_NAME": "Random Forest",
        "PARAMS": {
            "max_depth": [40],
            "min_samples_leaf": [5],
            "max_features": ["sqrt"],
            "ccp_alpha": [0.0001],
            "max_samples": [0.8],
        },
    },
    {
        "RUN_NAME": "Logistic Regression - threshold",
        "MODEL_NAME": "Logistic Regression",
        "PARAMS": [
            {
                "C": [0.8],
                "penalty": ["l2"],
                "solver": ["newton-cholesky"],
            },
        ],
    },
    {
        "RUN_NAME": "Decision Tree - threshold",
        "MODEL_NAME": "Decision Tree",
        "PARAMS": {
            "max_depth": [10],
            "min_samples_split": [50],
            "ccp_alpha": [0.0001],
            "max_features": ["sqrt"],
        },
    },
    {
        "RUN_NAME": "GAM - threshold",
        "MODEL_NAME": "GAM",
        "PARAMS": {"max_iter": [150], "tol": [0.005]},
    },
    {
        "RUN_NAME": "XGB - threshold",
        "MODEL_NAME": "XGB",
        "PARAMS": {
            "max_depth": [6],
            "n_estimators": [100],
            "eta": [0.1],
            "subsample": [0.9],
            "colsample_bytree": [0.8],
            "alpha": [1],
        },
    },
    {
        "RUN_NAME": "ANN - threshold",
        "MODEL_NAME": "ANN",
        "PARAMS": {
            "hidden_layer_sizes": [(100,)],
            "solver": ["adam"],
        },
    },
]

# inference
THRESHOLD = 0.5
