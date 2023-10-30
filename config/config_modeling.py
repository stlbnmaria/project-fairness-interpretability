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
        "PARAMS": {
            "max_depth": [10, 20, 30, 40, 50],
            "min_samples_leaf": [5, 10, 20, 100],
            "max_features": ["log2", "sqrt"],
            "ccp_alpha": [0.0001, 0.005, 0.001],
            "max_samples": [0.7, 0.8, 0.9, 1],
        },
    },
    {
        "RUN_NAME": "Tuning Run 1.",
        "MODEL_NAME": "Logistic Regression",
        "PARAMS": [
            {
                "C": [0.8, 0.9, 1.0, 1.1],
                "penalty": ["l2"],
                "solver": ["lbfgs", "newton-cholesky"],
            },
            {
                "C": [0.8, 0.9, 1.0, 1.1],
                "penalty": ["l2", "l1"],
                "solver": ["saga"],
            },
            {
                "C": [0.8, 0.9, 1.0, 1.1],
                "penalty": ["elasticnet"],
                "solver": ["saga"],
                "l1_ratio": [0.3, 0.5, 0.7],
            },
        ],
    },
    {
        "RUN_NAME": "Tuning Run 1.",
        "MODEL_NAME": "Decision Tree",
        "PARAMS": {
            "max_depth": [5, 10, 15, 20],
            "min_samples_split": [10, 20, 30, 40, 50],
            "ccp_alpha": [0.0001, 0.005, 0.001],
            "max_features": ["log2", "sqrt", 1],
        },
    },
    {
        "RUN_NAME": "Tuning Run 1.",
        "MODEL_NAME": "GAM",
        "PARAMS": {"max_iter": [50, 100, 150], "tol": [0.001, 0.005, 0.0001]},
    },
    {
        "RUN_NAME": "Tuning Run 1.",
        "MODEL_NAME": "XGB",
        "PARAMS": {
            "max_depth": [6, 8, 10],
            "n_estimators": [100, 200, 300],
            "eta": [0.1, 0.2, 0.3],
            "subsample": [0.7, 0.8, 0.9],
            "colsample_bytree": [0.4, 0.6, 0.8, 1],
            "alpha": [0, 1],
        },
    },
    {
        "RUN_NAME": "Tuning Run 1.",
        "MODEL_NAME": "ANN",
        "PARAMS": {
            "hidden_layer_sizes": [
                (100,),
                (
                    120,
                    60,
                    30,
                ),
                (
                    120,
                    60,
                    30,
                    15,
                ),
            ],
            "solver": ["lbfgs", "sgd", "adam"],
        },
    },
]

# inference
THRESHOLD = 0.5
