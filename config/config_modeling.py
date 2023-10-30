# data processing
CAT_COLS = ["Color", "Gender", "Make", "Race", "VehicleType"]
TRAIN_SIZE = 0.6
TEST_FROM_VAL = 0.5
RANDOM_STATE = 42

# modeling
EXPERIMENT = "Final Models"
RUN_NAME = "TEST ALL - Base"
# MODEL_NAME = "Random Forest"
# PARAMS = {"max_depth": 50, "ccp_alpha": 0.001, "max_samples": 0.8}
MODELS_TO_COMPARE = [
    {
        "MODEL_NAME": "Random Forest",
        "PARAMS": {"max_depth": 50, "ccp_alpha": 0.001, "max_samples": 0.8},
    },
    {"MODEL_NAME": "Logistic Regression", "PARAMS": {"C": 1.0, "penalty": "l2", "solver": "lbfgs"}},
    {"MODEL_NAME": "Decision Tree", "PARAMS": {"max_depth": 20, "min_samples_split": 2}},
    # {
    #     "MODEL_NAME": "GAM",
    #     "PARAMS": {"max_iter": 100, "tol": 0.0001}
    # },
    {"MODEL_NAME": "XGB", "PARAMS": {"max_depth": 20, "n_estimators": 100}},
    {"MODEL_NAME": "ANN", "PARAMS": {"hidden_layer_sizes": (100,), "activation": "relu"}},
]

# inference
THRESHOLD = 0.5
