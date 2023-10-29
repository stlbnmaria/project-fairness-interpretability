# data processing
CAT_COLS = ["Color", "Gender", "Make", "Race", "VehicleType"]
TRAIN_SIZE = 0.6
TEST_FROM_VAL = 0.5
RANDOM_STATE = 42

# modeling
EXPERIMENT = "Final Models"
RUN_NAME = "Random Forest - Base"
MODEL_NAME = "Random Forest"
PARAMS = {"max_depth": 50, "ccp_alpha": 0.001, "max_samples": 0.8}

# inference
THRESHOLD = 0.5
