from pathlib import Path

DATA_PATH = Path("data") / "file65ef3a759daf.arff"
DROP_COLS = [
    "Description",
    "Model",
    "Charge",
    "Driver.City",
    "Arrest.Type",
    "Commercial.Vehicle",
    "description_clean",
]
OUT_PATH = Path("data") / "data.csv"
DICT_PATH = Path("config") / "dict.yaml"
DICT_H_PATH = Path("config") / "dict_hard.yaml"
N_CATEGORIES = 15
N_TOPICS = 3
THRESHOLD = 0.334
NLP_FEATURE_COLS = ["Fail.Obey.Signals", "Speeding", "Obey.Police.Doc.Light"]
