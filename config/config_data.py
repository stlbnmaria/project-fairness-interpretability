from pathlib import Path

DATA_PATH = Path("data") / "file65ef3a759daf.arff"
DROP_COLS = ["Description", "Model", "Charge", "Driver.City", "Arrest.Type", "Commercial.Vehicle"]
DICT_PATH = Path("config") / "dict.yaml"
DICT_H_PATH = Path("config") / "dict_hard.yaml"
N_CATEGORIES = 15
