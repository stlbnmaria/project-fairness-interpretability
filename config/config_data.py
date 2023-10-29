from pathlib import Path

DATA_PATH = Path("data") / "file65ef3a759daf.arff"
DROP_COLS = ["Description", "Model", "Charge", "Driver.City", "Arrest.Type"]
OUT_PATH = Path("data") / "data.csv"
M_DICT_PATH = Path("config") / "make_dict.yaml"
M_DICT_H_PATH = Path("config") / "make_dict_hard.yaml"
N_CATEGORIES = 20
