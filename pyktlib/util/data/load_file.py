import json
import pandas as pd


def load_csv(data_path, useful_cols=None, rename_dict=None, num_rows=None):
    try:
        df = pd.read_csv(data_path, usecols=useful_cols, encoding="utf-8", low_memory=False, index_col=False, nrows=num_rows)
    except UnicodeDecodeError:
        df = pd.read_csv(data_path, usecols=useful_cols, encoding="ISO-8859-1", low_memory=False, index_col=False, nrows=num_rows)
    if rename_dict is not None:
        df.rename(columns=rename_dict, inplace=True)
    return df


def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        result = json.load(f)
    return result
