import pandas as pd
import os

__all__ = ["write_csv", "save_data_csv"]


def write_csv(file_path, exp_name, data, index_names):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df[exp_name] = data
    else:
        df = pd.DataFrame(data, columns=[exp_name], index=index_names)
    df.to_csv(file_path)


def save_data_csv(file_path, data, label, column_names=[], label_name="label"):
    data_df = pd.DataFrame(data, columns=column_names)
    label_df = pd.DataFrame(label, columns=[label_name])
    df = pd.concat([label_df, data_df], axis=1)
    df.to_csv(file_path, index=False)
