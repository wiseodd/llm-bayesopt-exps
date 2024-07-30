import dill
import pandas as pd
import numpy as np


def dill_save(obj, filepath):
    with open(filepath, "wb") as f:
        dill.dump(obj, f)


def dill_load(filepath):
    with open(filepath, "rb") as f:
        obj = dill.load(f)
    return obj


def pop_df(df: pd.DataFrame, idx: int):
    row = df.loc[idx]
    df.drop(idx, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return row


def y_unscale(y, scaler):
    return scaler.inverse_transform(np.array(y).reshape(-1, 1)).item()


def y_transform(y, maximize: bool):
    """
    Negating y if maximize=False.
    Rationale: we define the BayesOpt as a maximization problem.
    """
    return y if maximize else -y
