import pandas as pd
import numpy as np
from pygobnilp.gobnilp import Gobnilp
from typing import Tuple
import pickle


def forbidden_backward_arrow(par_cols: list, ch_cols: list, model: Gobnilp) -> None:
    for child in ch_cols:
        for parent in par_cols:
            model.add_forbidden_ancestor(child, parent)


def delete_arrows_in_generation(gen_cols: list, model: Gobnilp) -> None:
    for i, col_i in enumerate(gen_cols):
        if i < len(gen_cols):
            for col_j in gen_cols[i + 1 :]:
                model.add_forbidden_adjacency((col_i, col_j))
                # model.add_forbidden_adjacency((f"{col_i}_shifted", f"{col_j}_shifted"))
        else:
            next


def shift_cols(df: pd.DataFrame, suffix: int = 1) -> pd.DataFrame:
    df_shifted = df.copy()
    cols_to_shift = [col for col in df.columns if not "_shifted" in col]
    for col in cols_to_shift:
        df_shifted[f"{col}_shifted_{suffix}"] = df[col].shift(1)
    return df_shifted.iloc[1:]


def train_bn(df: pd.DataFrame, p: int = 1) -> Tuple[list, np.matrix]:
    df_train = df.copy()
    for i in range(1, p + 1):
        df_train = shift_cols(df_train, i)
    # save dataset
    df_train.to_csv("df_train_bn.csv", index=False, sep=" ")
    # init model
    par_cols = [col for col in df_train.columns if "_shifted" in col]
    ch_cols = [col for col in df_train.columns if not "_shifted" in col]

    model = Gobnilp()
    model.learn("df_train_bn.csv", end="data", data_type="continuous", score="BGe")

    # delete adjacency in each generation
    delete_arrows_in_generation(par_cols, model)
    delete_arrows_in_generation(ch_cols, model)
    # delete backwards adjacency
    forbidden_backward_arrow(par_cols, ch_cols, model)
    # learn BN
    model.learn(
        "df_train_bn.csv", data_type="continuous", score="BGe", palim=None, end="BN(s)"
    )
    # variable order is different from dataframe
    variable_list = [col[0] for col in list(model.learned_bn.adjacency())]
    adjacency_matrix = model.learned_bn.adjacency_matrix()

    return variable_list, adjacency_matrix, model
