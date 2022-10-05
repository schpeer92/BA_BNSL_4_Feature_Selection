from collections import defaultdict
from textwrap import fill
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from var_model.var_bn import VAR_bn
from var_model.gobnilp_utils import shift_cols
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from typing import Tuple


CMAP = sns.diverging_palette(240, 10, n=9)


def find_one_values_adj_mat(adj_mat: np.array) -> list[tuple]:
    """generate a list of all indices where the matrix has one values of a adjacency matrix

    Args:
        adj_mat(np.array): 2D adjacency matrix

    Returns:
        List[tuple]: tuples contain indices of one values
    """
    adj_mat = np.array(adj_mat)
    return [
        (i, j)
        for i in range(len(adj_mat))
        for j in range(len(adj_mat[i]))
        if adj_mat[i][j] == 1
    ]


def coeff_matrix_no_mask(var_model: VAR_bn) -> np.array:
    models = []

    for model in var_model.models:
        models.append(var_model.models[model].params)

    return np.stack(models)


def coeff_matrix_mask(var_model: VAR_bn, mask: np.matrix) -> np.matrix:
    values = []
    for model, coeff_mask in zip(var_model.models, mask):
        i = 0
        params = var_model.models[model].params
        coeff_mask = np.array(coeff_mask[0])[0]
        for j, item in enumerate(coeff_mask):
            if item > 0:
                coeff_mask[j] = params[i]
                i += 1
            else:
                next
        values.append(coeff_mask)

    return np.stack(values)


def extract_x_y_label(var_list: list):
    x_labels = [x for x in var_list if "shifted" in x]
    y_labels = [x for x in var_list if not "shifted" in x]
    return x_labels, y_labels


def plot_coeff_heatmaps(
    coeff_matrices: list, var_list: list, adj_mat: np.matrix, index=0
):
    x_labels, y_labels = extract_x_y_label(var_list)
    vmin = min(coeff_matrices[0].min(), coeff_matrices[1].min())
    vmax = max(coeff_matrices[0].max(), coeff_matrices[1].max())
    fig, axn = plt.subplots(1, 2, sharex=True, sharey=True)
    cbar_ax = fig.add_axes([0.91, 0.3, 0.03, 0.4])
    list_adj_highlights = find_one_values_adj_mat(adj_mat.T)

    for i, ax in enumerate(axn.flat):
        sns.heatmap(
            coeff_matrices[i],
            ax=ax,
            cbar=i == 0,
            vmin=vmin,
            vmax=vmax,
            cbar_ax=None if i else cbar_ax,
            yticklabels=y_labels,
            xticklabels=x_labels,
            linewidths=0.5,
            cmap=CMAP,
        )
        if i == index:
            for indices in list_adj_highlights:
                ax.add_patch(
                    Rectangle(indices, 1, 1, fill=False, edgecolor="red", lw=1)
                )


def plot_adj_matrix(adj_mat: np.array, var_list: list):
    x_labels, y_labels = extract_x_y_label(var_list)
    sns.heatmap(
        adj_mat,
        vmin=0,
        vmax=1,
        linewidths=0.5,
        xticklabels=x_labels,
        yticklabels=y_labels,
        cmap=CMAP,
    )


def plot_corr_mat(df: pd.DataFrame):
    num_cols = len(df.columns)
    df_corr = shift_cols(df)
    corr = df_corr.corr()
    corr = corr.iloc[:num_cols, num_cols:]
    mask = np.triu(np.ones_like(corr))
    sns.heatmap(corr, mask=mask, linewidths=0.5, cmap=CMAP)


def reform_dict_to_df(dict: dict, cols: list[str]) -> pd.DataFrame:
    reformed_dict = {}
    for outerKey, innerDict in dict.items():
        for innerKey, values in innerDict.items():
            reformed_dict[(outerKey, innerKey)] = values

    return pd.DataFrame(reformed_dict, index=cols)


def calculate_error_table(
    dict_preds: dict, df_test
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cols = df_test.columns
    error_dict_mse = defaultdict(lambda: defaultdict(list))
    error_dict_mape = defaultdict(lambda: defaultdict(list))
    for normalization, dfs in dict_preds.items():
        for col in cols:
            error_dict_mse[normalization]["mask"].append(
                mean_squared_error(dfs["mask"][col], df_test[col])
            )
            error_dict_mse[normalization]["no mask"].append(
                mean_squared_error(dfs["no mask"][col], df_test[col])
            )
            error_dict_mape[normalization]["mask"].append(
                mean_absolute_percentage_error(dfs["mask"][col], df_test[col])
            )
            error_dict_mape[normalization]["no mask"].append(
                mean_absolute_percentage_error(dfs["no mask"][col], df_test[col])
            )

    df_mse_error = reform_dict_to_df(error_dict_mse, cols)
    df_mape_error = reform_dict_to_df(error_dict_mape, cols)

    return df_mse_error, df_mape_error


def plot_ts(
    df_mask: pd.DataFrame,
    df_no_mask: pd.DataFrame,
    df_test: pd.DataFrame,
    col: str,
    time_steps: int = 100,
):
    df_all = df_test.join(df_mask, rsuffix="_mask")
    df_all = df_all.join(df_no_mask, rsuffix="_no_mask")
    df_lineplot = df_all[
        [column for column in df_all.columns if column.startswith(f"{col}")]
    ].copy()
    df_lineplot = df_lineplot.iloc[:time_steps]
    sns.lineplot(data=df_lineplot)
    plt.title(f"forecast of first {time_steps} time steps of column '{col}' ")
