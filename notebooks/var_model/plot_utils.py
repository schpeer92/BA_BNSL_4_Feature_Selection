from collections import defaultdict
from textwrap import fill
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from var_model.var_bn import VAR_bn
from var_model.gobnilp_utils import shift_cols
from var_model.utils import read_pickle
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from typing import Tuple, List


CMAP = sns.diverging_palette(240, 10, n=9)


def find_one_values_adj_mat(adj_mat: np.ndarray) -> list[tuple]:
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


def coeff_matrix_no_mask(var_model: VAR_bn) -> np.ndarray:
    models = []

    for model in var_model.models:
        models.append(var_model.models[model].params)

    return np.stack(models)


def coeff_matrix_mask(var_model: VAR_bn, mask: np.ndarray) -> np.ndarray:

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


def extract_x_y_label(var_list: List[str]) -> Tuple[List[str], List[str]]:
    x_labels = [x for x in var_list if "shifted" in x]
    y_labels = [x for x in var_list if not "shifted" in x]
    return x_labels, y_labels


def reorder_index(list_a: list, list_b: list) -> list:
    """returns list of indices to reorder list_b that it is equal to list_a"""
    return [
        i
        for string_a in list_a
        for i, string_b in enumerate(list_b)
        if string_a == string_b
    ]


def plot_coeff_heatmaps(
    # coeff_matrices: list, var_list: list, adj_mat: np.ndarray, index=0,
    dataset_name: str,
    order: int,
    normalization: str = "",
):
    mask = read_pickle(
        f"../data/05_model_output/{dataset_name}_mask{normalization}_order_{order}.pkl"
    )
    var_model_mask = read_pickle(
        f"../data/05_model_output/{dataset_name}_var_mask{normalization}_order_{order}.pkl"
    )
    var_model_no_mask = read_pickle(
        f"../data/05_model_output/{dataset_name}_var_no_mask{normalization}_order_{order}.pkl"
    )

    var_list = read_pickle(
        f"../data/04_model_bn_output/{dataset_name}_var_list_order_{order}.pkl"
    )

    coeff_matrices = [
        coeff_matrix_mask(var_model_mask, mask),
        coeff_matrix_no_mask(var_model_no_mask),
    ]

    normalization_dict = {
        "": "",
        "_std": "standard scaling",
        "_minmax": "minmax scaling",
    }

    x_labels, y_labels = extract_x_y_label(var_list)
    vmin = min(coeff_matrices[0].min(), coeff_matrices[1].min())
    vmax = max(coeff_matrices[0].max(), coeff_matrices[1].max())
    fig, axn = plt.subplots(1, 2, sharex=True, sharey=True)

    fig.suptitle(f"Learned Coefficients per Model {normalization_dict[normalization]}")
    cbar_ax = fig.add_axes([0.91, 0.3, 0.03, 0.4])
    list_adj_highlights = find_one_values_adj_mat(mask.T)

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
            # annot=True,
        )
    for indices in list_adj_highlights:
        axn[0].add_patch(Rectangle(indices, 1, 1, fill=False, edgecolor="black", lw=1))


def plot_adj_matrix(adj_mat: np.ndarray, var_list: list):
    x_labels, y_labels = extract_x_y_label(var_list)
    sns.heatmap(
        adj_mat,
        vmin=0,
        vmax=1,
        linewidths=0.5,
        cbar_ax=False,
        cbar=False,
        xticklabels=x_labels,
        yticklabels=y_labels,
        cmap=CMAP,
        annot=True,
    )
    plt.title("Adjacency Matrix of trained Bayesian Network")


def plot_multiple_adj_matrices(
    adj_mat_list: List[np.ndarray],
    subplot_titles: List[str],
    var_lists: List[list],
    sup_title: str,
):
    x_labels_order, y_labels_order = extract_x_y_label(var_lists[0])
    fig, axes = plt.subplots(len(adj_mat_list), 1, sharex=True, sharey=False)
    cbar_ax = fig.add_axes([0.91, 0.3, 0.03, 0.4])
    for i, (adj_mat, var_list, normalization) in enumerate(
        zip(adj_mat_list, var_lists, subplot_titles)
    ):
        x_labels, y_labels = extract_x_y_label(var_list)
        reorder_x = reorder_index(x_labels_order, x_labels)
        reorder_y = reorder_index(y_labels_order, y_labels)
        adj_mat = np.array(adj_mat)
        adj_mat = adj_mat[:, reorder_x]
        adj_mat = adj_mat[reorder_y, :]
        sns.heatmap(
            adj_mat,
            ax=axes[i],
            vmin=0,
            vmax=1,
            linewidths=0.5,
            xticklabels=x_labels_order,
            yticklabels=y_labels_order,
            cmap=CMAP,
            annot=True,
            cbar_ax=None if i else cbar_ax,
            cbar=i == 0,
        )

        axes[i].set_title(f"Adjacency matrix for {normalization}")
    fig.suptitle(
        f"Adjacency matrices for different normalizations of the {sup_title} data set"
    )
    fig.set_size_inches(8, 8)


def plot_corr_mat(dataset_name: str, order: int = 1):
    df = pd.read_csv(f"../data/03_model_input/{dataset_name}_train.csv")
    num_cols = len(df.columns)
    for shift in range(1, order + 1):
        df = shift_cols(df, shift)
    df_corr = df[order:]
    corr = df_corr.corr()
    if order == 1:
        corr = corr.iloc[:num_cols, num_cols:]
    else:
        corr = corr.iloc[:num_cols, -(order * num_cols) :]
    # mask = np.triu(np.ones_like(corr))
    sns.heatmap(
        corr,
        # mask=mask,
        linewidths=0.5,
        cmap=CMAP,
        # annot=True,
    )
    plt.title("Correlation Matrix of all Features")


def reform_dict_to_df(dict: dict, cols: list[str]) -> pd.DataFrame:
    reformed_dict = {}
    for outerKey, innerDict in dict.items():
        for innerKey, values in innerDict.items():
            reformed_dict[(outerKey, innerKey)] = values

    return pd.DataFrame(reformed_dict, index=cols)


def calculate_error_table(
    dataset_name: str,
    order: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dict_preds = defaultdict(defaultdict)
    for normalization in ["", "_std", "_minmax"]:
        match normalization:
            case "":
                dict_preds["no normalization"]["mask"] = pd.read_csv(
                    f"../data/05_model_output/{dataset_name}_pred_mask{normalization}_order_{order}.csv"
                )
                dict_preds["no normalization"]["no mask"] = pd.read_csv(
                    f"../data/05_model_output/{dataset_name}_pred_no_mask{normalization}_order_{order}.csv"
                )
            case "_std":
                dict_preds["standard scaler"]["mask"] = pd.read_csv(
                    f"../data/05_model_output/{dataset_name}_pred_mask{normalization}_order_{order}.csv"
                )
                dict_preds["standard scaler"]["no mask"] = pd.read_csv(
                    f"../data/05_model_output/{dataset_name}_pred_no_mask{normalization}_order_{order}.csv"
                )
            case "_minmax":
                dict_preds["minmax scaler"]["mask"] = pd.read_csv(
                    f"../data/05_model_output/{dataset_name}_pred_mask{normalization}_order_{order}.csv"
                )
                dict_preds["minmax scaler"]["no mask"] = pd.read_csv(
                    f"../data/05_model_output/{dataset_name}_pred_no_mask{normalization}_order_{order}.csv"
                )

    df_test = pd.read_csv(f"../data/03_model_input/{dataset_name}_test.csv")

    cols = df_test.columns
    error_dict_mse = defaultdict(lambda: defaultdict(list))
    error_dict_mape = defaultdict(lambda: defaultdict(list))
    for normalization, dfs in dict_preds.items():
        for col in cols:
            error_dict_mse[normalization]["mask"].append(
                mean_squared_error(dfs["mask"][col], df_test[col], squared=False)
            )
            error_dict_mse[normalization]["no mask"].append(
                mean_squared_error(dfs["no mask"][col], df_test[col], squared=False)
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
