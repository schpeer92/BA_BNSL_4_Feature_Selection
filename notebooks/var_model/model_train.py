import numpy as np
import pandas as pd
from var_model.var_bn import VAR_bn
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from var_model.preprocessing import undo_ts_differencing, ts_differencing


def transform_adj_matrix(adj_matrix: np.matrix, var_list: list) -> np.matrix:
    """Transform adjacency matrix that it only contains children on the row level and parents on the column level

    Args:
        adj_matrix (np.matrix): adjacency matrix of Bayesian network row (child) to col (parent)
        var_list (list): list of variables and their order

    Returns:
        np.matrix: child to parent, children without shifted, parents with shifted
    """
    var_list = pd.Series(var_list).copy()
    shifted_cols = var_list.str.contains("_shifted")
    original_cols = ~var_list.str.contains("_shifted")
    trans_adj_matrix = adj_matrix.copy()
    trans_adj_matrix = trans_adj_matrix[shifted_cols][:, original_cols]
    for i, pred_col in enumerate(var_list[original_cols]):
        for j, feature_col in enumerate(var_list[shifted_cols]):
            if pred_col in feature_col:
                trans_adj_matrix[j, i] = 1

    return trans_adj_matrix.T


def order_cols(df: pd.DataFrame, var_list: list) -> pd.DataFrame:
    """Change order of columns depending on variable list, which is based on the trained BN"""
    df_ordered = df.copy()
    cols_wo_shifted = [col for col in var_list if not "_shifted" in col]
    df_ordered = df_ordered[cols_wo_shifted]

    return df_ordered


def normalize(df: pd.DataFrame, normalizer) -> pd.DataFrame:
    df_normalized = pd.DataFrame(
        normalizer.transform(df.to_numpy()), columns=df.columns
    )
    return df_normalized


def train_predict_save(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    filepath_input: str,
    filepath_output: str,
    dataset_name: str,
    normalization: str = "",
    differenced: int = 0,
    p: int = 1,
) -> None:
    """Train models for masking and non-masking for a given normalization. Predict the test dataframe and save the results

    Args:
        df_train (pd.DataFrame): DataFrame to train the models
        df_test (pd.DataFrame): DataFrame to predict
        filepath_input (str): Filepath to directory of input data
        filepath_output (str): Filepath to directory of output data
        dataset_name (str): "air" for "Air Quality" dataset or "prsa" for "Beijing PM2.5 Data" data set.
        normalization (str, optional): "std" for Standardscaler, "minmax" for MinMaxScaler. Defaults to "" - no normalization.
        differenced (int, optional): Order of time series differencing, that was applied to df_train. Defaults to 0.
        p (int, optional): Lag order of desired models. Defaults to 1.
    """

    # if differenced df_test has to be differenced also
    if differenced != 0:
        df_original = df_test.copy()
        df_test = ts_differencing(df_original, differenced)

    # if normalization df_test and df_train have to be rescaled
    if normalization:
        normalizer = {"std": StandardScaler(), "minmax": MinMaxScaler()}
        normalizer = normalizer[normalization]
        normalizer.fit(df_train.to_numpy())
        df_train = normalize(df_train, normalizer)
        df_test = normalize(df_test, normalizer)
        normalization = f"_{normalization}"
    # load data
    with open(
        f"{filepath_input}{dataset_name}_adj_matrix{normalization}_order_{p}.pkl", "rb"
    ) as input_file:
        adj_matrix = pickle.load(input_file)

    with open(
        f"{filepath_input}{dataset_name}_var_list{normalization}_order_{p}.pkl", "rb"
    ) as input_file:
        var_list = pickle.load(input_file)

    trans_adj_matrix = transform_adj_matrix(adj_matrix, var_list)

    # save mask for coeff matrix for plots
    with open(
        f"{filepath_output}{dataset_name}_mask{normalization}_order_{p}.pkl", "wb"
    ) as filename:
        pickle.dump(trans_adj_matrix, filename, protocol=pickle.HIGHEST_PROTOCOL)

    # GOBNILP algorithm has other order than input dataframes
    df_train = order_cols(df_train, var_list)
    df_test = order_cols(df_test, var_list)

    # train without mask
    var_model_mask = VAR_bn(
        data_train=df_train, p=p, var_list=var_list, mask=trans_adj_matrix
    )
    var_model_mask.train_model_per_col()
    # train with mask
    var_model_no_mask = VAR_bn(data_train=df_train, p=p, var_list=var_list)
    var_model_no_mask.train_model_per_col()

    # save models
    with open(
        f"{filepath_output}{dataset_name}_var_mask{normalization}_order_{p}.pkl", "wb"
    ) as filename:
        pickle.dump(var_model_mask, filename, protocol=pickle.HIGHEST_PROTOCOL)

    with open(
        f"{filepath_output}{dataset_name}_var_no_mask{normalization}_order_{p}.pkl",
        "wb",
    ) as filename:
        pickle.dump(var_model_no_mask, filename, protocol=pickle.HIGHEST_PROTOCOL)

    # predict and inverse normalization
    df_pred_mask = var_model_mask.predict_1_step_ahead_per_col(df_test)
    df_pred_no_mask = var_model_no_mask.predict_1_step_ahead_per_col(df_test)

    # inverse if normalized
    if normalization:
        df_pred_mask = pd.DataFrame(
            normalizer.inverse_transform(df_pred_mask.to_numpy()),
            columns=df_pred_mask.columns,
        )
        df_pred_no_mask = pd.DataFrame(
            normalizer.inverse_transform(df_pred_no_mask.to_numpy()),
            columns=df_pred_no_mask.columns,
        )
    # undo differencing on predictions for accuracy tests with original data
    if differenced != 0:
        df_pred_mask = undo_ts_differencing(df_pred_mask, df_original, differenced)
        df_pred_no_mask = undo_ts_differencing(
            df_pred_no_mask, df_original, differenced
        )

    # save predictions
    df_pred_mask.to_csv(
        f"{filepath_output}{dataset_name}_pred_mask{normalization}_order_{p}.csv",
        index=False,
    )
    df_pred_no_mask.to_csv(
        f"{filepath_output}{dataset_name}_pred_no_mask{normalization}_order_{p}.csv",
        index=False,
    )
