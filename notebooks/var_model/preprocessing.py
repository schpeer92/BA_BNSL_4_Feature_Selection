import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller


def ts_differencing(df: pd.DataFrame) -> pd.DataFrame:
    df_stationary = df.copy()
    # df_stationary = np.sqrt(df_stationary)
    df_stationary = df_stationary.diff().dropna()
    return df_stationary


def undo_ts_differencing(
    df_stationary: pd.DataFrame, df_original: pd.DataFrame
) -> pd.DataFrame:
    df_unstationary = df_stationary.copy()
    df_unstationary = df_unstationary**2
    df_unstationary = df_unstationary + df_original.shift(1)
    return df_unstationary


def adf_test(df: pd.DataFrame, title=""):
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    for series in df.columns:

        print(f"Augmented Dickey-Fuller Test: {series}")
        result = adfuller(
            df[series].dropna(), autolag="AIC"
        )  # .dropna() handles differenced data
        labels = ["ADF test statistic", "p-value", "# lags used", "# observations"]
        out = pd.Series(result[0:4], index=labels)
        for key, val in result[4].items():
            out[f"critical value ({key})"] = val
        # print(out.to_string())  # .to_string() removes the line "dtype: float64"
        if result[1] <= 0.05:
            # print("Strong evidence against the null hypothesis")
            # print("Reject the null hypothesis")
            print("Data has no unit root and is stationary")
        else:
            # print("Weak evidence against the null hypothesis")
            # print("Fail to reject the null hypothesis")
            print("Data has a unit root and is non-stationary")
