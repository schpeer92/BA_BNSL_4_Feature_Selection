import numpy as np
import pandas as pd
import statsmodels.api as sm


class VAR_bn:
    def __init__(
        self,
        data_train: pd.DataFrame,
        p: int,
        mask: np.array = None,
    ) -> None:
        """
        Args:
            data_train (pd.DataFrame): training subset of Time Series dataset
            p (int): number of time lag variables - order of lag
            mask (np.array, optional): masking of variables to learn linear regression. default none, will take all variables
        """

        self.data_train = data_train

        if mask is None:
            self.mask = np.ones((self.data_train.shape[1], self.data_train.shape[1]))
        else:
            self.mask = mask
        self.p = p
        self.models = {}

    def insert_shifted_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Takes Dataframe and adds time lag features for all columns depending on the mask

        Args:
            data (pd.DataFrame): Dataframe of which we build time lag features

        Returns:
            pd.DataFrame: DataFrame with additional time lag features
        """
        data_shifted = data.copy()
        cols_to_shift = data.columns
        for shift in range(1, self.p + 1):
            for col in cols_to_shift:
                data_shifted.loc[:, f"{col}_{shift}_shifted"] = data.loc[:, col].shift(
                    shift
                )

        return data_shifted

    def train_model_per_col(self) -> None:
        """Runs for each column multivariate linear regression by only taking into account features that are unmasked"""
        for i, col in enumerate(self.data_train.columns):
            mask = np.squeeze(np.asarray(self.mask[i, :]) > 0)
            data = self.data_train.iloc[:, mask].copy()
            #
            data_shifted = self.insert_shifted_data(data)
            data_shifted = data_shifted[
                [col for col in data_shifted.columns if col.endswith("_shifted")]
            ].copy()
            data_shifted = data_shifted[self.p :].copy()
            col_model = sm.OLS(
                self.data_train[self.p :][col].values, data_shifted.values
            ).fit()
            self.models[col] = col_model

    def predict_1_step_ahead_per_col(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run prediction on DataFrame

        Args:
            data (pd.DataFrame): DataFrame to predict, has to have at least p rows and datetime index

        Returns:
            pd.DataFrame: dataframe with predictions and actual values
        """
        data_cols = data.columns
        data_pred = data.copy()
        data_train = pd.concat([self.data_train.tail(self.p), data_pred])
        for i, col in enumerate(self.data_train.columns):
            data_pred[f"{col}_pred"] = None
            mask = np.squeeze(np.asarray(self.mask[i, :]) > 0)
            data_shifted = self.insert_shifted_data(data_train.iloc[:, mask])
            data_shifted = data_shifted.fillna(0)
            data_shifted = data_shifted[
                [col for col in data_shifted.columns if col.endswith("_shifted")]
            ]
            data_pred[f"{col}_pred"] = self.models[col].predict(data_shifted.values)[1:]
        data_pred = data_pred.drop(columns=data_cols)
        data_pred.columns = data_cols
        return data_pred

    def recursive_prediction(self, prediction_steps: int = 7) -> pd.DataFrame:
        """runs prediction_steps many iterations of recursive forecasting which uses iteratively prediction values as input

        Args:
            prediction_steps (int, optional):number of prediction steps Defaults to 7.

        Returns:
            pd.DataFrame: Dataframe with predictions as columns for prediction_steps many values
        """
        data = self.data_train.copy()
        pred_cols = data.columns
        for prediction_step in range(prediction_steps):
            data_pred = pd.concat(
                [
                    data.tail(self.p),
                    pd.DataFrame.from_dict({col: [0] for col in pred_cols}),
                ]
            )
            data_pred = self.predict_1_step_ahead_per_col(data_pred)
            data_cols = data.columns
            data_pred.drop(columns=data_cols, inplace=True)
            data_pred.columns = data_cols
            data = pd.concat([data, data_pred.tail(1)])

        return data.tail(prediction_steps)
