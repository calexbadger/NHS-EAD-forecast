import pandas as pd
import xgboost as xgb

"""
Hyperparameters for XGBoost algorithm. Engineered to
minimise squared error and to regularise large number of
features.
"""
PARAMS = {
    "objective": "reg:squarederror",
    "max_depth": 5,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.06,
    "colsample_bylevel": 0.5,
    "colsample_bynode": 0.5,
    "min_child_weight": 25,
    "reg_alpha": 2.0,
    "reg_lambda": 2.0,
    "eval_metric": "rmse",
    "seed": 42,
}


class NHSForecaster:
    """
    Class to train and forecast NHS EAD target variable for a future 10 days. Trains 10 independent XGBoost
    models for each forecasted day, and uses each trained model to forecast out.
    """

    def __init__(self, params: dict = PARAMS, num_boost_round: int = 2000):
        self.params = params
        self.num_boost_round = num_boost_round
        self.models: dict[str, xgb.XGBClassifier] = {}

    def train_models(
        self,
        X_train_data: pd.DataFrame,
        y_train_data: pd.DataFrame,
    ) -> dict[str, xgb.XGBClassifier]:
        """
        Trains 10 separate XGBoost models using the defined hyperparameters for a specialised
        forecasting model for each forecasted day N.

        Parameters
        ----------
        X_train_data : pd.DataFrame
            The feature matrix used for training.
        y_train_data : pd.DataFrame
            The multi-step target matrix for training.


        Returns
        -------
        trained_models : dict[str, xgb.XGBClassifier]
            A dictionary containing the trained XGBoost for each forecasted day out.
        """

        self.models = {}
        for day in range(1, 11):
            target_day_var = f"estimated_avoidable_deaths_step_{day}"
            dtrain = xgb.DMatrix(X_train_data, label=y_train_data[target_day_var])
            self.models[target_day_var] = xgb.train(self.params, dtrain, num_boost_round=self.num_boost_round)

    def forecast(
        self,
        X_input_data: pd.DataFrame,
    ) -> pd.Series:
        """
        Forecasts the next 10 days from input data using the 10 separate XGBoost models.

        Parameters
        ----------
        X_input_data : pd.DataFrame
            The feature matrix used to forecast 10 days out from.
        xgb_models_trained : dict[str, xgb.XGBClassifier]
            The dictionary of trained XGBoost models for each day forecast.


        Returns
        -------
        forecast_df : pd.Series
            The forecasted values of the target variables.
        """
        dtest = xgb.DMatrix(X_input_data)
        forecast = {}

        for day in range(1, 11):
            target_day_var = f"estimated_avoidable_deaths_step_{day}"
            pred = self.models[target_day_var].predict(dtest)[0]
            forecast[f"day_{day}_forecast"] = pred

        forecast_df = pd.Series(forecast, name="fore_avoidable_deaths")
        return forecast_df
