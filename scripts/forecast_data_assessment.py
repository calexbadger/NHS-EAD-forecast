import sys
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append("../src")
from diagnostic_toolkit import mse_metric
from forecasting import create_features_and_targets
from preprocessing import load_and_preprocess_data
from feature_engineering import weekly_yearly_features, holiday_features, prepare_features
from xgboost_model import NHSForecaster


logger = logging.getLogger(__name__)

logging.basicConfig(
        level=logging.INFO,
        format="%(name)s %(levelname)s %(message)s",
    )

# NOTE: Script to load assessment data as a separate file following
# confirmation from competition organisers (A. Rabeau, email 06/06/2026).
# Original submission assumed a single updated dataset file.
logger.info("Loading dataframes.")
df_train = pd.read_csv("../data/turingAI_forecasting_challenge_dataset.csv")
df_train = df_train[df_train['value'] != -9999]
df_val = pd.read_csv("../data/turingAI_forecasting_challenge_validation_dataset.csv")
logger.info("Combining dataframes.")
df_comb = pd.concat([df_train, df_val])
logger.info("Saving dataframes.")
df_comb.to_csv("../data/turingAI_forecasting_challenge_full_dataset.csv", index=False)
logger.info("Saved combining dataframe.")

# Load data #
logger.info("Loading data.")
df = load_and_preprocess_data("../data/turingAI_forecasting_challenge_full_dataset.csv", cutoff_date = "2026-04-01")
df = df.reset_index()
df["midday_day"] = pd.to_datetime(df["midday_day"])
logger.info("Data loaded.")


ASSESSMENT_START = pd.Timestamp("2025-10-01", tz="UTC")
df_dev = df[df["midday_day"] <  ASSESSMENT_START].copy()
df_assess = df[df["midday_day"] >= ASSESSMENT_START].copy()

HORIZON = 10
n_periods = len(df_assess) - HORIZON + 1

test_data_list = []
forecast_data_list = []

for i in range(n_periods):

    forecast_start = df_assess.iloc[i]["midday_day"]
    cutoff = forecast_start

    # Training data: full development set + assessment rows up to cutoff
    df_train = pd.concat([df_dev, df_assess[df_assess["midday_day"] <= cutoff],]).copy()
    df_train["midday_day"] = df_train["midday_day"].dt.date

    df_train = weekly_yearly_features(df_train)
    df_train = holiday_features(df_train)

    X, y = create_features_and_targets(
        df_train,
        feature_lags=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 21),
        target_lags=(3, 4, 5, 6, 7, 10, 14, 21),
    )
    X_train_i = prepare_features(X, np.arange(len(X)), X.columns)

    nhs_model = NHSForecaster()
    nhs_model.train_models(X_train_i, y)

    # Forecast from last training row #
    forecasts = nhs_model.forecast(X_train_i.iloc[[-1]])

    # Store actuals and predictions as arrays #
    actuals = df_assess.iloc[i : i + HORIZON]["estimated_avoidable_deaths"].values
    test_data_list.append(np.array(actuals))
    forecast_data_list.append(np.array(forecasts[:HORIZON].values))

    logger.info("Period %d / %d complete.", i + 1, n_periods)

# Compute MSE #
mse_list, summed_mse, mse_1_5_list, summed_mse_1_5, mse_6_10_list, summed_mse_6_10 = mse_metric(
    test_data_list, forecast_data_list
)
logger.info("Overall MSE (1-10): %.4f", summed_mse)
logger.info("Overall MSE (1-5):  %.4f", summed_mse_1_5)
logger.info("Overall MSE (6-10): %.4f", summed_mse_6_10)

# Save submission files #

pred_out = pd.DataFrame(
    forecast_data_list, columns=[f"day_{d+1}" for d in range(HORIZON)]
)
pred_out.insert(0, "forecast_id", range(1, n_periods + 1))
pred_out.to_csv("pred_matrix.csv", index=False)
logger.info("Saved pred_matrix.csv")

mse_df = pd.DataFrame({
    "forecast_id": range(1, n_periods + 1),
    "mse_1_5":     mse_1_5_list,
    "mse_6_10":    mse_6_10_list,
})
mse_df.to_csv("mse_summary.csv", index=False)