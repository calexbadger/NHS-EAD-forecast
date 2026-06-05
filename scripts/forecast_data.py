import sys
import logging

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.append("../src")
from diagnostic_toolkit import (
    full_metric_analysis,
    mae_metric,
    mase_metric,
    mbe_metric,
    mse_metric,
    plot_results,
)
from forecasting import create_features_and_targets
from preprocessing import load_and_preprocess_data
from feature_engineering import weekly_yearly_features, holiday_features, prepare_features
from xgboost_model import NHSForecaster


logger = logging.getLogger(__name__)

logging.basicConfig(
        level=logging.INFO,
        format="%(name)s %(levelname)s %(message)s",
    )

# Load data #
logger.info("Loading data.")
df = load_and_preprocess_data("../data/turingAI_forecasting_challenge_dataset.csv")
df = df.reset_index()
logger.info("Data loaded.")

# Feature selection #
forecasting_labels = list(df.columns.values)
target_label = ["estimated_avoidable_deaths"]
if forecasting_labels is None:
    forecasting_df = df[["midday_day"] + target_label]
    forecasting_exog = None
else:
    forecasting_df = df  # df[["midday_day"] + forecasting_labels + target_label]
    forecasting_exog = df[forecasting_labels]
forecasting_df["midday_day"] = pd.to_datetime(forecasting_df["midday_day"]).dt.date

forecasting_df = weekly_yearly_features(forecasting_df)
forecasting_df = holiday_features(forecasting_df)

# Generate feature and target lags #
X, y = create_features_and_targets(
    forecasting_df, feature_lags=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 21), target_lags=(3, 4, 5, 6, 7, 10, 14, 21)
)

logger.info("Feature engineering complete and lag matrices created.")

# Select whole data set for training #
idx = np.arange(len(X))
X_train = prepare_features(X, idx, X.columns)
y_train = y

# Initialise forecaster #
logger.info("Training forecaster.")
nhs_model = NHSForecaster()
trained_models = nhs_model.train_models(X_train, y_train)
# Save trained forecasting models
joblib.dump(trained_models, "forecasting_models.pkl")
logger.info("Forecasted models trained and saved.")

# Forecasting from final training data values #
todays_values = X_train.iloc[[-1]]
forecast_df = nhs_model.forecast(todays_values)

logger.info("Forecast complete.")
logger.info("-" * 35)
for step, val in enumerate(forecast_df, start=1):
    logger.info("  Forecast Day D+%d: Target Variable = %.4f", step, val)