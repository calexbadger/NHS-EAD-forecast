import logging

from preprocessing import load_and_preprocess_data

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(name)s %(levelname)s %(message)s",
)

# Full data range is from '2023-03-16 00:00:00+00:00' to '2025-09-30 00:00:00+00:00'
# Select a data cut-off where all data after this date is to be separated away from
# the remaining set for future testing.

test_data_date = "2025-05-16 00:00:00+00:00"

df = load_and_preprocess_data("../data/turingAI_forecasting_challenge_dataset.csv")

# Section off from date, save
df_validation = df.loc[test_data_date:]
logger.info("Saving testing data (last %.2f precent of data)", (df_validation.size / df.size) * 100.0)
df_validation.to_csv("nhs_ead_forecast_testing_dataset.csv")

# Save remaining data for future easy data loading to be sued for training and validation.
df_train_valid = df.loc[:test_data_date]
logger.info(
    "Saving training and validation data (first %.2f precent of data)", (1 - (df_validation.size / df.size)) * 100.0
)
df_train_valid.to_csv("nhs_ead_forecast_training_validation_dataset.csv")
