import logging

import pandas as pd

logger = logging.getLogger(__name__)


def filter_and_shift_to_midday_dates(df: pd.DataFrame, cutoff_date: str = "2025-09-30") -> pd.DataFrame:
    """
    Parses timestamps and applies an operational midday boundary.

    Filters records exceeding the cutoff date and shifts the temporal
    reference to midday to align with D+1 to D+10 forecast availability
    constraints.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing a 'dt' column.
    cutoff_date : str, default="2025-09-30"
        The date threshold for filtering the assessment dataset.

    Returns
    -------
    pd.DataFrame
        Dataframe with UTC-localised timestamps and a new 'midday_day' column.
    """

    df["dt"] = pd.to_datetime(df["dt"], format="ISO8601", utc=True)

    # Filter out assessment dataset
    cutoff = pd.Timestamp(cutoff_date, tz="UTC")
    df = df.loc[df["dt"] <= cutoff].copy()  # .copy() avoids SettingWithCopyWarning later

    # Data cleaning: Midday threshold
    df["midday_day"] = (df["dt"] - pd.Timedelta(hours=12)).dt.ceil("D")

    return df


def clean_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardises feature metadata and converts metric names to snake_case.

    Expands abbreviations, replaces mathematical symbols with text descriptors,
    and enforces a consistent naming convention across 'coverage_label'
    and 'metric_name'.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing raw feature and metric names.

    Returns
    -------
    pd.DataFrame
        Dataframe with cleaned and standardised string columns.
    """

    str_replacements = {
        r"(\d)mins": r"\1 mins",
        r"no\. of": "number of",  # escaped
        ">=": " gte ",
        "<=": " lte ",
        ">": " gt ",
        "<": " lt ",
        "%": " pct ",
        "&": " and ",
        r"\+": " plus ",  # escaped
        "-": " ",
    }

    df[["coverage_label", "coverage"]] = df[["coverage_label", "coverage"]].replace("SWAST", "SWASFT")

    unique_metrics = pd.Series(df["metric_name"].unique())

    cleaned_metrics = (
        unique_metrics.str.lower()
        .replace(str_replacements, regex=True)
        # strictly enforcing the order of the operations
        .str.replace(r"[^\w\s]", "", regex=True)  # drops brackets, commas, quotes
        .str.replace(r"\s+", "_", regex=True)
        .str.strip("_")
    )

    metric_map = dict(zip(unique_metrics, cleaned_metrics, strict=False))

    df["metric_name"] = df["metric_name"].map(metric_map)

    return df


def aggregate_daily_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates metric values to daily means based on the midday boundary.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing 'midday_day', 'metric_name', and 'value'.

    Returns
    -------
    pd.DataFrame
        Long-format dataframe grouped by day and metric.
    """
    aggregated_df = (
        df[["midday_day", "metric_name", "value"]]
        .groupby(["midday_day", "metric_name"])["value"]
        .mean()
        .reset_index()
        .sort_values(by=["metric_name", "midday_day"], ignore_index=True)
    )

    return aggregated_df


def to_wide_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms the dataset from long format to a wide feature matrix.

    Pivots unique metric names into individual columns indexed by the
    midday operational date.

    Parameters
    ----------
    df : pd.DataFrame
        Aggregated long-format dataframe.

    Returns
    -------
    pd.DataFrame
        Wide-format dataframe with metrics as columns.
    """
    wide_df = (
        df.pivot_table(
            index="midday_day",
            columns="metric_name",
            values="value",
        ).rename_axis(columns=None)
        # .reset_index()
    )

    return wide_df


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Addresses missing values within the feature matrix.

    Currently implements a forward-fill (ffill) strategy to propagate
    the last known valid observation.

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format dataframe containing potential NaNs.

    Returns
    -------
    pd.DataFrame
        Imputed dataframe.
    """
    return df.ffill()


def load_and_preprocess_data(filepath: str) -> pd.DataFrame:
    """
    Orchestrates the end-to-end data preprocessing pipeline.

    Loads raw CSV data, cleans feature names, applies the midday operational
    boundary, aggregates to daily means, pivots to wide format, and imputes
    missing values.

    Parameters
    ----------
    filepath : str
        Path to the raw CSV dataset.

    Returns
    -------
    pd.DataFrame
        Fully preprocessed wide-format feature matrix.
    """

    logger.info("Loading raw data from: %s", filepath)
    df = pd.read_csv(filepath)

    logger.info("Filtering dates and applying midday boundary...")
    df = filter_and_shift_to_midday_dates(df)

    logger.info("Cleaning and standardising feature names...")
    df = clean_feature_names(df)

    logger.info("Aggregating metrics to daily means...")
    agg_df = aggregate_daily_metrics(df)

    logger.info("Pivoting data to wide format...")
    wide_df = to_wide_format(agg_df)

    wide_df = impute_missing_values(wide_df)

    logger.info("Preprocessing complete. Final shape: %s", wide_df.shape)

    return wide_df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s %(levelname)s %(message)s",
    )

    wide_df = load_and_preprocess_data("../data/turingAI_forecasting_challenge_dataset.csv")
