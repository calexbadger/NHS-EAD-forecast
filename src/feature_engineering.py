import numpy as np
import pandas as pd
from govuk_bank_holidays.bank_holidays import BankHolidays

HOL_WINDOW = 3

def weekly_yearly_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Quantifies seasonal and weekend behaviours. Specifically models
    sinusoidal weekly and yearly behaviour, marks whether it is a
    weekend or not, the day of the year.

    Parameters
    ----------
    df : pd.DataFrame
        The input matrix to be transformed.


    Returns
    -------
    subset : pd.DataFrame
        The resulting subset data frame that has been feature engineered.
    """

    # Transform midday_day into numeric calendar features
    dt = pd.to_datetime(df["midday_day"])
    df["is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)
    df["dayofyear"] = dt.dt.dayofyear

    df["sin_weekly"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
    df["sin_yearly"] = np.sin(2 * np.pi * dt.dt.dayofyear / 365.25)

    return df


def _is_bridge_day(date, holiday_dates_set):
    """
    Identifies bridge days.

    Parameters
    ----------
    date : pd.Timestamp
        The individual holiday date dataframe to be labeled.
    holiday_dates_set: set of pd.Timestamps
        The dataframe of holiday dates.

    Returns
    -------
    prev_is_off and next_is_off : bool
        True if `date` is a bridge day, False otherwise.
    """
    if date.weekday() >= 5:
        return False
    prev_day = date - pd.Timedelta(days=1)
    next_day = date + pd.Timedelta(days=1)
    prev_is_off = prev_day.weekday() >= 5 or prev_day in holiday_dates_set
    next_is_off = next_day.weekday() >= 5 or next_day in holiday_dates_set
    return prev_is_off and next_is_off


def holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies holidays, approximate holiday periods, with a specific
    marker for the Christmas and NYE period. Assumes the holiday periods
    are 3 days around holidays specifically.

    Parameters
    ----------
    df : pd.DataFrame
        The input matrix to be transformed.


    Returns
    -------
    subset : pd.DataFrame
        The resulting subset data frame that has been feature engineered.
    """

    # Loads holidays #
    bank_holidays = BankHolidays(use_cached_holidays=True)
    holidays = bank_holidays.get_holidays(division=BankHolidays.ENGLAND_AND_WALES)
    holiday_map = {h["date"]: h["title"] for h in holidays}

    date_range = pd.to_datetime(df["midday_day"])

    df_hol = pd.DataFrame({"date": date_range})
    df_hol["is_holiday"] = df_hol["date"].dt.date.map(holiday_map).notna().astype(int)

    # Flag holiday period (±3 day window around each holiday) #
    holiday_dates = pd.to_datetime(df_hol.loc[df_hol["is_holiday"] == 1, "date"])
    holiday_dates_set = set(holiday_dates)

    in_window = pd.Series(False, index=df_hol.index)
    for hdate in holiday_dates:
        mask = (df_hol["date"] >= hdate - pd.Timedelta(days=HOL_WINDOW)) & (
            df_hol["date"] <= hdate + pd.Timedelta(days=HOL_WINDOW)
        )
        in_window |= mask

    bridge_days = df_hol["date"].apply(lambda d: _is_bridge_day(d, holiday_dates_set))

    df_hol["is_holiday_period"] = (in_window | bridge_days).astype(int)

    # Christmas/NYE season flag: Dec 10 – Jan 10 (independent of holiday period)
    df_hol["is_xmas_nye_season"] = (
        ((df_hol["date"].dt.month == 12) & (df_hol["date"].dt.day >= 10))
        | ((df_hol["date"].dt.month == 1) & (df_hol["date"].dt.day <= 10))
    ).astype(int)

    df_hol = df_hol.drop(columns=["date"])
    df = pd.merge(df, df_hol, left_index=True, right_index=True)

    return df


def prepare_features(df: pd.DataFrame, idx: int, feature_cols: list[str]):
    """
    Selects subset of dataframe given inputted feature columns, and feature
    engineers input dataframe to more granular time stamps, specifically:
    - Trend indicators.
    - Target rolling means, acceleration, and weekly changes.

    Parameters
    ----------
    df : pd.DataFrame
        The input matrix to be transformed.
    idx : int
        The target index that will be transformed.
    feature_cols: list[str]
        String of feature columns to be selected in transformation (assumes
        "midday_day" will be included.)


    Returns
    -------
    subset : pd.DataFrame
        The resulting subset data frame that has been feature engineered.
    """
    subset = df.iloc[idx][feature_cols].copy()

    # Find all midday_day lag columns
    date_cols = [c for c in subset.columns if "midday_day" in c]

    # Keep only lag_0, rename it to midday_day for clarity
    subset = subset.rename(columns={"midday_day_lag_0": "midday_day"})

    # Drop the redundant date lag columns
    redundant_date_cols = [c for c in date_cols if c != "midday_day_lag_0"]
    subset = subset.drop(columns=redundant_date_cols)

    subset["same_dow_last_week"] = subset["estimated_avoidable_deaths_lag_7"]
    subset["same_dow_2weeks_ago"] = subset["estimated_avoidable_deaths_lag_14"]
    subset["wow_change"] = subset["estimated_avoidable_deaths_lag_7"] - subset["estimated_avoidable_deaths_lag_14"]

    subset["estimated_avoidable_deaths_rolling_mean_3_7"] = subset[
        [
            "estimated_avoidable_deaths_lag_3",
            "estimated_avoidable_deaths_lag_4",
            "estimated_avoidable_deaths_lag_5",
            "estimated_avoidable_deaths_lag_6",
            "estimated_avoidable_deaths_lag_7",
        ]
    ].mean(axis=1)
    subset["estimated_avoidable_deaths_rolling_std_3_7"] = subset[
        [
            "estimated_avoidable_deaths_lag_3",
            "estimated_avoidable_deaths_lag_4",
            "estimated_avoidable_deaths_lag_5",
            "estimated_avoidable_deaths_lag_6",
            "estimated_avoidable_deaths_lag_7",
        ]
    ].std(axis=1)
    subset["estimated_avoidable_deaths_trend_3_7"] = (
        subset["estimated_avoidable_deaths_lag_3"] - subset["estimated_avoidable_deaths_lag_7"]
    )
    subset["estimated_avoidable_deaths_rolling_mean_7_14"] = subset[
        ["estimated_avoidable_deaths_lag_7", "estimated_avoidable_deaths_lag_14"]
    ].mean(axis=1)
    subset["estimated_avoidable_deaths_weekly_change"] = (
        subset["estimated_avoidable_deaths_lag_7"] - subset["estimated_avoidable_deaths_lag_14"]
    )

    subset["trend_acceleration"] = (
        subset["estimated_avoidable_deaths_lag_3"] - subset["estimated_avoidable_deaths_lag_5"]
    ) - (subset["estimated_avoidable_deaths_lag_5"] - subset["estimated_avoidable_deaths_lag_7"])

    subset = subset.drop(columns=["midday_day"])

    return subset
