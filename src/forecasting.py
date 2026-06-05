import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def create_features_and_targets(
    df: pd.DataFrame,
    target_col: str = "estimated_avoidable_deaths",
    horizon_length: int = 10,
    feature_lags: Sequence[int] = (0, 1, 2),
    target_lags: Sequence[int] = (3, 4, 5),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Transform a time series DataFrame into a supervised learning format.

    Creates lagged versions of features and the target variable to be used as
    inputs (X), and shifts the target variable forward to create a
    multi-step forecasting horizon (y).

    Parameters
    ----------
    df : pd.DataFrame
        Input time series data containing features and the target column.
    target_col : str, default="estimated_avoidable_deaths"
        The name of the column to be forecasted.
    horizon_length : int, default=10
        The number of future steps to predict (D+1, D+2, ..., D+n).
    feature_lags : Sequence[int], default=(0, 1, 2)
        The specific lags of the feature columns to include in X.
        Lag 0 represents the current time step.
    target_lags : Sequence[int], default=(3, 4, 5)
        The specific autoregressive lags of the target column to include in X.
        Must be >= 3 per domain constraints.

    Returns
    -------
    X : pd.DataFrame
        The feature matrix containing lagged features and lagged targets.
    y : pd.DataFrame
        The target matrix containing the multi-step future horizon.

    Raises
    ------
    ValueError
        If any value in `target_lags` is less than 3.
    """

    # Enforce D-3 constraint for autoregressive features
    if any(lag < 3 for lag in target_lags):
        raise ValueError("target_lags must be >= 3 to prevent data leakage.")

    # Split raw data into feature set and target series
    X_raw = df.drop(columns=[target_col])
    y_raw = df[target_col]

    # Generate lagged versions of all features (X)
    X_lags_list = [X_raw.shift(lag).add_suffix(f"_lag_{lag}") for lag in feature_lags]
    X_lagged = pd.concat(X_lags_list, axis=1)

    # Generate autoregressive lags of the target to use as features (X)
    y_lags_list = [y_raw.shift(lag).rename(f"{target_col}_lag_{lag}") for lag in target_lags]
    y_lagged = pd.concat(y_lags_list, axis=1)

    # Generate the future steps to predict (y)
    y_steps_list = [y_raw.shift(-step).rename(f"{target_col}_step_{step}") for step in range(1, horizon_length + 1)]
    y_horizon = pd.concat(y_steps_list, axis=1)

    # Align all components and remove rows with boundary NaNs
    tabular_df = pd.concat([X_lagged, y_lagged, y_horizon], axis=1).dropna()

    # Extract final X (inputs) and y (outputs)
    feature_cols = X_lagged.columns.tolist() + y_lagged.columns.tolist()
    target_cols = y_horizon.columns.tolist()

    X = tabular_df[feature_cols]
    y = tabular_df[target_cols]

    return X, y



def create_time_series_splits(
    X: pd.DataFrame,
    use_cross_validation: bool = True,
    test_size: int = 10,
    n_splits: int = 5,
    gap: int = 0,
    max_train_size: int | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Generate indices for temporal cross-validation or a single train-test split.

    For cross-validation, it uses an expanding window approach where each
    successive fold includes more historical data. For a single split, it
    reserves the most recent data for testing.

    Parameters
    ----------
    X : pd.DataFrame
        The feature matrix (used to determine the total number of samples).
    use_cross_validation : bool, default=True
        If True, returns multiple folds using TimeSeriesSplit.
        If False, returns a single (train, test) pair.
    test_size : int, default=30
        The number of samples in each test/validation window.
    n_splits : int, default=5
        Number of folds to create (only used if use_cross_validation is True).
    gap : int, default=0
        Size of gap between training and testing data indexes.
    max_train_size : int, optional
        Maximum size for a single training set. If provided, creates a
        rolling window instead of an expanding window.

    Returns
    -------
    splits : list of tuple of np.ndarray
        A list containing (train_indices, test_indices) for each fold.
    """

    if use_cross_validation:
        splitter = TimeSeriesSplit(
            n_splits=n_splits,
            test_size=test_size,
            gap=gap,
            max_train_size=max_train_size,
        )
        splits = list(splitter.split(X))

    else:
        n_samples = len(X)

        train_indices = np.arange(n_samples - test_size)
        test_indices = np.arange(n_samples - test_size, n_samples)

        splits = [(train_indices, test_indices)]

    return splits
    splits = create_time_series_splits(X, use_cross_validation=True)
    mse, y_pred = backtest_forecaster(X, y, pipeline, splits)
