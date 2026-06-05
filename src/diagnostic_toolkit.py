import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def mse_metric(
    test_data: list[pd.DataFrame], forecast_data: list[pd.DataFrame]
) -> list[list[float], float, list[float], float, list[float], float]:
    """
    Calculates the Mean Squared Error (MSE) for each 10-day forecast period in the test data
    set and aggregates the results into specific performance buckets (Days 1-10, Days 1-5, and Days 6-10).
    Definition can be found in SPHERE-PPL Git.

    Parameters
    ----------
    test_data : list of pd.DataFrame
        A list of all 10-day testing data that is forecasted.
    forecast_data : list of pd.DataFrame
        A list of all forecasted data that corresponds to the list of test data.


    Returns
    -------
    mse_list : list[float]
        MSE for the whole forecast range (Days 1-10) for each testing
        and forecast pair.
    summed_mse : float
        The sum of the MSE for the whole forecast range (Days 1-10) testing
        and forecasting pairs.
    mse_1_5_list : list[float]
        MSE for the 'short_horizon' (Days 1-5) for each testing
        and forecast pair.
    summed_mse_1_5 : float
        The sum of the MSE for the 'short_horizon' (Days 1-5) testing
        and forecasting pairs.
    mse_6_10_list : list[float]
        MSE for the 'long_horizon' (Days 6-10) for each testing
        and forecast pair.
    summed_mse_6_10 : float
        The sum of the MSE for the 'long_horizon' (Days 1-5) testing
        and forecasting pairs.
    """

    if len(test_data) != len(forecast_data):
        raise ValueError("testing and forecast data sets must be same length.")

    mse_list, mse_1_5_list, mse_6_10_list = [], [], []
    for k in range(len(test_data)):
        mse_list.append(mean_squared_error(test_data[k], forecast_data[k]))
        mse_1_5_list.append(mean_squared_error(test_data[k][:5], forecast_data[k][:5]))
        mse_6_10_list.append(mean_squared_error(test_data[k][5:], forecast_data[k][5:]))
    summed_mse, summed_mse_1_5, summed_mse_6_10 = (
        sum(mse_list) / len(test_data),
        sum(mse_1_5_list) / len(test_data),
        sum(mse_6_10_list) / len(test_data),
    )

    return mse_list, summed_mse, mse_1_5_list, summed_mse_1_5, mse_6_10_list, summed_mse_6_10


def mae_metric(
    test_data: list[pd.DataFrame], forecast_data: list[pd.DataFrame]
) -> list[list[float], float, list[float], float, list[float], float]:
    """
    Calculates the Mean Absolute Error (MAE) for each 10-day forecast period in the test data
    set and aggregates the results into specific performance buckets (Days 1-10, Days 1-5 and Days 6-10).

    Parameters
    ----------
    test_data : list of pd.DataFrame
        A list of all 10-day testing data that is forecasted.
    forecast_data : list of pd.DataFrame
        A list of all forecasted data that corresponds to the list of test data.


    Returns
    -------
    mae_list : list[float]
        MAE for the whole forecast range (Days 1-10) for each testing
        and forecast pair.
    summed_mae : float
        The sum of the MAE for the whole forecast range (Days 1-10) testing
        and forecasting pairs.
    mae_1_5_list : list[float]
        MAE for the 'short_horizon' (Days 1-5) for each testing
        and forecast pair.
    summed_mae_1_5 : float
        The sum of the MAE for the 'short_horizon' (Days 1-5) testing
        and forecasting pairs.
    mae_6_10_list : list[float]
        MAE for the 'long_horizon' (Days 6-10) for each testing
        and forecast pair.
    summed_mae_6_10 : float
        The sum of the MAE for the 'long_horizon' (Days 1-5) testing
        and forecasting pairs.
    """

    if len(test_data) != len(forecast_data):
        raise ValueError("testing and forecast data sets must be same length.")

    mae_list, mae_1_5_list, mae_6_10_list = [], [], []
    for k in range(len(test_data)):
        mae_list.append(mean_absolute_error(test_data[k], forecast_data[k]))
        mae_1_5_list.append(mean_absolute_error(test_data[k][:5], forecast_data[k][:5]))
        mae_6_10_list.append(mean_absolute_error(test_data[k][5:], forecast_data[k][5:]))
    summed_mae, summed_mae_1_5, summed_mae_6_10 = (
        sum(mae_list) / len(test_data),
        sum(mae_1_5_list) / len(test_data),
        sum(mae_6_10_list) / len(forecast_data),
    )

    return mae_list, summed_mae, mae_1_5_list, summed_mae_1_5, mae_6_10_list, summed_mae_6_10


def mbe_metric(
    test_data: list[pd.DataFrame], forecast_data: list[pd.DataFrame]
) -> list[list[float], list[float], list[float]]:
    """
    Calculates the Mean Bias Error (MBE) for each 10-day forecast period in the test data
    set and aggregates the results into specific performance buckets (Days 1-10, Days 1-5 and Days 6-10).

    Parameters
    ----------
    test_data : list of pd.DataFrame
        A list of all 10-day testing data that is forecasted.
    forecast_data : list of pd.DataFrame
        A list of all forecasted data that corresponds to the list of test data.


    Returns
    -------
    mbe_list : list[float]
        MBE for the whole forecast range (Days 1-10) for each testing
        and forecast pair.
    mbe_1_5_list : list[float]
        MBE for the 'short_horizon' (Days 1-5) for each testing
        and forecast pair.
    mbe_6_10_list : list[float]
        MBE for the 'long_horizon' (Days 6-10) for each testing
        and forecast pair.
    """

    if len(test_data) != len(forecast_data):
        raise ValueError("testing and forecast data sets must be same length.")

    mbe_list, mbe_1_5_list, mbe_6_10_list = [], [], []
    for k in range(len(test_data)):
        dif = np.array(forecast_data[k]) - np.array(test_data[k])
        mbe_list.append(np.mean(dif))
        mbe_1_5_list.append(np.mean(dif[:5]))
        mbe_6_10_list.append(np.mean(dif[5:]))

    return mbe_list, mbe_1_5_list, mbe_6_10_list


def mase_metric(
    train_data: list[pd.DataFrame], test_data: list[pd.DataFrame], forecast_data: list[pd.DataFrame], s: int
) -> list[list[float], list[float], list[float]]:
    """
    Calculates the Mean Absolute Scaled Error (MASE) for each 10-day forecast period in the test data
    set and aggregates the results into specific performance buckets (Days 1-10, Days 1-5 and Days 6-10).

    Parameters
    ----------
    train_data : list of pd.DataFrame
        A list of all training data sets that are used for forecasting.
    test_data : list of pd.DataFrame
        A list of all 10-day testing data that is forecasted.
    forecast_data : list of pd.DataFrame
        A list of all forecasted data that corresponds to the list of test data.


    Returns
    -------
    mase_list : list[float]
        MASE for the whole forecast range (Days 1-10) for each testing
        and forecast pair.
    mase_1_5_list : list[float]
        MASE for the 'short_horizon' (Days 1-5) for each testing
        and forecast pair.
    mase_6_10_list : list[float]
        MASE for the 'long_horizon' (Days 6-10) for each testing
        and forecast pair.
    s : int
        Seasonality of the data.
        Weekly seasonality: s = 7
        Annual seasonality: s = 365
    """

    train_data = [np.array(y_t) for y_t in train_data]
    mase_list, mase_1_5_list, mase_6_10_list = [], [], []
    mae_list, _, mae_1_5_list, _, mae_6_10_list, _ = mae_metric(test_data, forecast_data)
    for k in range(len(test_data)):
        denom = float(np.sum(np.abs(train_data[k][s:] - train_data[k][:-s])) / (len(train_data[k]) - s))
        mase_list.append(float(mae_list[k] / denom))
        mase_1_5_list.append(float(mae_1_5_list[k] / denom))
        mase_6_10_list.append(float(mae_6_10_list[k] / denom))
    return mase_list, mase_1_5_list, mase_6_10_list


def full_metric_analysis(
    train_data: list[pd.DataFrame], test_data: list[pd.DataFrame], forecast_data: list[pd.DataFrame], s: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculates the MSE, MAE, MBE, and MASE for each 10-day forecast period in the test data
    set, aggregates the results into specific performance buckets (Days 1-10, Days 1-5
    and Days 6-10), then saves the results to two separate pandas dataframes.

    Note: the index is open-ended with the intention of it being changed after
    creation as either a timeseries or category

    Parameters
    ----------
    train_data : list of pd.DataFrame
        A list of all training data sets that are used for forecasting.
    test_data : list of pd.DataFrame
        A list of all 10-day testing data that is forecasted.
    forecast_data : list of pd.DataFrame
        A list of all forecasted data that corresponds to the list of test data.


    Returns
    -------
    metric_list_results : pd.DataFrame
        Data frame containing the whole forecast range (Days 1-10), the 'short_horizon' (Days 1-5),
        and the 'long_horizon' (Days 6-10) lists for the MSE, MAE, MBE, and MASE statistics.
    metric_single_results : pd.DataFrame
        Data frame containing singular, summed values of MSE and MAE for the whole forecast range (Days 1-10),
        the 'short_horizon' (Days 1-5), and the 'long_horizon' (Days 6-10) windows.
    """

    mse_list, summed_mse, mse_1_5_list, summed_mse_1_5, mse_6_10_list, summed_mse_6_10 = mse_metric(
        test_data, forecast_data
    )
    mae_list, summed_mae, mae_1_5_list, summed_mae_1_5, mae_6_10_list, summed_mae_6_10 = mae_metric(
        test_data, forecast_data
    )
    mbe_list, mbe_1_5_list, mbe_6_10_list = mbe_metric(test_data, forecast_data)
    mase_list, mase_1_5_list, mase_6_10_list = mase_metric(train_data, test_data, forecast_data, s)
    metric_list_results = pd.DataFrame(
        {
            "mse": mse_list,
            "mse_1_5": mse_1_5_list,
            "mse_6_10": mse_6_10_list,
            "mae": mae_list,
            "mae_1_5": mae_1_5_list,
            "mae_6_10": mae_6_10_list,
            "mbe": mbe_list,
            "mbe_1_5": mbe_1_5_list,
            "mbe_6_10": mbe_6_10_list,
            "mase": mase_list,
            "mase_1_5": mase_1_5_list,
            "mase_6_10": mase_6_10_list,
        }
    )
    metric_single_results = pd.DataFrame(
        {
            "summed_mse": [summed_mse],
            "summed_mse_1_5": [summed_mse_1_5],
            "summed_mse_6_10": [summed_mse_6_10],
            "summed_mae": [summed_mae],
            "summed_mae_1_5": [summed_mae_1_5],
            "summed_mae_6_10": [summed_mae_6_10],
        }
    )
    return metric_list_results, metric_single_results


def model_comparison(
    train_data: list[pd.DataFrame],
    test_data: list[pd.DataFrame],
    forecast_dataset: list[pd.DataFrame],
    plot_range: list[int] | None = None,
) -> None:
    """
    Calculates the MSE, MAE, MBE, and MASE for each 10-day forecast period in the test data
    set, aggregates the results into specific performance buckets (Days 1-10, Days 1-5 and Days 6-10),
    then saves the results to two separate pandas dataframes.

    Note: the index is open-ended with the intention of it being changed after creation
    as either a timeseries or category

    Parameters
    ----------
    train_data : list of pd.DataFrame
        A list of all training data sets that are used for forecasting.
    test_data : list of pd.DataFrame
        A list of all 10-day testing data that is forecasted.
    forecast_data : list of pd.DataFrame
        A list of all forecasted data that corresponds to the list of test data.
    plot_range: list of integers
        A list to indicate which numerical slices of training, testing, and forecasted
        data to plot. If set to 'all', plots all slices of data.


    Returns
    -------
    None: Plots training, testing, and forecasted data.
    """
    if plot_range is None:
        plot_range = [0]
    if plot_range == "all":
        plot_range = range(0, len(train_data))
    for k in plot_range:
        plt.figure()
        start = test_data[k]["midday_day"].min() - pd.Timedelta(days=60)
        end = test_data[k]["midday_day"].max() + pd.Timedelta(days=10)

        plt.plot(train_data[k]["midday_day"], train_data[k]["estimated_avoidable_deaths"], label="training data")
        plt.plot(test_data[k]["midday_day"], test_data[k]["estimated_avoidable_deaths"], label="testing data")
        plt.plot(
            forecast_dataset[k]["midday_day"], forecast_dataset[k]["estimated_avoidable_deaths"], label="predicted data"
        )
        plt.xlabel("Date")
        plt.title("Forecast for cut %i" % k)
        plt.legend()
        plt.xlim(start, end)


def plot_results(results_list: pd.DataFrame) -> None:
    """
    Plots the MSE, MAE, MBE, and MASE for each Days 1-10, Days 1-5 and Days 6-10 forecast period.

    Note: the index is open-ended with the intention of it being changed after creation
    as either a timeseries or category

    Parameters
    ----------
    metric_list_results : pd.DataFrame
        Data frame containing the whole forecast range (Days 1-10), the 'short_horizon' (Days 1-5),
        and the 'long_horizon' (Days 6-10) lists for the MSE, MAE, MBE, and MASE statistics.


    Returns
    -------
    None: Plots MSE, MAE, MBE, and MASE results over all periods onto subplots and displayed.
    """

    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 10))

    results_list[["mse", "mse_1_5", "mse_6_10"]].plot(ax=axes[0], rot=45)
    axes[0].set_title("Mean Square Error")

    results_list[["mae", "mae_1_5", "mae_6_10"]].plot(ax=axes[1], rot=45)
    axes[1].set_title("Mean Absolute Error")

    results_list[["mbe", "mbe_1_5", "mbe_6_10"]].plot(ax=axes[2], rot=45)
    axes[2].set_title("Mean Bias Error")
    axes[2].axhline(y=0, color="k", linestyle="--")

    results_list[["mase", "mase_1_5", "mase_6_10"]].plot(ax=axes[3], rot=45)
    axes[3].set_title("Mean Absolute Scaled Error")
    axes[3].axhline(y=1, color="k", linestyle="--")

    fig.suptitle("Forecast metric results")
    plt.tight_layout()
