from typing import Any, Dict, Tuple
from collections import Counter

import numpy as np
import pandas as pd
from scipy.fft import _pocketfft
from scipy.signal import find_peaks
from statsmodels.tsa.stattools import adfuller
from tsfresh.feature_extraction import extract_features
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import _prepare_data_corr_plot
import base64
from io import BytesIO
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from .versions import pandas_version

if pandas_version() >= [1, 5]:
    from pandas.core.arrays.integer import IntegerDtype
else:
    from pandas.core.arrays.integer import _IntegerDtype as IntegerDtype

import pandas as pd
import numpy as np
from .utils import (calculate_value_counts, to_numeric, histogram_compute,
                    numeric_stats_numpy, numeric_stats_pandas, mad, reduceCategoricalDict)


def __calculate_gaps(df: pd.DataFrame, types_dict: dict):
    all_gap_dict = {'table': {}, 'variables': {}}
    max_gap_all = -np.Inf
    min_gap_all = np.Inf
    average_gap_all = 0
    count_gaps = 0

    # Dictionary with gap size as key and count as value
    gaps_dict = Counter()

    for column in df:
        if types_dict[column]['type'] == 'TimeSeries':
            all_gap_dict['variables'][column] = {}
            gaps = list(df[column].isnull().astype(int).groupby(df[column].notnull().astype(int).cumsum()).sum())

            true_gaps = [gap for gap in gaps if gap > 0]
            if len(true_gaps) != 0:
                # Calculate the statistics for the observed gap sizes
                s = pd.Series(true_gaps)
                stats = s.describe(percentiles=[.10, .25, .75, .90])

                gaps_distribution = {
                    'name': column,
                    'count': stats[0],
                    'min': stats[3],
                    'max': stats[9],
                    'average': stats[1],
                    'stddev': stats[2],
                    'median': stats[6],
                    'kurtosis': s.kurtosis(),
                    'skewness': s.skew(),
                    'variance': s.var(),
                    'percentile10': stats[4],
                    'percentile25': stats[5],
                    'percentile75': stats[7],
                    'percentile90': stats[8],
                }

                all_gap_dict['variables'][column]['gaps_distribution'] = gaps_distribution

                # increase the global count for each gap size
                gaps_dict = gaps_dict + Counter(true_gaps)

                max_gap = max(true_gaps)
                min_gap = min(true_gaps)

                if min_gap < min_gap_all:
                    min_gap_all = min_gap

                if max_gap > max_gap_all:
                    max_gap_all = max_gap

                length_gaps = len(true_gaps)
                sum_gaps = sum(true_gaps)
                count_gaps += length_gaps
                average_gap_all += sum_gaps
                avg_gap = sum_gaps / length_gaps
            else:
                all_gap_dict['variables'][column]['gaps_distribution'] = {}
    if count_gaps != 0:
        average_gap_all = round(average_gap_all / count_gaps)
    else:
        min_gap_all = 0
        max_gap_all = 0
    if len(gaps_dict) != 0:
        gaps_dict = dict(gaps_dict)
    else:
        gaps_dict = {}

    all_gap_dict['table']['ts_min_gap'] = min_gap_all
    all_gap_dict['table']['ts_max_gap'] = max_gap_all
    all_gap_dict['table']['ts_avg_gap'] = average_gap_all
    all_gap_dict['table']['ts_gaps_frequency_distribution'] = gaps_dict

    return all_gap_dict


def stationarity_test(series: pd.Series) -> Tuple[bool, float]:
    # make sure the data has no missing values
    adfuller_test = adfuller(
        series.dropna(),
        autolag="AIC",
        maxlag=None,
    )
    p_value = adfuller_test[1]

    significance_threshold = 0.05
    return p_value < significance_threshold, p_value


def fftfreq(n: int, d: float = 1.0) -> np.ndarray:
    """
    Return the Discrete Fourier Transform sample frequencies.

    Args:
        n : int
            Window length.
        d : scalar, optional
            Sample spacing (inverse of the sampling rate). Defaults to 1.

    Returns:
        f : ndarray
            Array of length `n` containing the sample frequencies.
    """
    val = 1.0 / (n * d)
    results = np.empty(n, int)
    N = (n - 1) // 2 + 1
    p1 = np.arange(0, N, dtype=int)
    results[:N] = p1
    p2 = np.arange(-(n // 2), 0, dtype=int)
    results[N:] = p2
    return results * val


def seasonality_test(series: pd.Series, mad_threshold: float = 6.0) -> Dict[str, Any]:
    """Detect seasonality with FFT

    Source: https://github.com/facebookresearch/Kats/blob/main/kats/detectors/seasonality.py

    Args:
        mad_threshold: Optional; float; constant for the outlier algorithm for peak
            detector. The larger the value the less sensitive the outlier algorithm
            is.

    Returns:
        FFT Plot with peaks, selected peaks, and outlier boundary line.
    """

    fft = get_fft(series)
    _, _, peaks = get_fft_peaks(fft, mad_threshold)
    seasonality_presence = len(peaks.index) > 0
    selected_seasonalities = []
    if seasonality_presence:
        selected_seasonalities = peaks["freq"].transform(lambda x: 1 / x).tolist()

    return {
        "seasonality_presence": seasonality_presence,
        "seasonalities": selected_seasonalities,
    }


def get_fft(series: pd.Series) -> pd.DataFrame:
    """Computes FFT

    Args:
        series: pd.Series
            time series

    Returns:
        DataFrame with columns 'freq' and 'ampl'.
    """
    data_fft = _pocketfft.fft(series.to_numpy())
    data_psd = np.abs(data_fft) ** 2
    fftfreq_ = fftfreq(len(data_psd), 1.0)
    pos_freq_ix = fftfreq_ > 0

    freq = fftfreq_[pos_freq_ix]
    ampl = 10 * np.log10(data_psd[pos_freq_ix])

    return pd.DataFrame({"freq": freq, "ampl": ampl})


def get_fft_peaks(
    fft: pd.DataFrame, mad_threshold: float = 6.0
) -> Tuple[float, pd.DataFrame, pd.DataFrame]:
    """Computes peaks in fft, selects the highest peaks (outliers) and
        removes the harmonics (multiplies of the base harmonics found)

    Args:
        fft: FFT computed by get_fft
        mad_threshold: Optional; constant for the outlier algorithm for peak detector.
            The larger the value the less sensitive the outlier algorithm is.

    Returns:
        outlier threshold, peaks, selected peaks.
    """
    pos_fft = fft.loc[fft["ampl"] > 0]
    median = pos_fft["ampl"].median()
    pos_fft_above_med = pos_fft[pos_fft["ampl"] > median]
    mad = abs(pos_fft_above_med["ampl"] - pos_fft_above_med["ampl"].mean()).mean()

    threshold = median + mad * mad_threshold

    peak_indices = find_peaks(fft["ampl"], threshold=0.1)
    peaks = fft.loc[peak_indices[0], :]

    orig_peaks = peaks.copy()

    peaks = peaks.loc[peaks["ampl"] > threshold].copy()
    peaks["Remove"] = [False] * len(peaks.index)
    peaks.reset_index(inplace=True)

    # Filter out harmonics
    for idx1 in range(len(peaks)):
        curr = peaks.loc[idx1, "freq"]
        for idx2 in range(idx1 + 1, len(peaks)):
            if peaks.loc[idx2, "Remove"] is True:
                continue
            fraction = (peaks.loc[idx2, "freq"] / curr) % 1
            if fraction < 0.01 or fraction > 0.99:
                peaks.loc[idx2, "Remove"] = True
    peaks = peaks.loc[~peaks["Remove"]]
    peaks.drop(inplace=True, columns="Remove")
    return threshold, orig_peaks, peaks


def __ts_fresh_json(df, json_decoded, no_time=False) -> pd.DataFrame:
    """
    This method uses tsfresh to calculate a comprehensive number of features.

    :param df: A pandas Dataframe with 3 columns (time, value,  id) or 2 columns (value,  id) as required to extract
    features from tsfresh.
    :type df: pandas.DataFrame
    :param json_decoded: A dictionary containing the tsfresh features.
    :type json_decoded: dict
    :param no_time: A boolean that if 'True' means that the 'time' column doesn't exist on the pandas Dataframe.
    :type no_time: bool
    :return:
        -tf (pandas.DataFrame) - A pandas DataFrame containing  the loaded time series as rows and the extracted
        features as columns.
    """
    if no_time:
        tf = extract_features(df, column_id="id",
                              column_value="value", default_fc_parameters=json_decoded, n_jobs=0,
                              disable_progressbar=True)
    else:
        tf = extract_features(df, column_id="id", column_sort="time",
                              column_value="value", default_fc_parameters=json_decoded, n_jobs=0,
                              disable_progressbar=True)

    return tf.rename(columns=lambda x: x.split("value__")[1]).rename(columns=lambda x: x.replace("_", " "))


def get_correlation_data(pd_series, fft=True, method="ywm"):
    """Capture all plot components needed for _plot_corr in one dict"""
    # Common parameters
    color = "#377eb8"
    alpha = 0.05

    # Determine lag (preserve original logic)
    def _get_ts_lag(s):
        return min(100, (len(s) // 2) - 1)

    lag = _get_ts_lag(pd_series)

    # ===== ACF Data =====
    lags_acf, nlags_acf, irregular_acf = _prepare_data_corr_plot(pd_series, lag, True)
    acf_results = acf(
        pd_series,
        nlags=nlags_acf,
        alpha=alpha,
        fft=fft,
        bartlett_confint=True,
        adjusted=False
    )

    # ===== PACF Data =====
    lags_pacf, nlags_pacf, irregular_pacf = _prepare_data_corr_plot(pd_series, lag, True)
    pacf_results = pacf(
        pd_series,
        nlags=nlags_pacf,
        alpha=alpha,
        method=method
    )

    return {
        "acf": {
            "values": acf_results[0],
            "confint": acf_results[1] if alpha else None,
            "lags": lags_acf,
            "irregular": irregular_acf
        },
        "pacf": {
            "values": pacf_results[0],
            "confint": pacf_results[1] if alpha else None,
            "lags": lags_pacf,
            "irregular": irregular_pacf
        }
    }


def recreate_plots(corr_data, ax1=None, ax2=None):
    """Recreate plots using _plot_corr with stored data"""
    from statsmodels.graphics.tsaplots import _plot_corr

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot ACF
    _plot_corr(
        ax=ax1,
        title='ACF',
        acf_x=np.array(corr_data["acf"]["values"]),
        confint=np.array(corr_data["acf"]["confint"]),
        lags=np.array(corr_data["acf"]["lags"]),
        irregular=np.array(corr_data["acf"]["irregular"]),
        use_vlines=True,
        vlines_kwargs={"colors": "#377eb8"},
        auto_ylims=False,
        **{"color": "#377eb8"}
    )

    # Plot PACF
    _plot_corr(
        ax=ax2,
        title='PACF',
        acf_x=np.array(corr_data["pacf"]["values"]),
        confint=np.array(corr_data["pacf"]["confint"]),
        lags=np.array(corr_data["pacf"]["lags"]),
        irregular=np.array(corr_data["pacf"]["irregular"]),
        use_vlines=True,
        vlines_kwargs={"colors": "#377eb8"},
        **{"color": "#377eb8"}
    )

    # Apply styling
    for ax, title in zip((ax1, ax2), ['ACF', 'PACF']):
        ax.set_facecolor("#002b36")
        ax.set_title(title, color="white")
        ax.tick_params(axis="both", colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        for spine in ax.spines.values():
            spine.set_color("white")

    fig.patch.set_facecolor("#002b36")
    return fig

def describe_timeseries(series: pd.Series, var_dict: dict, var_name: str, max_freq_distr: int, gaps_dict: dict) -> dict:
    series = to_numeric(series)
    _, value_counts, value_counts_index_sorted = calculate_value_counts(series)
    series = series.dropna()

    quantiles = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]

    negative_index = value_counts.index < 0
    var_dict["n_negative"] = value_counts.loc[negative_index].sum()
    var_dict["p_negative"] = var_dict["n_negative"] / (var_dict["count"] + var_dict["num_missing"])

    infinity_values = [np.inf, -np.inf]
    infinity_index = value_counts.index.isin(infinity_values)
    var_dict["n_infinite"] = value_counts.loc[infinity_index].sum()

    var_dict["n_zeros"] = 0
    if 0 in value_counts.index:
        var_dict["n_zeros"] = value_counts.loc[0]

    if isinstance(series.dtype, IntegerDtype):
        var_dict.update(numeric_stats_pandas(series))
        present_values = series.astype(str(series.dtype).lower())
    else:
        present_values = series.values
        var_dict.update(numeric_stats_numpy(present_values, series, value_counts))

    var_dict.update(
        {
            "mad": mad(present_values),
        }
    )

    var_dict["range"] = var_dict["max"] - var_dict["min"]
    var_dict.update(
        {
            f"percentile{int(percentile * 100)}": value
            for percentile, value in series.quantile(quantiles).to_dict().items()
        }
    )

    var_dict["median"] = var_dict["percentile50"]
    del var_dict["percentile50"]
    var_dict["iqr"] = var_dict["percentile75"] - var_dict["percentile25"]
    var_dict["cv"] = var_dict["stddev"] / var_dict["average"] if var_dict["average"] else np.nan

    var_dict["p_zeros"] = var_dict["n_zeros"] / (var_dict["count"] + var_dict["num_missing"])
    var_dict["p_infinite"] = var_dict["n_infinite"] / (var_dict["count"] + var_dict["num_missing"])

    monotonic_increase = series.is_monotonic_increasing
    monotonic_decrease = series.is_monotonic_decreasing

    monotonic_increase_strict = (
            monotonic_increase and series.is_unique
    )
    monotonic_decrease_strict = (
            monotonic_decrease and series.is_unique
    )
    if monotonic_increase_strict:
        var_dict["monotonic"] = 2
    elif monotonic_decrease_strict:
        var_dict["monotonic"] = -2
    elif monotonic_increase:
        var_dict["monotonic"] = 1
    elif monotonic_decrease:
        var_dict["monotonic"] = -1
    else:
        var_dict["monotonic"] = 0

    if len(value_counts[~infinity_index].index.values) > 0:
        hist = histogram_compute(value_counts[~infinity_index].index.values, var_dict["n_distinct"],
                                 weights=value_counts[~infinity_index].values)

        var_dict['histogram_counts'] = hist['histogram'][0]
        var_dict['histogram_bins'] = hist['histogram'][1]

    value_counts_count_sorted = dict(sorted(value_counts.items(), key=lambda item: item[1], reverse=True))

    value_counts_count_sorted = reduceCategoricalDict(value_counts_count_sorted, max_freq_distr)

    var_dict['freq_value_counts'] = []
    for value, count in value_counts_count_sorted.items():
        var_dict['freq_value_counts'].append({'name': var_name, 'value': value, "count": count})

    list_index_sorted = list(value_counts_index_sorted.items())
    var_dict['freq_five_max_values'] = []
    for value, count in list_index_sorted[:5]:
        var_dict['freq_five_max_values'].append({'name': var_name, 'value': value, "count": count})

    var_dict['freq_five_min_values'] = []
    for value, count in list_index_sorted[-5:]:
        var_dict['freq_five_min_values'].append({'name': var_name, 'value': value, "count": count})

    # add gap stastistics
    var_dict['gaps_distribution'] = gaps_dict['gaps_distribution']

    # add statistics about seasonality, stationarity, Dickey-Fuller
    var_dict["seasonal"] = seasonality_test(series)["seasonality_presence"]
    is_stationary, p_value = stationarity_test(series)
    var_dict["stationary"] = is_stationary and not var_dict["seasonal"]
    var_dict["add_fuller"] = p_value

    # add tsfresh statistics
    tsfresh_stats_dict = {
        "abs_energy": None,
        "absolute_sum_of_changes": None,
        "count_above_mean": None,
        "count_below_mean": None,
        "number_cwt_peaks": [{"n": 10}]
    }

    df = pd.DataFrame()
    dates_float = range(len(series))
    df['time'] = dates_float
    df['id'] = series.name
    df['value'] = series.values

    ts_fresh_results = __ts_fresh_json(df, tsfresh_stats_dict, no_time=False).to_dict(orient='records')[0]
    var_dict['abs_energy'] = ts_fresh_results['abs energy']
    var_dict['abs_sum_changes'] = ts_fresh_results['absolute sum of changes']
    var_dict['len_above_mean'] = ts_fresh_results['count above mean']
    var_dict['len_below_mean'] = ts_fresh_results['count below mean']
    var_dict['num_peaks'] = ts_fresh_results['number cwt peaks  n 10']

    # add acf and pacf plots
    var_dict['acf_pacf'] = get_correlation_data(series)

    return var_dict


