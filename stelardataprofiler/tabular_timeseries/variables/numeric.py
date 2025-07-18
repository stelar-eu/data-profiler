from .versions import pandas_version

if pandas_version() >= [1, 5]:
    from pandas.core.arrays.integer import IntegerDtype
else:
    from pandas.core.arrays.integer import _IntegerDtype as IntegerDtype

import pandas as pd
import numpy as np
from .utils import (calculate_value_counts, to_numeric, histogram_compute,
                    numeric_stats_numpy, numeric_stats_pandas, mad, reduceCategoricalDict)

def describe_numeric(series: pd.Series, var_dict: dict, var_name: str, max_freq_distr: int) -> dict:
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

    return var_dict

