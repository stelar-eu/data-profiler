import pandas as pd
import numpy as np
from .utils import calculate_value_counts, to_datetime, histogram_compute

def describe_datetime(series: pd.Series, var_dict: dict) -> dict:
    _, value_counts, _ = calculate_value_counts(series)

    og_series = series.dropna()
    series = to_datetime(og_series)

    series = series.dropna()

    if value_counts.empty:
        values = series.values
        var_dict.update(
            {
                "start": pd.NaT,
                "end": pd.NaT,
                "date_range": 0,
            }
        )
    else:
        var_dict.update(
            {
                "start": pd.Timestamp.to_pydatetime(series.min()),
                "end": pd.Timestamp.to_pydatetime(series.max()),
            }
        )

        var_dict["date_range"] = var_dict["end"] - var_dict["start"]

        values = series.values.astype(np.int64) // 10 ** 9

    hist = histogram_compute(values, series.nunique())

    var_dict['histogram_counts'] = hist['histogram'][0]
    var_dict['histogram_bins'] = hist['histogram'][1]

    return var_dict
