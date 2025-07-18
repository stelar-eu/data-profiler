import pandas as pd
from .utils import calculate_value_counts

def describe_categorical(series: pd.Series, var_dict: dict, var_name: str) -> dict:
    _, value_counts, _ = calculate_value_counts(series)
    series = series.astype(str)
    series = series.dropna()

    value_counts.index = value_counts.index.astype(str)

    # category distribution
    var_dict['frequency_distribution'] = []
    if not value_counts.empty:
        for value, count in value_counts.items():
            var_dict['frequency_distribution'].append({'name': var_name, 'type': value, 'count': count})

    # top 5 samples
    var_dict['samples'] = []
    for cat, count in series.head(5).items():
        var_dict['samples'].append({'row': cat, "cat": count})

    return var_dict

