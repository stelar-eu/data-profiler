import pandas as pd
from .utils import calculate_value_counts

def describe_boolean(series: pd.Series, var_dict: dict, var_name: str) -> dict:
    _, value_counts, _ = calculate_value_counts(series)
    var_dict['value_counts_without_nan'] = []
    if not value_counts.empty:
        for value, count in value_counts.items():
            var_dict['value_counts_without_nan'].append({'name': var_name, 'value': value, 'count': count})

    return var_dict
