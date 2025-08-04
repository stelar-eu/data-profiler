import pandas as pd
import numpy as np
import os
import geopandas as gpd
from typing import Union, Any
from collections import Counter
from datetime import datetime
from .variables import (describe_datetime, describe_boolean,
                        describe_numeric, describe_timeseries,
                        describe_categorical, describe_textual,
                        describe_geometry, read_tabular_timeseries,
                        calculate_generic_df, calculate_table_stats,
                        find_types)

from ..utils import read_config, write_to_json


def profile_tabular(input_path: Union[str, pd.DataFrame, gpd.GeoDataFrame],
                    header: Union[str, int] = 0, sep: str = ',',
                    light_mode: bool = False, crs: str = 'EPSG:4326',
                    num_cat_perc_threshold: float = 0.5, max_freq_distr=10,
                    eps_distance=1000, extra_geometry_columns: list = None,
                    types_dict=None) -> dict:
    """
    Profiles a tabular DataFrame (or file path) and returns a profiling report as a dictionary.

    :param input_path: Path to input file or a DataFrame/GeoDataFrame to profile.
    :type input_path: str or pandas.DataFrame or geopandas.GeoDataFrame
    :param header: Row to use as column names (passed to pandas.read).
    :type header: int or str
    :param sep: Separator to use for reading CSV files.
    :type sep: str
    :param light_mode: If True, skip expensive computations.
    :type light_mode: bool
    :param crs: Coordinate reference system for geometry data.
    :type crs: str
    :param num_cat_perc_threshold: Threshold for treating numeric as categorical.
    :type num_cat_perc_threshold: float
    :param max_freq_distr: Top-K most frequent values to be displayed in the frequency distribution.
    :type max_freq_distr: int
    :param eps_distance: Distance tolerance for geometry heatmap calculations.
    :type eps_distance: int
    :param extra_geometry_columns: Additional geometry columns to consider.
    :type extra_geometry_columns: list
    :param types_dict: Pre-computed types dictionary to use instead of detecting.
    :type types_dict: dict or None
    :return: Profiling report dictionary.
    :rtype: dict
    """

    input_dict = {
        'input_path': input_path,
        'header': header,
        'sep': sep,
        'light_mode': light_mode,
        'num_cat_perc_threshold': num_cat_perc_threshold,
        'max_freq_distr': max_freq_distr,
        'ts_mode': False,
        'extra_geometry_columns': extra_geometry_columns,
        'crs': crs,
        'eps_distance': eps_distance
    }

    return __profiler_tabular_timeseries(input_dict, types_dict)


def profile_tabular_with_config(config: dict) -> None:
    """
    This method performs profiling on tabular and/or vector data and write the resulting profile dictionary based on a configuration dictionary.

    :param config: a dictionary with all configuration settings.
    :type config: dict
    :return: None.
    :rtype: None

    """

    input_config = config.get("input", {})
    output_config = config.get("output", {})

    input_file_path = input_config.get("files")
    if isinstance(input_file_path, list):
        if len(input_file_path) == 1:
            my_file_path = os.path.abspath(input_file_path[0])
            types_dict = None
        elif len(input_file_path) == 2:
            my_file_path = os.path.abspath(input_file_path[0])
            types_dict = read_config(os.path.abspath(input_file_path[1]))
        else:
            raise ValueError("Expected one or two file paths in 'files'.")
    elif isinstance(input_file_path, str):
        my_file_path = os.path.abspath(input_file_path)
        types_dict = None
    else:
        raise ValueError("Invalid input path format.")

    output_json_path = os.path.abspath(output_config.get("json"))

    # Extract parameters explicitly
    header = input_config.get("header", 0)
    sep = input_config.get("sep", ',')
    light_mode = input_config.get("light_mode", False)
    crs = input_config.get("crs", 'EPSG:4326')
    num_cat_perc_threshold = input_config.get("num_cat_perc_threshold", 0.5)
    max_freq_distr = input_config.get("max_freq_distr", 10)
    eps_distance = input_config.get("eps_distance", 1000)
    extra_geometry_columns = input_config.get("extra_geometry_columns", None)

    profile_dict = profile_tabular(
        input_path=my_file_path,
        header=header,
        sep=sep,
        light_mode=light_mode,
        crs=crs,
        num_cat_perc_threshold=num_cat_perc_threshold,
        max_freq_distr=max_freq_distr,
        eps_distance=eps_distance,
        extra_geometry_columns=extra_geometry_columns,
        types_dict=types_dict
    )

    write_to_json(profile_dict, output_json_path)


def profile_timeseries(input_path: Union[str, pd.DataFrame],
                       ts_mode_datetime_col: str = 'date',
                       header: Union[str, int] = 0, sep: str = ',',
                       light_mode: bool = False,
                       num_cat_perc_threshold: float = 0.5, max_freq_distr=10,
                       types_dict=None) -> dict:
    """
       Profiles a timeseries DataFrame or file and returns a profiling report dictionary.

       :param input_path: Path to input file or DataFrame to profile.
       :type input_path: str or pandas.DataFrame
       :param ts_mode_datetime_col: Column name for datetime index.
       :type ts_mode_datetime_col: str
       :param header: Row to use as column names.
       :type header: int or str
       :param sep: Field separator for CSV.
       :type sep: str
       :param light_mode: If True, skip expensive computations.
       :type light_mode: bool
       :param num_cat_perc_threshold: Threshold for treating numeric as categorical.
       :type num_cat_perc_threshold: float
       :param max_freq_distr: Top-K most frequent values to be displayed in the frequency distribution.
       :type max_freq_distr: int
       :param types_dict: Pre-computed types dictionary to use instead of detecting.
       :type types_dict: dict or None
       :return: Profiling report dictionary.
       :rtype: dict
       """
    input_dict = {
        'input_path': input_path,
        'header': header,
        'sep': sep,
        'light_mode': light_mode,
        'num_cat_perc_threshold': num_cat_perc_threshold,
        'max_freq_distr': max_freq_distr,
        'ts_mode': True,
        'ts_mode_datetime_col': ts_mode_datetime_col,
    }

    return __profiler_tabular_timeseries(input_dict, types_dict)


def profile_timeseries_with_config(config: dict) -> None:
    """
    This method performs profiling on timeseries data and writes the resulting profile dictionary based on a configuration dictionary.

    :param config: a dictionary with all configuration settings.
    :type config: dict
    :return: None.
    :rtype: None

    """

    input_config = config.get("input", {})
    output_config = config.get("output", {})

    input_file_path = input_config.get("files")
    if isinstance(input_file_path, list):
        if len(input_file_path) == 1:
            my_file_path = os.path.abspath(input_file_path[0])
            types_dict = None
        elif len(input_file_path) == 2:
            my_file_path = os.path.abspath(input_file_path[0])
            types_dict = read_config(os.path.abspath(input_file_path[1]))
        else:
            raise ValueError("Expected one or two file paths in 'files'.")
    elif isinstance(input_file_path, str):
        my_file_path = os.path.abspath(input_file_path)
        types_dict = None
    else:
        raise ValueError("Invalid input path format.")

    output_json_path = os.path.abspath(output_config.get("json"))

    # Extract parameters explicitly
    header = input_config.get("header", 0)
    sep = input_config.get("sep", ',')
    light_mode = input_config.get("light_mode", False)
    ts_mode_datetime_col = input_config.get("ts_mode_datetime_col", 'date')
    num_cat_perc_threshold = input_config.get("num_cat_perc_threshold", 0.5)
    max_freq_distr = input_config.get("max_freq_distr", 10)

    profile_dict = profile_timeseries(
        input_path=my_file_path,
        header=header,
        sep=sep,
        light_mode=light_mode,
        ts_mode_datetime_col=ts_mode_datetime_col,
        num_cat_perc_threshold=num_cat_perc_threshold,
        max_freq_distr=max_freq_distr,
        types_dict=types_dict
    )

    write_to_json(profile_dict, output_json_path)


def type_detection(input_path: Union[str, pd.DataFrame, gpd.GeoDataFrame],
                   header: Union[str, int] = 0, sep: str = ',',
                   ts_mode: bool = False, ts_mode_datetime_col: str = None,
                   crs: str = 'EPSG:4326', num_cat_perc_threshold: float = 0.5,
                   max_freq_distr=10, eps_distance: int = 1000,
                   extra_geometry_columns: list = None, **kwargs) -> dict:
    """
   Detects data types for each column in tabular or timeseries data.

   :param input_path: Path or DataFrame/GeoDataFrame to inspect.
   :type input_path: str or pandas.DataFrame or geopandas.GeoDataFrame
   :param header: Header row index or name.
   :type header: int or str
   :param sep: Separator character.
   :type sep: str
   :param ts_mode: Whether to treat data as timeseries.
   :type ts_mode: bool
   :param ts_mode_datetime_col: Datetime column for timeseries.
   :type ts_mode_datetime_col: str
   :param crs: Coordinate reference system for geometry data.
   :type crs: str
   :param num_cat_perc_threshold: Threshold for treating numeric as categorical.
   :type num_cat_perc_threshold: float
   :param max_freq_distr: Top-K most frequent values to be displayed in the frequency distribution.
   :type max_freq_distr: int
   :param eps_distance: Distance tolerance for geometry heatmap calculations.
   :type eps_distance: int
   :param extra_geometry_columns: Additional geometry columns to consider.
   :type extra_geometry_columns: list
   :return: Types dictionary mapping column to detected type info.
   :rtype: dict
   """

    df, crs = read_tabular_timeseries(input_path=input_path, header=header,
                                      sep=sep, crs=crs,
                                      ts_mode_datetime_col=ts_mode_datetime_col,
                                      extra_geometry_columns=extra_geometry_columns)
    input_dict = {
        'input_path': input_path,
        'header': header,
        'sep': sep,
        'num_cat_perc_threshold': num_cat_perc_threshold,
        'max_freq_distr': max_freq_distr,
        'ts_mode': ts_mode,
        'ts_mode_datetime_col': ts_mode_datetime_col,
        'crs': crs,
        'extra_geometry_columns': extra_geometry_columns,
        'eps_distance': eps_distance}

    types_dict = find_types(df, input_dict=input_dict)

    return types_dict


def type_detection_with_config(config: dict) -> None:
    """
    This method performs type detection on tabular or timeseries data and writes the resulting type detection dictionary based on a configuration dictionary.

    :param config: a dictionary with all configuration settings.
    :type config: dict
    :return: None.
    :rtype: None

    """

    input_config = config.get("input", {})
    output_config = config.get("output", {})

    input_file_path = input_config.get("files")
    if isinstance(input_file_path, list):
        if len(input_file_path) == 1:
            my_file_path = os.path.abspath(input_file_path[0])
        else:
            raise ValueError(f"Invalid input: {input_file_path} must be a valid file path or list with one file path")
    elif isinstance(input_file_path, str) and os.path.isfile(os.path.abspath(input_file_path)):
        my_file_path = os.path.abspath(input_file_path)
    else:
        raise ValueError(f"Invalid input: {input_file_path} must be a valid file path or list of file paths")

    output_json_path = os.path.abspath(output_config.get("json"))

    # Extract parameters with defaults if not provided
    header = input_config.get("header", 0)
    sep = input_config.get("sep", ',')
    ts_mode = input_config.get("ts_mode", False)
    ts_mode_datetime_col = input_config.get("ts_mode_datetime_col", None)
    crs = input_config.get("crs", 'EPSG:4326')
    num_cat_perc_threshold = input_config.get("num_cat_perc_threshold", 0.5)
    max_freq_distr = input_config.get("max_freq_distr", 10)
    eps_distance = input_config.get("eps_distance", 1000)
    extra_geometry_columns = input_config.get("extra_geometry_columns", None)

    profile_dict = type_detection(
        input_path=my_file_path,
        header=header,
        sep=sep,
        ts_mode=ts_mode,
        ts_mode_datetime_col=ts_mode_datetime_col,
        crs=crs,
        num_cat_perc_threshold=num_cat_perc_threshold,
        max_freq_distr=max_freq_distr,
        eps_distance=eps_distance,
        extra_geometry_columns=extra_geometry_columns
    )

    write_to_json(profile_dict, output_json_path)


def __profiler_tabular_timeseries(input_dict, types_dict=None):
    df, crs = read_tabular_timeseries(**input_dict)

    input_dict['crs'] = crs

    filename_df = False
    if isinstance(input_dict['input_path'], str):
        filename_df = True

    if types_dict is None:
        types_dict = find_types(df, input_dict)

    generic_dict = calculate_generic_df(df)

    profile_dict = {
        'analysis': {
            'title': 'Profiling Report',
            'date_start': datetime.utcnow(),
            'date_end': datetime.utcnow(),
            'duration': ''
        },
        'table': {
            'profiler_type': '',
            'light_mode': input_dict['light_mode'],
            'memory_size': 0,
            'record_size': 0,
            'num_rows': 0,
            'num_attributes': 0,
            'n_cells_missing': 0,
            'p_cells_missing': 0.0,
            'types': []
        },
        'variables': []
    }

    if filename_df:
        profile_dict['analysis']['filenames'] = [input_dict['input_path']]
    else:
        profile_dict['analysis']['filenames'] = ['Profiler was not given a path']

    # calculate table stats
    if input_dict['ts_mode']:
        profile_dict['table']['profiler_type'] = 'TimeSeries'
    else:
        profile_dict['table']['profiler_type'] = 'Tabular'

    tables_stats = calculate_table_stats(df, types_dict, generic_dict)
    common_keys = profile_dict['table'].keys() & tables_stats.keys()  # Get common keys
    profile_dict['table'].update({key: tables_stats[key] for key in common_keys})

    # if timeseries add gap stats
    if input_dict['ts_mode'] and not input_dict['light_mode']:
        all_gaps_dict = __calculate_gaps(df, types_dict)
        profile_dict['table']['ts_min_gap'] = all_gaps_dict['table']['ts_min_gap']
        profile_dict['table']['ts_max_gap'] = all_gaps_dict['table']['ts_max_gap']
        profile_dict['table']['ts_avg_gap'] = all_gaps_dict['table']['ts_avg_gap']

        profile_dict['table']['ts_gaps_frequency_distribution'] = []
        gaps_freq_distr = []
        for gap, count in all_gaps_dict['table']['ts_gaps_frequency_distribution'].items():
            profile_dict['table']['ts_gaps_frequency_distribution'].append({'gap_size': gap, "count": count})
        # calculate variable stats + time series gap statistics
        profile_dict['variables'] = __calculate_variable_stats(df, types_dict, generic_dict, all_gaps_dict['variables'],
                                                               light_mode=input_dict['light_mode'])

    else:
        # calculate variable stats
        profile_dict['variables'] = __calculate_variable_stats(df, types_dict, generic_dict,
                                                               light_mode=input_dict['light_mode'])

    profile_dict['analysis']['date_end'] = datetime.utcnow()
    profile_dict['analysis']['duration'] = str(
        profile_dict['analysis']['date_end'] - profile_dict['analysis']['date_start'])

    return profile_dict


# calculate variables statistics
def __calculate_variable_stats(df: pd.DataFrame, types_dict: dict, generic_dict: dict, gaps_variable_dict: dict = None,
                               light_mode=False) -> dict:
    all_var_list = []

    for column in df:
        var_general = generic_dict[column]

        var_dict = {
            'name': column,
            'type': types_dict[column]['type'],
            'count': var_general['count'],
            'num_missing': var_general['n_missing'],
            'uniqueness': var_general['p_unique'],
            'p_missing': var_general['p_missing'],
            'memory_size': var_general['memory_size'],
            'n_unique': var_general['n_unique'],
            'n_distinct': var_general['n_distinct'],
            'p_distinct': var_general['p_distinct'],

        }

        if var_general['hashable'] and not light_mode:
            var_type = var_dict['type'].lower()
            
            if var_type == 'datetime':
                describe_datetime(df[column], var_dict)

            if var_type == 'boolean':
                describe_boolean(df[column], var_dict, column)

            if var_type == 'numeric':
                describe_numeric(df[column], var_dict, column, types_dict[column]['max_freq_distr'])

            if var_type == 'timeseries':
                describe_timeseries(df[column], var_dict, column, types_dict[column]['max_freq_distr'],
                                    gaps_variable_dict[column])

            if var_type == 'categorical':
                describe_categorical(df[column], var_dict, column)

            if var_type == 'textual':
                describe_textual(df[column], var_dict, column)

            if var_type == 'geometry':
                describe_geometry(df[column], var_dict, column, types_dict[column]['crs'],
                                  types_dict[column]['eps_distance'])

        all_var_list.append(var_dict)

    return all_var_list


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
