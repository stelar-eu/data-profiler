import pandas as pd
import numpy as np
import geopandas as gpd
from typing import Union, Any, Optional
from typing import Dict
from .versions import is_pandas_1, pandas_version

if pandas_version() >= [1, 5]:
    from pandas.core.arrays.integer import IntegerDtype
else:
    from pandas.core.arrays.integer import _IntegerDtype as IntegerDtype
from datetime import datetime
from shapely.geometry import Point, LineString, Polygon, MultiPoint, MultiLineString, MultiPolygon
from shapely import wkt
from collections import Counter
import warnings


# Functions
# read
def __read_csv_files(my_file, header=None, sep=',', encoding='UTF-8'):
    try:
        df = pd.read_csv(my_file, header=header, sep=sep, encoding=encoding)
    except:
        return pd.DataFrame()

    return df


def check_geometries(gs: gpd.GeoSeries) -> gpd.GeoSeries:
    if gs.is_valid.all():
        return True
    else:
        return False


def check_transform_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    valid_geometries = check_geometries(gdf['geometry'])
    if valid_geometries:
        return gdf
    else:
        gdf['geometry'] = gdf['geometry'].buffer(0)
        return gdf


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df.rename(columns={"index": "df_index"}, inplace=True)

    if "index" in df.index.names:
        df.index.names = [x if x != "index" else "df_index" for x in df.index.names]

    # Ensure that columns are strings
    df.columns = df.columns.astype("str")

    return df


def read_tabular_timeseries(input_path: Union[str, pd.DataFrame, gpd.GeoDataFrame], header: Union[str, int] = 0,
                            sep: str = ',', crs: str = None, ts_mode_datetime_col: str = None,
                            extra_geometry_columns: list = None, **kwargs) -> pd.DataFrame:
    if crs is None:
        crs = 'EPSG:4326'

    if isinstance(input_path, str):
        if input_path.__contains__('.shp'):
            pois = gpd.read_file(input_path)
            crs = pois.crs
            pois = check_transform_geometries(pois)
            df = pd.DataFrame(pois)
            df.geometry = df.geometry.apply(lambda x: x.wkt)
        else:
            df = __read_csv_files(input_path, header, sep)
    elif isinstance(input_path, pd.DataFrame):
        df = input_path
    else:
        pois = input_path
        crs = pois.crs
        pois = check_transform_geometries(pois)
        df = pd.DataFrame(pois)
        df.geometry = df.geometry.apply(lambda x: x.wkt)

    if extra_geometry_columns is not None:
        for lat_lon in extra_geometry_columns:
            latitude_column = lat_lon['latitude']
            longitude_column = lat_lon['longitude']

            if all(col in df.columns for col in [latitude_column, longitude_column]):
                geom_lon_lat = "geometry_" + longitude_column + "_" + latitude_column
                s = gpd.GeoSeries.from_xy(df[longitude_column], df[latitude_column], crs=crs)
                df[geom_lon_lat] = s.to_wkt()

    if ts_mode_datetime_col is not None:
        if ts_mode_datetime_col not in df.columns:
            raise KeyError(f"Date column '{ts_mode_datetime_col}' not found in CSV headers")

        df[ts_mode_datetime_col] = to_datetime(
            df[ts_mode_datetime_col]
        )

        # Set datetime index and sort
        df = df.set_index(ts_mode_datetime_col)
        df = df.sort_index(ascending=True)  # Ensure chronological order

        # Remove potential duplicate indices
        if df.index.duplicated().any():
            df = df[~df.index.duplicated(keep='first')]

        df.reset_index(inplace=True)

    df = prepare_df(df)
    return df, str(crs)


# generic_variables.py
def calculate_value_counts(series: Union[pd.Series, gpd.GeoSeries], summary=None) -> (
        bool, dict, Union[pd.Series, gpd.GeoSeries, None], Union[pd.Series, gpd.GeoSeries, None]):
    if summary is None:
        summary = {}
    try:
        value_counts_with_nan = series.value_counts(dropna=False)
        _ = set(value_counts_with_nan.index)
        hashable = True
    except:  # noqa: E722
        hashable = False

    summary["hashable"] = hashable

    value_counts_without_nan = None
    value_counts_index_sorted = None

    if hashable:
        value_counts_with_nan = value_counts_with_nan[value_counts_with_nan > 0]

        null_index = value_counts_with_nan.index.isnull()
        if null_index.any():
            n_missing = value_counts_with_nan[null_index].sum()
            value_counts_without_nan = value_counts_with_nan[~null_index]
        else:
            n_missing = 0
            value_counts_without_nan = value_counts_with_nan

        try:
            value_counts_index_sorted = value_counts_without_nan.sort_index(ascending=False)
            ordering = True
        except TypeError:
            ordering = False
    else:
        n_missing = series.isna().sum()
        ordering = False

    summary["ordering"] = ordering
    summary["n_missing"] = n_missing

    return summary, value_counts_without_nan, value_counts_index_sorted


def calculate_generic(series: Union[pd.Series, gpd.GeoSeries], summary=None) -> dict:
    if summary is None:
        summary = {}
    summary, value_counts, _ = calculate_value_counts(series)

    # number of observations in the Series
    length = len(series)

    summary.update(
        {
            "n": length,
            "p_missing": summary["n_missing"] / length if length > 0 else 0,
            "count": length - summary["n_missing"],
            "memory_size": series.memory_usage(deep=True),
        }
    )

    if summary['hashable']:
        # number of non-NaN observations in the Series
        count = summary["count"]

        distinct_count = len(value_counts)
        unique_count = value_counts.where(value_counts == 1).count()

        summary.update(
            {
                "n_distinct": distinct_count,
                "p_distinct": distinct_count / count if count > 0 else 0,
                "is_unique": unique_count == count and count > 0,
                "n_unique": unique_count,
                "p_unique": unique_count / count if count > 0 else 0,
            }
        )

    return summary


def calculate_generic_df(df: pd.DataFrame) -> dict:
    generic_dict = {}

    # Global calculations
    for column in df.columns:
        generic_dict[column] = calculate_generic(df[column])

    return generic_dict


# find_types.py
def check_if_datetime(series: Union[pd.Series, gpd.GeoSeries]) -> bool:
    """If we can transform data to datetime and at least one is valid date."""
    try:
        # Convert to datetime, errors='coerce' will turn invalid parsing into NaT
        converted = to_datetime(series)
        # Check if all non-null values were successfully converted
        if converted.notna().all():
            return True
    except Exception as e:
        # If there's an error in conversion, skip the column
        return False


def check_if_geometry(series: Union[pd.Series, gpd.GeoSeries]) -> bool:
    # Check if all non-null entries are either geometries or valid WKT strings
    for item in series:
        if isinstance(item, (Point, LineString, Polygon, MultiPoint, MultiLineString, MultiPolygon)):
            continue  # It's already a geometry
        try:
            # Try to convert from WKT
            wkt.loads(item)
        except Exception as e:
            return False  # Not a valid geometry or WKT string

    return True


def check_if_boolean(series: Union[pd.Series, gpd.GeoSeries]) -> bool:
    mappings: Dict[str, bool] = {
        "t": True,
        "f": False,
        "yes": True,
        "no": False,
        "y": True,
        "n": False,
        "true": True,
        "false": False,
    }

    try:
        return series.str.lower().isin(mappings.keys()).all()
    except:  # noqa: E722
        try:
            return series.isin({True, False}).all()
        except:
            return False


def check_if_numeric(series: Union[pd.Series, gpd.GeoSeries]) -> bool:
    try:
        _ = series.astype(float)
        r = pd.to_numeric(series, errors="coerce")
        if r.hasnans and r.count() == 0:
            return False
    except:  # noqa: E722
        return False

    return True


def check_if_string(series: Union[pd.Series, gpd.GeoSeries]) -> bool:
    if not all(isinstance(v, str) for v in series.values[0:5]):
        return False
    try:
        return (series.astype(str).values == series.values).all()
    except (TypeError, ValueError):
        return False


def check_if_timedependent(series: Union[pd.Series, gpd.GeoSeries]) -> bool:
    series = pd.to_numeric(series)
    autocorrelation_threshold = 0.1
    lags = [1, 7, 12, 24, 30]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for lag in lags:
            aut_corr = series.autocorr(lag=lag)
            if aut_corr >= autocorrelation_threshold:
                return True

    return False


def find_types(df: pd.DataFrame, input_dict: dict = None) -> dict:
    if input_dict is None:
        input_dict = {
            'num_cat_perc_threshold': 0.5,
            'max_freq_distr': 10,
            'ts_mode': False,
            'crs': 'EPSG:4326',
            'eps_distance': 1000
        }

    # List of required keys
    required_keys = [
        'ts_mode'
    ]

    # Function to check if the input dictionary has all required keys
    def has_required_keys(input_dict, required_keys):
        return all(key in input_dict for key in required_keys)

    if not has_required_keys(input_dict, required_keys):
        return {}
    else:
        types_dict = {}

    # check num_cat_perc_threshold, crs, max_freq_distr and eps_distance if missing or error take default
    if 'crs' not in input_dict:
        crs = 'EPSG:4326'
    else:
        if input_dict['crs'] is None:
            crs = 'EPSG:4326'
        else:
            crs = input_dict['crs']

    try:
        if 'eps_distance' not in input_dict:
            eps_distance = 1000
        else:
            if input_dict['eps_distance'] is None:
                eps_distance = 1000
            else:
                eps_distance = float(input_dict['eps_distance'])
    except:
        eps_distance = 1000

    try:
        if 'num_cat_perc_threshold' not in input_dict:
            num_cat_perc_threshold = 0.5
        else:
            if input_dict['num_cat_perc_threshold'] is None:
                num_cat_perc_threshold = 0.5
            else:
                num_cat_perc_threshold = float(input_dict['num_cat_perc_threshold'])
    except:
        num_cat_perc_threshold = 0.5

    try:
        if 'max_freq_distr' not in input_dict:
            max_freq_distr = 10
        else:
            if input_dict['max_freq_distr'] is None:
                max_freq_distr = 10
            else:
                max_freq_distr = int(input_dict['max_freq_distr'])
    except:
        max_freq_distr = 10

    for column in df.columns:
        values = df[column].dropna()
        if values.empty:
            types_dict[column] = {'type': 'Unsupported'}
            continue

        if pd.api.types.is_datetime64_any_dtype(values):
            types_dict[column] = {'type': 'DateTime'}
            continue

        if df[column].dtype == 'geometry':
            types_dict[column] = {'type': 'Geometry', 'crs': crs, 'eps_distance': eps_distance}
            continue

        if isinstance(df[column].dtype, pd.CategoricalDtype) and not pd.api.types.is_bool_dtype(values):
            types_dict[column] = {'type': 'Categorical'}
            continue

        if pd.api.types.is_numeric_dtype(values) and not pd.api.types.is_bool_dtype(values):
            n_unique = values.nunique()
            p_unique = n_unique / values.size
            unique_threshold = num_cat_perc_threshold

            if 1 <= n_unique <= 5 and (
                    p_unique < unique_threshold if unique_threshold <= 1 else p_unique <= unique_threshold):
                types_dict[column] = {'type': 'Categorical'}
                continue
            else:
                if input_dict['ts_mode'] and check_if_timedependent(values):
                    types_dict[column] = {'type': 'TimeSeries', 'max_freq_distr': max_freq_distr}
                    continue
                else:
                    types_dict[column] = {'type': 'Numeric', 'max_freq_distr': max_freq_distr}
                    continue

        if pd.api.types.is_bool_dtype(values):
            types_dict[column] = {'type': 'Boolean'}
            continue

        if pd.api.types.is_object_dtype(values):
            if check_if_datetime(values):
                types_dict[column] = {'type': 'DateTime'}
                continue
            elif check_if_geometry(values):
                types_dict[column] = {'type': 'Geometry', 'crs': crs, 'eps_distance': eps_distance}
                continue
            elif check_if_boolean(values):
                types_dict[column] = {'type': 'Boolean'}
                continue
            else:
                n_unique = values.nunique()
                p_unique = n_unique / values.size
                unique_threshold = num_cat_perc_threshold

                if check_if_numeric(values):
                    if 1 <= n_unique <= 5 and (
                            p_unique < unique_threshold if unique_threshold <= 1 else p_unique <= unique_threshold):
                        types_dict[column] = {'type': 'Categorical'}
                        continue
                    else:
                        if input_dict['ts_mode'] and check_if_timedependent(values):
                            types_dict[column] = {'type': 'TimeSeries', 'max_freq_distr': max_freq_distr}
                        else:
                            types_dict[column] = {'type': 'Numeric', 'max_freq_distr': max_freq_distr}
                        continue
                else:
                    if pd.api.types.is_string_dtype(values) and check_if_string(values):
                        if p_unique < unique_threshold if unique_threshold <= 1 else p_unique <= unique_threshold:
                            types_dict[column] = {'type': 'Categorical'}
                        else:
                            types_dict[column] = {'type': 'Textual'}
                        continue
                    else:
                        types_dict[column] = {'type': 'Unsupported'}
                        continue

    return types_dict


# calculate table statistics
def calculate_table_stats(df: pd.DataFrame, types_dict: dict, generic_dict: dict) -> dict:
    n = len(df) if not df.empty else 0

    memory_size = df.memory_usage(deep=True).sum()
    record_size = float(memory_size) / n if n > 0 else 0

    table_stats = {
        "num_rows": n,
        "num_attributes": len(df.columns),
        "memory_size": memory_size,
        "record_size": record_size,
        "n_cells_missing": 0,
        "n_vars_with_missing": 0,
        "n_vars_all_missing": 0,
    }

    for column in generic_dict:
        series_summary = generic_dict[column]
        if "n_missing" in series_summary and series_summary["n_missing"] > 0:
            table_stats["n_vars_with_missing"] += 1
            table_stats["n_cells_missing"] += series_summary["n_missing"]
            if series_summary["n_missing"] == n:
                table_stats["n_vars_all_missing"] += 1

    table_stats["p_cells_missing"] = (
        table_stats["n_cells_missing"] / (table_stats["num_rows"] * table_stats["num_attributes"])
        if table_stats["num_rows"] > 0 and table_stats["num_attributes"] > 0
        else 0
    )

    types = []
    for column in types_dict:
        series_type = types_dict[column]['type']
        types.append(series_type)

    types_list = []
    for k, v in sorted(Counter(types).items(), key=lambda x: x[1], reverse=True):
        types_list.append({'type': k, 'count': v})

    # Variable type counts
    table_stats.update(
        {"types": types_list}
    )

    return table_stats


# calculate general statistics
# utils

def to_datetime(series: pd.Series) -> pd.Series:
    if is_pandas_1():
        return pd.to_datetime(series, errors="coerce")
    return pd.to_datetime(series, format="mixed", errors="coerce")


def to_numeric(series: pd.Series) -> pd.Series:
    if not pd.api.types.is_numeric_dtype(series):
        series = pd.to_numeric(series, errors="coerce")

    return series


def reduceCategoricalDict(dictionary: dict, n: int):
    if len(dictionary) > n:
        first_n_pairs = {k: dictionary[k] for k in list(dictionary)[:n]}
        sum_other_pairs = sum(list(dictionary.values())[n:])
        first_n_pairs['Other Values (' + str(len(dictionary) - len(first_n_pairs)) + ')'] = sum_other_pairs
    else:
        first_n_pairs = dictionary
    return first_n_pairs


def histogram_compute(
        finite_values: np.ndarray,
        n_unique: int,
        name: str = "histogram",
        weights: Optional[np.ndarray] = None,
        default_bins: int = 50,
        max_bins: int = 250,
        density: bool = False
) -> dict:
    stats = {}
    if len(finite_values) == 0:
        return {name: []}
    bins_arg = "auto" if default_bins == 0 else min(default_bins, n_unique)
    bins = np.histogram_bin_edges(finite_values, bins=bins_arg)
    if len(bins) > max_bins:
        bins = np.histogram_bin_edges(finite_values, bins=max_bins)
        weights = weights if weights and len(weights) == max_bins else None

    stats[name] = np.histogram(
        finite_values, bins=bins, weights=weights, density=density
    )

    return stats


def mad(arr: np.ndarray) -> np.ndarray:
    """Median Absolute Deviation: a "Robust" version of standard deviation.
    Indices variability of the sample.
    https://en.wikipedia.org/wiki/Median_absolute_deviation
    """
    return np.median(np.abs(arr - np.median(arr)))


def numeric_stats_pandas(series: pd.Series) -> Dict[str, Any]:
    return {
        "average": series.mean(),
        "stddev": series.std(),
        "variance": series.var(),
        "min": series.min(),
        "max": series.max(),
        # Unbiased kurtosis obtained using Fisher's definition (kurtosis of normal == 0.0). Normalized by N-1.
        "kurtosis": series.kurt(),
        # Unbiased skew normalized by N-1
        "skewness": series.skew(),
        "sum": series.sum(),
    }


def numeric_stats_numpy(
        present_values: np.ndarray, series: pd.Series, vc: pd.Series) -> Dict[str, Any]:
    index_values = vc.index.values

    if len(index_values):
        return {
            "average": np.average(index_values, weights=vc.values),
            "stddev": np.std(present_values, ddof=1),
            "variance": np.var(present_values, ddof=1),
            "min": np.min(index_values),
            "max": np.max(index_values),
            # Unbiased kurtosis obtained using Fisher's definition (kurtosis of normal == 0.0). Normalized by N-1.
            "kurtosis": series.kurt(),
            # Unbiased skew normalized by N-1
            "skewness": series.skew(),
            "sum": np.dot(index_values, vc.values),
        }
    else:  # Empty numerical series
        return {
            "average": np.nan,
            "stddev": 0.0,
            "variance": 0.0,
            "min": np.nan,
            "max": np.nan,
            "kurtosis": 0.0,
            "skewness": 0.0,
            "sum": 0,
        }
