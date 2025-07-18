import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import box, MultiPoint
from sklearn.cluster import DBSCAN
from pyproj import CRS
from .utils import calculate_value_counts, reduceCategoricalDict


def sample_points(geometry, num_points=10):
    """Sample up to num_points from a geometry."""
    if geometry.is_empty:
        return []
    points = []
    if geometry.geom_type == 'Point':
        points.append(geometry)
    elif geometry.geom_type in ['LineString', 'LinearRing']:
        for i in range(num_points):
            point = geometry.interpolate(i / max(num_points - 1, 1), normalized=True)
            points.append(point)
    elif geometry.geom_type == 'Polygon':
        exterior = geometry.exterior
        for i in range(num_points):
            point = exterior.interpolate(i / max(num_points - 1, 1), normalized=True)
            points.append(point)
    elif geometry.geom_type.startswith('Multi'):
        for geom in geometry.geoms:
            points.extend(sample_points(geom, num_points))
            if len(points) >= num_points:
                break
    return points[:num_points]


def __get_clusters_dict(geo_data: gpd.GeoSeries, geometry_column: str = None, eps_distance: float = 1000):
    MIN_SAMPLE_POLYGONS = 5

    # Create a GeoDataFrame with the provided GeoSeries.
    wkt = gpd.GeoDataFrame(geometry=geo_data)
    if geometry_column is not None:
        wkt = wkt.rename(columns={'geometry': geometry_column})
        wkt = wkt.set_geometry(geometry_column)

    # Compute centroids in the current (projected) CRS.
    # Save these centroids in a new column.
    wkt['centroid'] = wkt.geometry.centroid
    # Extract x and y from the centroids for clustering.
    wkt['x_proj'] = wkt['centroid'].x
    wkt['y_proj'] = wkt['centroid'].y
    coords = wkt[['x_proj', 'y_proj']].values

    # Run DBSCAN clustering on the projected centroids.
    dbscan = DBSCAN(eps=eps_distance, min_samples=MIN_SAMPLE_POLYGONS)
    clusters = dbscan.fit(coords)
    labels = clusters.labels_

    # Cap clusters to max 2000
    valid_labels = labels[labels != -1]
    unique, counts = np.unique(valid_labels, return_counts=True)
    if len(unique) > 2000:
        sorted_indices = np.argsort(-counts)[:2000]
        selected = unique[sorted_indices]
        mask = np.isin(labels, selected)
        new_labels = np.where(mask, labels, -1)
    else:
        new_labels = labels

    wkt['Clusters'] = pd.Series(new_labels, index=wkt.index)

    # wkt['Clusters'] = pd.Series(clusters.labels_, index=wkt.index)

    # Reproject the entire GeoDataFrame to EPSG:4326.
    wkt_4326 = wkt.to_crs('EPSG:4326')

    # Reproject the previously computed centroids.
    centroid_gs = gpd.GeoSeries(wkt['centroid'], crs=wkt.crs)
    centroid_4326 = centroid_gs.to_crs('EPSG:4326')

    # Use the reprojected centroids to set x and y for the heatmap.
    wkt_4326['x'] = centroid_4326.x
    wkt_4326['y'] = centroid_4326.y

    # Prepare the output dictionary with the heatmap data in EPSG:4326.
    data = wkt_4326[['y', 'x', 'Clusters']]
    return data.to_dict()


def describe_geometry(series: pd.Series, var_dict: dict, var_name: str, crs: str, eps_distance: float = 1000) -> dict:

    # Convert input series to GeoSeries
    try:
        geo_series = gpd.GeoSeries.from_wkt(series, crs=crs)
    except:
        geo_series = gpd.GeoSeries(series, crs=crs)

    # Convert to WGS84 (EPSG:4326)
    geo_series = geo_series.to_crs('EPSG:4326')

    _, value_counts, _ = calculate_value_counts(geo_series)
    geo_series = geo_series.dropna()

    var_dict['samples'] = []
    for cat, count in geo_series.head(5).items():
        var_dict['samples'].append({'row': cat, "cat": count})

    value_counts_count_sorted = dict(sorted(value_counts.items(), key=lambda item: item[1], reverse=True))

    value_counts_count_sorted = reduceCategoricalDict(value_counts_count_sorted, 10)

    var_dict['freq_value_counts'] = []
    for value, count in value_counts_count_sorted.items():
        var_dict['freq_value_counts'].append({'name': var_name, 'value': value, "count": count})

    # Compute MBR and convex hull in WGS84
    var_dict['mbr'] = box(*geo_series.total_bounds).wkt

    # Approximate convex hull by sampling points
    sampled_points = []
    for geom in geo_series.geometry:
        sampled_points.extend(sample_points(geom, num_points=10))
    if sampled_points:
        convex_hull = MultiPoint(sampled_points).convex_hull
        var_dict['union_convex_hull'] = convex_hull.wkt
    else:
        var_dict['union_convex_hull'] = None

    # var_dict['union_convex_hull'] = geo_series.unary_union.convex_hull.wkt

    # Determine UTM CRS based on centroid of the dataset
    centroid = geo_series.unary_union.centroid
    lon, lat = centroid.x, centroid.y
    zone_number = int((lon + 180) // 6) + 1
    utm_epsg = 32600 + zone_number if lat >= 0 else 32700 + zone_number
    utm_crs = f'EPSG:{utm_epsg}'

    # Project to UTM CRS
    try:
        geo_series_proj = geo_series.to_crs(utm_crs)
    except Exception as e:
        print(f"Error projecting to UTM CRS: {e}, using EPSG:3395 as fallback.")
        geo_series_proj = geo_series.to_crs('EPSG:3395')

    # Compute centroid in projected CRS and convert back to WGS84
    centroid_proj = geo_series_proj.unary_union.centroid
    centroid_wgs84 = gpd.GeoSeries([centroid_proj], crs=geo_series_proj.crs).to_crs('EPSG:4326').iloc[0]
    var_dict['centroid'] = centroid_wgs84.wkt

    # Generate heatmap using projected coordinates
    # sample_size = 2000 if len(geo_series_proj) > 2000 else None
    # heatmap_data = geo_series_proj[:sample_size] if sample_size else geo_series_proj
    # var_dict['heatmap'] = __get_clusters_dict(heatmap_data, var_name, eps_distance=2000)

    # Heatmap with all data points
    var_dict['heatmap'] = __get_clusters_dict(geo_series_proj, var_name, eps_distance=eps_distance)

    # CRS information
    if crs:
        parsed_crs = CRS.from_string(crs)
        var_dict['crs'] = f'EPSG:{parsed_crs.to_epsg()}'
    else:
        var_dict['crs'] = 'EPSG:4326'

    # Geometry type counts
    geom_types = geo_series.geom_type.value_counts().to_dict()

    var_dict['geom_type_distribution'] = []
    for geom_type, frequency in geom_types.items():
        var_dict['geom_type_distribution'].append({'name': var_name, 'type': geom_type, 'count': frequency})

    # Area distribution calculations
    area_stats = geo_series_proj.area.describe(percentiles=[.10, .25, .75, .90])
    var_dict['area_distribution'] = {
        'name': var_name,
        'count': area_stats['count'],
        'min': area_stats['min'],
        'max': area_stats['max'],
        'average': area_stats['mean'],
        'stddev': area_stats['std'],
        'median': area_stats['50%'],
        'kurtosis': geo_series_proj.area.kurtosis(),
        'skewness': geo_series_proj.area.skew(),
        'variance': geo_series_proj.area.var(),
        'percentile10': area_stats['10%'],
        'percentile25': area_stats['25%'],
        'percentile75': area_stats['75%'],
        'percentile90': area_stats['90%'],
    }

    # Length distribution calculations
    length_stats = geo_series_proj.length.describe(percentiles=[.10, .25, .75, .90])
    var_dict['length_distribution'] = {
        'name': var_name,
        'count': length_stats['count'],
        'min': length_stats['min'],
        'max': length_stats['max'],
        'average': length_stats['mean'],
        'stddev': length_stats['std'],
        'median': length_stats['50%'],
        'kurtosis': geo_series_proj.length.kurtosis(),
        'skewness': geo_series_proj.length.skew(),
        'variance': geo_series_proj.length.var(),
        'percentile10': length_stats['10%'],
        'percentile25': length_stats['25%'],
        'percentile75': length_stats['75%'],
        'percentile90': length_stats['90%'],
    }

    return var_dict
