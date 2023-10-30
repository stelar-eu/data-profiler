import os
import warnings
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
from pandas_profiling import ProfileReport
from pandas_profiling.model.typeset import ProfilingTypeSet
from pandas_profiling.config import Settings
from pandas_profiling.model.summarizer import PandasProfilingSummarizer
from pandas_profiling.report.presentation.core import Container
import geopandas as gp
from shapely.geometry import box
import pandas as pd
import numpy as np
from tsfresh.feature_extraction import extract_features
from pandas_profiling.utils.paths import get_config
from stelardataprofiler.report import (
    __get_report_structure,
    __get_html_report,
    __to_file,
    __to_json
)
from stelardataprofiler.profile_notebook import __get_notebook_iframe
import yaml
from sklearn.cluster import DBSCAN
from datetime import datetime
import rasterio as rio
from scipy import stats
import dateutil.parser
import json
import shutil
from pathlib import Path
from typing import Union, Any
from IPython.display import display
import nltk

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('vader_lexicon', quiet=True)
from nltk.corpus import stopwords
from spacy_language_detection import LanguageDetector, detect_langs, DetectorFactory
from ftlangdetect import detect
import fasttext

fasttext.FastText.eprint = lambda x: None
import spacy
from spacy.language import Language
import string
from nltk.stem import SnowballStemmer
from collections import Counter
import pycountry
from simplemma import lemmatize
import gensim
import math
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from dataprofiler import Data, Profiler
from rdflib import Graph
import networkx as nx
from rdflib import RDF, URIRef
from rdflib.extras.external_graph_libs import rdflib_to_networkx_multidigraph
from pyproj import CRS
import uuid
import re
from rasterio.warp import transform_bounds
from rasterio.transform import from_origin
from rasterio.io import MemoryFile

__all__ = ['run_profile', 'profile_timeseries', 'profile_timeseries_with_config',
           'profile_tabular', 'profile_tabular_with_config',
           'profile_raster', 'profile_raster_with_config',
           'profile_text', 'profile_text_with_config',
           'profile_hierarchical', 'profile_hierarchical_with_config',
           'profile_rdfGraph', 'profile_rdfGraph_with_config',
           'profile_vista_rasters', 'profile_vista_rasters_with_config',
           'prepare_mapping', 'profile_single_raster',
           'profile_multiple_rasters', 'profile_single_text',
           'profile_multiple_texts', 'write_to_json', 'read_config'
           ]

tsfresh_json_file = str(
    os.path.dirname(os.path.abspath(__file__))) + '/json_files/tsfresh_json.json'


# ------------------------------------#
# ------ PROFILER MAIN FUNCTION ------#
# ------------------------------------#
def run_profile(config: dict) -> None:
    """
    This method executes the specified profiler and writes the resulting profile dictionary, and HTML if specified, based on a configuration dictionary.

    :param config: a dictionary with all configuration settings.
    :type config: dict
    :return: None.
    :rtype: None

    """
    profile_type: str = config['profile']['type'].lower()
    if profile_type == 'timeseries':
        profile_timeseries_with_config(config)
    elif profile_type in ['tabular', 'vector']:
        profile_tabular_with_config(config)
    elif profile_type == 'raster':
        profile_raster_with_config(config)
    elif profile_type == 'textual':
        profile_text_with_config(config)
    elif profile_type == 'hierarchical':
        profile_hierarchical_with_config(config)
    elif profile_type == 'rdfgraph':
        profile_rdfGraph_with_config(config)
    elif profile_type == 'vista':
        profile_vista_rasters_with_config(config)
    else:
        print('The profile type is not available!\n'
              'Please use one of the following types:\n'
              "'timeseries', 'tabular', 'vector', 'raster', 'text', 'hierarchical', 'rdfGraph', 'vista")


def prepare_mapping(config: dict) -> None:
    """
    This method prepares the suitable mapping for subsequent generation of the RDF graph, if "rdf" and "serialization" options are specified in config.

    :param config: a dictionary with all configuration settings.
    :type config: dict
    :return: None.
    :rtype: None

    """

    import sys
    # Get parameters required for conversion to RDF
    output_path = config['output']['path']
    json_file = config['output']['json']
    rdf_file = config['output']['rdf']
    profile_type = config['profile']['type'].lower()
    rdf_serialization = config['output']['serialization']

    # Handle special cases (timeseries, vector) of tabular profile
    if profile_type == 'vector' or profile_type == 'timeseries':
        profile_type = 'tabular'

    # Handle special cases (raster, vista) of raster profile
    if profile_type == 'raster' or profile_type == 'vista':
        profile_type = 'raster'

    # Concatenate path and file names
    in_file = os.path.join(output_path, json_file)
    map_template = os.path.join(os.path.dirname(os.path.abspath(__file__)) +
                                '/mappings', profile_type + '_mapping.ttl')
    map_file = os.path.join(output_path, 'mapping.ttl')
    out_file = os.path.join(output_path, rdf_file)

    # Copy mapping template to temporary 'mapping.ttl'
    if not os.path.isfile(map_template):
        print('ERROR: Mapping ', map_template, 'not found! Check whether such mapping exists in',
              os.path.abspath(map_template))
        sys.exit(1)
    else:
        shutil.copyfile(map_template, map_file)
        print('Mapping ', map_template, ' copied to', map_file)

    # Check if mapping file exists
    if not os.path.isfile(map_file):
        print('ERROR: Mapping for', profile_type, 'profiles not found! Check whether such mapping exists in',
              os.path.abspath(map_file))
        sys.exit(1)

    # Edit the mapping file
    with open(map_file, 'r') as file:
        filedata = file.read()

    # Replace the input with the path to actual JSON profile
    filedata = filedata.replace('./out/profile.json', in_file)

    # Write the file out again
    with open(map_file, 'w') as file:
        file.write(filedata)


# ------------ TIMESERIES ------------#
def profile_timeseries_with_config(config: dict) -> None:
    """
    This method performs profiling on timeseries data and write the resulting profile dictionary based on a configuration dictionary.

    :param config: a dictionary with all configuration settings.
    :type config: dict
    :return: None.
    :rtype: None

    """
    input_dir_path = config['input']['path']
    input_file_name = config['input']['file']
    output_dir_path = config['output']['path']
    output_json_name = config['output']['json']
    output_html_name = ''
    if 'html' in config['output']:
        output_html_name = config['output']['html']
    only_directory_path = False

    # Create input file path
    my_file_path = ''
    if input_file_name == '':
        print('No input file was found for timeseries profile!')
        return None
    else:
        my_file_path = os.path.abspath(os.path.join(input_dir_path, input_file_name))

    # Create output file paths
    output_dir_path = os.path.abspath(output_dir_path)
    output_json_path = os.path.abspath(os.path.join(output_dir_path, output_json_name))
    output_html_path = ''
    if output_html_name != '':
        output_html_path = os.path.abspath(os.path.join(output_dir_path, output_html_name))

    # Run timeseries profile
    if 'time' in config['input']['columns']:
        time_column = config['input']['columns']['time']
        header = config['input']['header']
        sep = config['input']['separator']
        profile_dict = profile_timeseries(my_file_path=my_file_path, time_column=time_column,
                                          header=header, sep=sep, html_path=output_html_path)
        # Write resulting profile dictionary
        write_to_json(profile_dict, output_json_path)
    else:
        print("Please add 'time' as key and the time column name of the input .csv "
              'as value in the JSON under input.columns')


def profile_timeseries(my_file_path: str, time_column: str, header: int = 0, sep: str = ',',
                       html_path: str = '', display_html: bool = False, mode: str = 'verbose') -> dict:
    """
    This method performs profiling and generates a profiling dictionary for a given timeseries .csv file that exists in the given path.

    :param my_file_path: the path to a .csv file containing a datetime columns and one/multiple timeseries columns.
    :type my_file_path: str
    :param time_column: the name of the datetime column.
    :type time_column: str
    :param header: row to use to parse column labels. Defaults to the first row. Prior rows will be discarded.
    :type header: str, optional
    :param sep: separator character to use for the csv.
    :type sep: str, optional
    :param html_path: the file path where the html file will be saved.
    :type html_path: str, optional
    :param display_html: a boolean that determines whether the html will be displayed in the output.
    :type display_html: bool, optional
    :param mode: 'default' -> calculate tsfresh features for the timeseries and use them as variables (useful if many timeseries columns), 'verbose' -> use the timeseries as variables.
    :type mode: str, optional
    :return: A dict which contains the results of the profiler for the timeseries data.
    :rtype: dict

    """
    profile_dict, config, html_dict, sample_timeseries = __profile_timeseries_main(my_file_path, time_column, header,
                                                                                   sep, mode=mode, minimal=True)

    if html_path.strip() or display_html:
        html_report = __get_html_report(config, html_dict, sample_timeseries)
        if display_html:
            display(__get_notebook_iframe(config, html_report))
        if html_path.strip():
            if not isinstance(html_path, Path):
                html_path = Path(str(html_path))

            # create parent folders if they do not exist
            path = Path(str(html_path.parent))
            path.mkdir(parents=True, exist_ok=True)
            __to_file(config, html_report, html_path)

    return profile_dict


# -------------- TABULAR + VECTOR --------------#
def profile_tabular_with_config(config: dict) -> None:
    """
    This method performs profiling on tabular and/or vector data and write the resulting profile dictionary based on a configuration dictionary.

    :param config: a dictionary with all configuration settings.
    :type config: dict
    :return: None.
    :rtype: None

    """
    input_dir_path = config['input']['path']
    input_file_name = config['input']['file']
    output_dir_path = config['output']['path']
    output_json_name = config['output']['json']
    output_html_name = ''
    if 'html' in config['output']:
        output_html_name = config['output']['html']
    only_directory_path = False

    # Create input file path
    my_file_path = ''
    if input_file_name == '':
        print('No input file was found for tabular and/or vector profiles!')
        return None
    else:
        my_file_path = os.path.abspath(os.path.join(input_dir_path, input_file_name))

    # Create output file paths
    output_dir_path = os.path.abspath(output_dir_path)
    output_json_path = os.path.abspath(os.path.join(output_dir_path, output_json_name))
    output_html_path = ''
    if output_html_name != '':
        output_html_path = os.path.abspath(os.path.join(output_dir_path, output_html_name))

    # Run tabular/vector profile
    header = 0
    sep = ','
    if 'header' in config['input']:
        header = config['input']['header']
    if 'separator' in config['input']:
        sep = config['input']['separator']
    longitude_column: str = None
    latitude_column: str = None
    wkt_column: str = None
    if 'columns' in config['input']:
        columns_dict: dict = config['input']['columns']
        if ('longitude' in columns_dict) and ('latitude' in columns_dict) and ('wkt' in columns_dict):
            longitude_column = columns_dict['longitude']
            latitude_column = columns_dict['latitude']
            wkt_column = columns_dict['wkt']

        elif ('longitude' in columns_dict) and ('latitude' in columns_dict):
            longitude_column = columns_dict['longitude']
            latitude_column = columns_dict['latitude']

        elif 'wkt' in columns_dict:
            wkt_column = columns_dict['wkt']

    profile_dict = profile_tabular(my_file_path=my_file_path, header=header, sep=sep,
                                   longitude_column=longitude_column, latitude_column=latitude_column,
                                   wkt_column=wkt_column, html_path=output_html_path)

    # Write resulting profile dictionary
    write_to_json(profile_dict, output_json_path)


def profile_tabular(my_file_path: str, header: int = 0, sep: str = ',', crs: str = "EPSG:4326",
                    longitude_column: str = None, latitude_column: str = None,
                    wkt_column: str = None, html_path: str = '', display_html: bool = False) -> dict:
    """
    This method performs profiling and generates a profiling dictionary for a given tabular .csv or .shp file that exists in the given path.

    :param my_file_path: the path to a .csv or .shp file containing different data types of columns.
    :type my_file_path: str
    :param header: row to use to parse column labels. Defaults to the first row. Prior rows will be discarded.
    :type header: str, optional
    :param sep: separator character to use for the csv.
    :type sep: str, optional
    :param crs: the Coordinate Reference System (CRS) represented as an authority string (eg "EPSG:4326").
    :type crs: str, optional
    :param longitude_column: the name of the longitude column.
    :type longitude_column: str, optional
    :param latitude_column: the name of the latitude column.
    :type latitude_column: str, optional
    :param wkt_column: the name of the column that has wkt geometries.
    :type wkt_column: str, optional
    :param html_path: the file path where the html file will be saved.
    :type html_path: str, optional
    :param display_html: a boolean that determines whether the html will be displayed in the output.
    :type display_html: bool, optional
    :return: A dict which contains the results of the profiler for the tabular data.
    :rtype: dict

    """
    profile_dict, config, html_dict = __profile_tabular_main(my_file_path=my_file_path, header=header,
                                                             sep=sep, longitude_column=longitude_column,
                                                             latitude_column=latitude_column, wkt_column=wkt_column,
                                                             minimal=True)

    if html_path.strip() or display_html:
        html_report = __get_html_report(config, html_dict, None)
        if display_html:
            display(__get_notebook_iframe(config, html_report))
        if html_path.strip():
            if not isinstance(html_path, Path):
                html_path = Path(str(html_path))

            # create parent folders if they do not exist
            path = Path(str(html_path.parent))
            path.mkdir(parents=True, exist_ok=True)
            __to_file(config, html_report, html_path)

    return profile_dict


# -------------- RASTER --------------#
# ----------- SINGLE IMAGE -----------#
def profile_single_raster(my_file_path: str) -> dict:
    """
    This method performs profiling and generates a profiling dictionary for an image file that exists in the given path.

    :param my_file_path: the path to an image file.
    :type my_file_path: str
    :return: A dict which contains the results of the profiler for the image.
    :rtype: dict

    """
    if os.path.isdir(my_file_path):
        print('The input is not a file!')
        return dict()

    profile_dict = {
        'analysis': {
            'title': 'Profiling Report',
            'date_start': '',
            'date_end': '',
            'duration': '',
            'filenames': [my_file_path]
        },
        'table': {
            'profiler_type': 'Raster',
            'byte_size': 0,
            'n_of_imgs': 1,
            'avg_width': 0,
            'avg_height': 0,
        },
        'variables': [], 'package': {
            'pandas_profiling_version': 'v3.5.0',
            'pandas_profiling_config': ''
        }
    }

    # Start time
    now = datetime.now()
    start_string = now.strftime("%Y-%m-%d %H:%M:%S.%f")
    profile_dict['analysis']['date_start'] = start_string

    # File size
    profile_dict['table']['byte_size'] = os.path.getsize(my_file_path)

    # Create image dictionary
    img_dict = {
        'name': '',
        'type': 'Raster',
        'crs': '',
        'spatial_coverage': '',
        'spatial_resolution': {
            'pixel_size_x': 0,
            'pixel_size_y': 0
        },
        'no_data_value': '',
        'format': ''
    }

    # Read image
    img = rio.open(my_file_path)

    # find image name
    name = Path(my_file_path).stem
    img_dict['name'] = name

    # find general image data
    img_dict.update(img.meta)

    # making transform JSON-serializable
    img_dict['transform'] = list(img_dict['transform'])

    profile_dict['table']['avg_width'] = img_dict['width']
    profile_dict['table']['avg_height'] = img_dict['height']

    # change nodata and driver keys
    img_dict['no_data_value'] = img_dict['nodata']
    del img_dict['nodata']

    img_dict['format'] = img_dict['driver']
    del img_dict['driver']

    # find tags
    img_dict['tags'] = []

    for k, v in img.tags().items():
        tag_dict = {
            'key': k,
            'value': v
        }

        img_dict['tags'].append(tag_dict)

    # change crs format
    if img.crs is not None:
        crs_list = CRS.from_string(str(img_dict['crs']))
        img_dict['crs'] = 'EPSG:' + str(crs_list.to_epsg())
    else:
        img_dict['crs'] = 'EPSG:4326'

    # calculate spatial resolution
    pixelSizeX, pixelSizeY = img.res
    img_dict['spatial_resolution']['pixel_size_x'] = pixelSizeX
    img_dict['spatial_resolution']['pixel_size_y'] = pixelSizeY

    # calculate spatial coverage
    # Bounding box (in the original CRS)
    bounds = img.bounds

    xmin, ymin, xmax, ymax = transform_bounds(CRS.from_string(img_dict['crs']), CRS.from_epsg(4326), *bounds)

    geom = box(xmin, ymin, xmax, ymax)
    img_dict['spatial_coverage'] = geom.wkt

    img_dict['bands'] = []
    # statistics for each band
    for band in range(1, img.count + 1):
        band_data = img.read(band).reshape(1, img.meta['width'] * img.meta['height'])[0].T

        # find band name
        if list(img.descriptions):
            band_name = img.descriptions[band - 1]
            if band_name is None:
                band_name = 'undefined'
        else:
            band_name = 'undefined'

        # find band statistics
        s = pd.Series(band_data)
        stats = s.describe(percentiles=[.10, .25, .75, .90])

        band_dict = {
            'uuid': str(uuid.uuid4()),
            'name': band_name,
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

        img_dict['bands'].append(band_dict)

    profile_dict['variables'].append(img_dict)

    # End time
    now = datetime.now()
    end_string = now.strftime("%Y-%m-%d %H:%M:%S.%f")
    profile_dict['analysis']['date_end'] = end_string

    # Time Difference
    profile_dict['analysis']['duration'] = str(
        dateutil.parser.parse(profile_dict['analysis']['date_end']) - dateutil.parser.parse(
            profile_dict['analysis']['date_start']))

    return profile_dict


# ----------- MULTIPLE IMAGES -----------#
def profile_multiple_rasters(my_folder_path: str, image_format: str = '.tif') -> dict:
    """
    This method performs profiling and generates a profiling dictionary for the image files that exist in the given folder path.

    :param my_folder_path: the path to a folder that has image files.
    :type my_folder_path: str
    :param image_format: the suffix of the images that exist in the given folder path.
    :type image_format: str, optional
    :return: A dict which contains the results of the profiler for the images.
    :rtype: dict

    """
    if os.path.isfile(my_folder_path):
        print('The input is not a folder!')
        return dict()

    profile_dict = {
        'analysis': {
            'title': 'Profiling Report',
            'date_start': '',
            'date_end': '',
            'duration': '',
            'filenames': []
        },
        'table': {
            'profiler_type': 'Raster',
            'byte_size': 0,
            'n_of_imgs': 0,
            'avg_width': 0,
            'avg_height': 0,
            'combined_band_stats': []
        },
        'variables': [], 'package': {
            'pandas_profiling_version': 'v3.5.0',
            'pandas_profiling_config': ''
        }
    }

    # in dictionary if same band name in more than one images
    band_images = dict()

    # Start time
    now = datetime.now()
    start_string = now.strftime("%Y-%m-%d %H:%M:%S.%f")
    profile_dict['analysis']['date_start'] = start_string

    for image in os.listdir(my_folder_path):
        if image.lower().endswith(image_format.lower()):
            my_file_path = my_folder_path + '/' + image

            profile_dict['analysis']['filenames'].append(my_file_path)

            # Files size
            profile_dict['table']['byte_size'] += os.path.getsize(my_file_path)

            # Increase the number of images
            profile_dict['table']['n_of_imgs'] += 1

            # Create image dictionary
            img_dict = {
                'name': '',
                'type': 'Raster',
                'crs': '',
                'spatial_coverage': '',
                'spatial_resolution': {
                    'pixel_size_x': 0,
                    'pixel_size_y': 0
                },
                'no_data_value': '',
                'format': ''
            }

            # Read image
            img = rio.open(my_file_path)

            # find image name
            name = Path(my_file_path).stem
            img_dict['name'] = name

            # find general image data
            img_dict.update(img.meta)

            # making transform JSON-serializable
            img_dict['transform'] = list(img_dict['transform'])

            profile_dict['table']['avg_width'] += img_dict['width']
            profile_dict['table']['avg_height'] += img_dict['height']

            # change nodata and driver keys
            img_dict['no_data_value'] = img_dict['nodata']
            del img_dict['nodata']

            img_dict['format'] = img_dict['driver']
            del img_dict['driver']

            # find tags
            img_dict['tags'] = []

            for k, v in img.tags().items():
                tag_dict = {
                    'key': k,
                    'value': v
                }

                img_dict['tags'].append(tag_dict)

            # change crs format
            if img.crs is not None:
                crs_list = CRS.from_string(str(img_dict['crs']))
                img_dict['crs'] = 'EPSG:' + str(crs_list.to_epsg())
            else:
                img_dict['crs'] = 'EPSG:4326'

            # calculate spatial resolution
            pixelSizeX, pixelSizeY = img.res
            img_dict['spatial_resolution']['pixel_size_x'] = pixelSizeX
            img_dict['spatial_resolution']['pixel_size_y'] = pixelSizeY

            # calculate spatial coverage
            # Bounding box (in the original CRS)
            bounds = img.bounds

            xmin, ymin, xmax, ymax = transform_bounds(CRS.from_string(img_dict['crs']), CRS.from_epsg(4326), *bounds)

            geom = box(xmin, ymin, xmax, ymax)
            img_dict['spatial_coverage'] = geom.wkt

            img_dict['bands'] = []
            # statistics for each band
            for band in range(1, img.count + 1):
                band_data = img.read(band).reshape(1, img.meta['width'] * img.meta['height'])[0].T

                # find band name
                band_name = 'undefined'
                if list(img.descriptions):
                    band_name = img.descriptions[band - 1]
                    if band_name is None:
                        band_name = 'undefined'
                else:
                    band_name = 'undefined'

                # find band statistics
                s = pd.Series(band_data)
                stats = s.describe(percentiles=[.10, .25, .75, .90])

                band_dict = {
                    'uuid': str(uuid.uuid4()),
                    'name': band_name,
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

                img_dict['bands'].append(band_dict)

                if band_name != 'undefined':
                    if band_name not in band_images:
                        band_images[band_name] = [img_dict['name']]
                    else:
                        band_images[band_name].append(img_dict['name'])

            profile_dict['variables'].append(img_dict)

    # calculate combined_band_stats
    for k, v in band_images.items():
        if len(v) > 1:
            combined_band_dict = {
                'name': k,
                'n_of_imgs': len(v),
                'img_names': v,
                'count': 0,
                'min': math.inf,
                'average': 0,
                'max': -math.inf,
                'variance': 0
            }

            for image in profile_dict['variables']:
                if image['name'] in v:
                    for band in image['bands']:
                        if band['name'] == k:
                            combined_band_dict['count'] += band['count']
                            combined_band_dict['average'] += band['average'] * band['count']

                            if band['min'] < combined_band_dict['min']:
                                combined_band_dict['min'] = band['min']

                            if band['max'] > combined_band_dict['max']:
                                combined_band_dict['max'] = band['max']

                            break

            combined_band_dict['average'] = combined_band_dict['average'] / combined_band_dict['count']

            # calculate combined_variance
            # comb_var = (n*std1 + n*d_sqrt1 + m*std2 + m*d_sqrt2 + k*std3 + k*d_sqrt3)/ n + m + k
            for image in profile_dict['variables']:
                if image['name'] in v:
                    for band in image['bands']:
                        if band['name'] == k:
                            count = band['count']
                            std = band['stddev']
                            mean = band['average']
                            comb_mean = combined_band_dict['average']
                            d_sqrt = (mean - comb_mean) * (mean - comb_mean)

                            combined_band_dict['variance'] += count * std + count * d_sqrt

                            break

            combined_band_dict['variance'] = combined_band_dict['variance'] / combined_band_dict['count']

            profile_dict['table']['combined_band_stats'].append(combined_band_dict)

    # fill general image folder data
    profile_dict['table']['avg_width'] = profile_dict['table']['avg_width'] / profile_dict['table']['n_of_imgs']
    profile_dict['table']['avg_height'] = profile_dict['table']['avg_height'] / profile_dict['table']['n_of_imgs']

    # End time
    now = datetime.now()
    end_string = now.strftime("%Y-%m-%d %H:%M:%S.%f")
    profile_dict['analysis']['date_end'] = end_string

    # Time Difference
    profile_dict['analysis']['duration'] = str(
        dateutil.parser.parse(profile_dict['analysis']['date_end']) - dateutil.parser.parse(
            profile_dict['analysis']['date_start']))

    return profile_dict


# ----------- MAIN FUNCTION ----------#
def profile_raster_with_config(config: dict) -> None:
    """
    This method performs profiling on raster data and write the resulting profile dictionary based on a configuration dictionary.

    :param config: a dictionary with all configuration settings.
    :type config: dict
    :return: None.
    :rtype: None

    """
    input_dir_path = config['input']['path']
    input_file_name = ''
    if 'file' in config['input']:
        input_file_name = config['input']['file']
    output_dir_path = config['output']['path']
    output_json_name = config['output']['json']

    # Create input file path
    only_directory_path = False
    if input_file_name == '':
        my_path = os.path.abspath(input_dir_path)
        only_directory_path = True
    else:
        my_path = os.path.abspath(os.path.join(input_dir_path, input_file_name))

    # Create output file paths
    output_dir_path = os.path.abspath(output_dir_path)
    output_json_path = os.path.abspath(os.path.join(output_dir_path, output_json_name))

    # Run raster profile
    if only_directory_path:
        print('You are running raster profile for multiple image files!\n'
              'Please make sure you have the right format for the image files.')
        if 'format' not in config['input']:
            print("No format is specified so the default '.tif' is used.")
            image_format: str = '.tif'
        else:
            image_format: str = str(config['input']['format']).lower()
        profile_dict = profile_raster(my_path=my_path, image_format=image_format)
    else:
        profile_dict = profile_raster(my_path=my_path)

    # Write resulting profile dictionary
    write_to_json(profile_dict, output_json_path)


def profile_raster(my_path: str, image_format: str = '.tif') -> dict:
    """
    This method performs profiling and generates a profiling dictionary for either a single image or many images.

    :param my_path: the path to either an image file or a folder that has image files.
    :type my_path: str
    :param image_format: the suffix of the images that exist in the folder if the given path is a folder path.
    :type image_format: str, optional
    :return: A dict which contains the results of the profiler for the image or images.
    :rtype: dict

    """
    if os.path.isfile(my_path):
        profile_dict = profile_single_raster(my_path)
    elif os.path.isdir(my_path):
        profile_dict = profile_multiple_rasters(my_path, image_format)
    else:
        profile_dict = dict()

    return profile_dict


# -------------- TEXTUAL -------------#
# ----------- SINGLE TEXT -----------#
def profile_single_text(my_file_path: str) -> dict:
    """
    This method performs profiling and generates a profiling dictionary for a text file that exists in the given path.

    :param my_file_path: the path to a text file.
    :type my_file_path: str
    :return: A dict which contains the results of the profiler for the text.
    :rtype: dict

    """

    # Used in language detection
    def __get_lang_detector(nlp, name):
        return LanguageDetector(seed=2023)

    # Calculate TermFrequency and generate a matrix
    def __create_tf_matrix(freq_matrix):
        tf_matrix = {}

        for sent, f_table in freq_matrix.items():
            tf_table = {}

            count_words_in_sentence = len(f_table)
            for word, count in f_table.items():
                tf_table[word] = count / count_words_in_sentence

            tf_matrix[sent] = tf_table

        return tf_matrix

    # Create a table for documents per words
    def __create_documents_per_words(freq_matrix):
        word_per_doc_table = {}

        for sent, f_table in freq_matrix.items():
            for word, count in f_table.items():
                if word in word_per_doc_table:
                    word_per_doc_table[word] += 1
                else:
                    word_per_doc_table[word] = 1

        return word_per_doc_table

    # Calculate IDF and generate a matrix
    def __create_idf_matrix(freq_matrix, count_doc_per_words, total_documents):
        idf_matrix = {}

        for sent, f_table in freq_matrix.items():
            idf_table = {}

            for word in f_table.keys():
                idf_table[word] = math.log10(total_documents / float(count_doc_per_words[word]))

            idf_matrix[sent] = idf_table

        return idf_matrix

    # Calculate TF-IDF and generate a matrix
    def __create_tf_idf_matrix(tf_matrix, idf_matrix):
        tf_idf_matrix = {}

        for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):

            tf_idf_table = {}

            for (word1, value1), (word2, value2) in zip(f_table1.items(),
                                                        f_table2.items()):  # here, keys are the same in both the table
                tf_idf_table[word1] = float(value1 * value2)

            tf_idf_matrix[sent1] = tf_idf_table

        return tf_idf_matrix

    # Important Algorithm: score the sentences
    def __score_sentences(tf_idf_matrix) -> dict:
        """
        score a sentence by its word's TF
        Basic algorithm: adding the TF frequency of every non-stop word in a sentence divided by total no of words in a sentence.
        :rtype: dict
        """

        sentenceValue = {}

        for sent, f_table in tf_idf_matrix.items():
            total_score_per_sentence = 0

            count_words_in_sentence = len(f_table)
            for word, score in f_table.items():
                total_score_per_sentence += score

            if count_words_in_sentence != 0:
                sentenceValue[sent] = total_score_per_sentence / count_words_in_sentence
            else:
                sentenceValue[sent] = 0

        return sentenceValue

    # Find the threshold
    def __find_average_score(sentenceValue) -> int:
        """
        Find the average score from the sentence value dictionary
        :rtype: int
        """
        sumValues = 0
        for entry in sentenceValue:
            sumValues += sentenceValue[entry]

        # Average value of a sentence from original summary_text
        average = (sumValues / len(sentenceValue))

        return average

    # Important Algorithm: Generate the summary
    def __generate_summary(sentences, sentenceValue, threshold):
        sentence_count = 0
        summary = ''

        for sentence in sentences:
            if sentence[:15] in sentenceValue and sentenceValue[sentence[:15]] >= threshold:
                summary += " " + sentence
                sentence_count += 1

        return summary.strip()

    if os.path.isdir(my_file_path):
        print('The input is not a file!')
        return dict()

    profile_dict = {
        'analysis': {
            'title': 'Profiling Report',
            'date_start': '',
            'date_end': '',
            'duration': '',
            'filenames': [my_file_path]
        },
        'table': {
            'profiler_type': 'Textual',
            'num_texts': 1,
            'num_words': 0,
            'num_sentences': 0,
            'num_distinct_words': 0,
            'num_characters': 0,
            'ratio_uppercase': 0,
            'ratio_digits': 0,
            'ratio_special_characters': 0,
            'language': '',
            'language_distribution': [],
            'sentiment': 0,
            'named_entities': [],
            'term_frequency': []

        },
        'variables': [],
        'package': {
            'pandas_profiling_version': 'v3.5.0',
            'pandas_profiling_config': ''
        }
    }

    now = datetime.now()
    start_string = now.strftime("%Y-%m-%d %H:%M:%S.%f")
    profile_dict['analysis']['date_start'] = start_string

    with open(my_file_path, 'r+') as text:
        text_dict = {
            'name': '',
            'type': 'Text',
            'num_words': 0,
            'num_sentences': 0,
            'num_distinct_words': 0,
            'num_characters': 0,
            'ratio_uppercase': 0,
            'ratio_digits': 0,
            'ratio_special_characters': 0,
            'language': '',
            'language_distribution': [],
            'summary': '',
            'topics': [],
            'sentiment': 0,
            'named_entities': [],
            'term_frequency': [],
            'special_characters_distribution': [],
            'sentence_length_distribution': dict(),
            'word_length_distribution': dict(),
        }

        # key is a special character and how many times is has been found in the text
        special_chars = {}

        # add the length of each word in the list to be used in the calculation of word_length_distribution
        word_length_list = []

        # add the length of each sentence in the list to be used in the calculation of sentence_length_distribution
        sentence_length_list = []

        # find text name
        pattern = '[\w-]+?(?=\.)'
        # searching the pattern
        a = re.search(pattern, my_file_path)

        text_dict['name'] = a.group()

        file_contents = text.read()
        file_contents = ' '.join(file_contents.split())
        string_encode = file_contents.encode("ascii", "ignore")
        file_contents = string_encode.decode()

        # Find number of words
        words = nltk.word_tokenize(file_contents.lower())
        words_count = 0
        for word in words:
            words_count += 1
            word_length_list.append(len(word))
        profile_dict['table']['num_words'] = words_count
        text_dict['num_words'] = words_count

        # Find number of sentences
        sentences = nltk.sent_tokenize(file_contents)
        sentences_count = 0
        for sentence in sentences:
            sentences_count += 1
            sentence_length_list.append(len(sentence))
        profile_dict['table']['num_sentences'] = sentences_count
        text_dict['num_sentences'] = sentences_count

        # Find Distinct/Unique words
        unique_words = sorted(set(words))
        unique_words_count = len(unique_words)
        # set_of_unique_words.update(unique_words)
        profile_dict['table']['num_distinct_words'] = unique_words_count
        text_dict['num_distinct_words'] = unique_words_count

        # Find number of characters
        numCharacters = len(file_contents)
        text_dict['num_characters'] = numCharacters
        profile_dict['table']['num_characters'] = numCharacters

        # ratio_uppercase, ratio_digits, ratio_special_characters
        ratioUppercase = 0
        ratioDigits = 0
        ratioSpecialChars = 0
        for c in file_contents:
            if c.isupper():
                ratioUppercase += 1
            if c.isdigit():
                ratioDigits += 1
            if not c.isalnum():
                ratioSpecialChars += 1
                if c not in special_chars:
                    special_chars[c] = 1
                else:
                    special_chars[c] += 1

        text_dict['ratio_uppercase'] = ratioUppercase / numCharacters
        text_dict['ratio_digits'] = ratioDigits / numCharacters
        text_dict['ratio_special_characters'] = ratioSpecialChars / numCharacters
        profile_dict['table']['ratio_uppercase'] = text_dict['ratio_uppercase']
        profile_dict['table']['ratio_digits'] = text_dict['ratio_digits']
        profile_dict['table']['ratio_special_characters'] = text_dict['ratio_special_characters']

        # Find languages
        try:
            nlp = spacy.load('en_core_web_sm')
        except OSError:
            print('Downloading language model for the spaCy POS tagger\n'
                  "(don't worry, this will only happen once)")
            from spacy.cli import download
            download('en')
            nlp = spacy.load('en_core_web_sm')
        if not Language.has_factory("language_detector"):
            Language.factory("language_detector", func=__get_lang_detector)
        nlp.add_pipe('language_detector', last=True)
        doc = nlp(file_contents)

        languages = {}
        cleaned_text = ' '
        lemma_text = ' '
        freq_matrix = Counter()
        for i, sent in enumerate(doc.sents):
            if sent.text:
                sentence = sent.text
                if pycountry.languages.get(alpha_2=sent._.language['language']) is not None:
                    language = pycountry.languages.get(alpha_2=sent._.language['language']).name.lower()
                else:
                    language = 'english'
                length_sent = len(sentence)
                if language not in languages:
                    languages[language] = float(sent._.language[
                                                    'score'] * length_sent / sentences_count * numCharacters)
                else:
                    languages[language] += float(sent._.language[
                                                     'score'] * length_sent / sentences_count * numCharacters)

                # Clean the sentence using the detecting language
                # Punctuation Removal
                cleaned_sentence = sentence.lower()
                for val in string.punctuation:
                    if val not in "'":
                        if val in "-":
                            cleaned_sentence = cleaned_sentence.replace(val, " ")
                        else:
                            cleaned_sentence = cleaned_sentence.replace(val, "")
                cleaned_sentence = ' '.join(cleaned_sentence.split()).strip()

                words = cleaned_sentence.split()

                # Stopword Removal
                if language in stopwords.fileids():
                    stop_words = set(stopwords.words(language))
                    cleaned_words = [w for w in words if not w in stop_words]
                else:
                    cleaned_words = words

                # Stemming
                stemmed_words = []
                if language in list(SnowballStemmer.languages):
                    stemmer = SnowballStemmer(language=language)
                    for word in cleaned_words:
                        word = stemmer.stem(word)
                        stemmed_words.append(word)
                else:
                    stemmed_words = cleaned_words

                # Lemma
                lemmatized_words = []
                if pycountry.languages.get(name=language) is not None:
                    for word in cleaned_words:
                        word = lemmatize(word, pycountry.languages.get(name=language).alpha_2)
                        lemmatized_words.append(word)
                else:
                    lemmatized_words = cleaned_words

                # freq_matrix will be used in summary extraction
                freq_matrix[sentence[:15]] = dict(Counter(stemmed_words))

                # add stemmed sentence to the cleaned_text
                cleaned_sentence = " ".join(stemmed_words)
                cleaned_text += cleaned_sentence.strip()
                cleaned_text += ' '

                # lemmatized text will be used in topic extraction
                lemmatized_text = " ".join(lemmatized_words)
                lemma_text += lemmatized_text.strip()
                lemma_text += ' '

        # Normalize language percentages
        total = sum(languages.values(), float(0))
        n_languages = {k: v * 100 / total for k, v in languages.items()}
        languages = n_languages
        # Find language most used in the text
        text_dict['language'] = max(languages, key=languages.get)
        profile_dict['table']['language'] = text_dict['language']

        # calculate language_distribution where all languages have percentages based on the sentences each language was detected
        total = sum(languages.values(), float(0))
        unknown_language_perc = 100
        for k, v in languages.items():
            if total >= 100:
                new_v = v * 100 / total
                text_dict['language_distribution'].append(
                    {'name': text_dict['name'], 'language': k, "percentage": new_v})
                profile_dict['table']['language_distribution'].append({'language': k, "percentage": new_v})
            else:
                text_dict['language_distribution'].append({'name': text_dict['name'], 'language': k, "percentage": v})
                profile_dict['table']['language_distribution'].append({'language': k, "percentage": v})
                unknown_language_perc -= v

        # Summary Extraction
        if len(file_contents.replace(" ", "")) > 300:
            '''
            Term frequency (TF) is how often a word appears in a document, divided by how many words are there in a document.
            '''
            # Calculate TermFrequency and generate a matrix
            tf_matrix = __create_tf_matrix(freq_matrix)
            # creating table for documents per words
            count_doc_per_words = __create_documents_per_words(freq_matrix)

            '''
            Inverse document frequency (IDF) is how unique or rare a word is.
            '''
            # Calculate IDF and generate a matrix
            idf_matrix = __create_idf_matrix(freq_matrix, count_doc_per_words, sentences_count)

            # Calculate TF-IDF and generate a matrix
            tf_idf_matrix = __create_tf_idf_matrix(tf_matrix, idf_matrix)

            # Important Algorithm: score the sentences
            sentence_scores = __score_sentences(tf_idf_matrix)

            # Find the threshold
            threshold = __find_average_score(sentence_scores)

            # Important Algorithm: Generate the summary
            summary = __generate_summary(sentences, sentence_scores, 1.8 * threshold)
            if not summary:
                summary = __generate_summary(sentences, sentence_scores, threshold)
                text_dict['summary'] = summary
            else:
                text_dict['summary'] = summary
        else:
            text_dict['summary'] = file_contents

        # Topic Extraction
        corpus = [lemma_text.split(' ')]

        dic = gensim.corpora.Dictionary(corpus)
        bow_corpus = [dic.doc2bow(doc) for doc in corpus]

        lda_model = gensim.models.LdaModel(bow_corpus,
                                           num_topics=1,
                                           id2word=dic,
                                           passes=100,
                                           iterations=100,
                                           random_state=2023,
                                           alpha='asymmetric')

        text_dict['topics'] = list(
            [token for token, score in lda_model.show_topic(i, topn=10)] for i in
            range(0, lda_model.num_topics))[0]

        # Sentiment Analysis
        sia = SentimentIntensityAnalyzer()
        compound_score = sia.polarity_scores(file_contents)['compound']

        text_dict['sentiment'] = compound_score
        profile_dict['table']['sentiment'] = compound_score

        # Named Entity Extraction
        named_entities = {}
        for X in doc.ents:
            sentence = X.text
            for val in string.punctuation:
                if val not in "'":
                    if val in "-":
                        sentence = sentence.replace(val, " ")
                    else:
                        sentence = sentence.replace(val, "")
            sentence = ' '.join(sentence.split()).strip()

            named_entities[sentence] = X.label_

        for ne, neType in named_entities.items():
            text_dict['named_entities'].append({'named_entity': ne, "type": neType})
            profile_dict['table']['named_entities'].append({'named_entity': ne, "type": neType})

        # Term Frequency
        data_analysis = dict(
            sorted(nltk.FreqDist(nltk.word_tokenize(cleaned_text)).items(), key=lambda item: item[1], reverse=True))

        for term, v in data_analysis.items():
            text_dict['term_frequency'].append({'name': text_dict['name'], 'term': term, "count": v})
            profile_dict['table']['term_frequency'].append({'term': term, "count": v})

        # text_dict['term_frequency'] = data_analysis
        # profile_dict['table']['term_frequency'] = data_analysis

        # calculate special_characters_distribution (FrequencyDistr)
        for k, v in special_chars.items():
            text_dict['special_characters_distribution'].append({'name': text_dict['name'], 'type': k, "count": v})

        # calculate sentence_length_distribution
        s = pd.Series(sentence_length_list)
        stats = s.describe(percentiles=[.10, .25, .75, .90])

        text_dict['sentence_length_distribution'] = {
            'name': text_dict['name'],
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

        # calculate word_length_distribution
        s = pd.Series(word_length_list)
        stats = s.describe(percentiles=[.10, .25, .75, .90])

        text_dict['word_length_distribution'] = {
            'name': text_dict['name'],
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

        profile_dict['variables'].append(text_dict)

    now = datetime.now()
    end_string = now.strftime("%Y-%m-%d %H:%M:%S.%f")
    profile_dict['analysis']['date_end'] = end_string

    profile_dict['analysis']['duration'] = str(
        dateutil.parser.parse(profile_dict['analysis']['date_end']) - dateutil.parser.parse(
            profile_dict['analysis']['date_start']))

    return profile_dict


# ----------- MULTIPLE TEXTS -----------#
def profile_multiple_texts(my_folder_path: str, text_format: str = 'txt') -> dict:
    """
    This method performs profiling and generates a profiling dictionary for the text files that exist in the given folder path.

    :param my_folder_path: the path to a folder that has text files.
    :type my_folder_path: str
    :param text_format: the suffix of the texts that exist in the given folder path.
    :type text_format: str, optional
    :return: A dict which contains the results of the profiler for the texts.
    :rtype: dict

    """

    # Used in language detection
    def __get_lang_detector(nlp, name):
        return LanguageDetector(seed=2023)

    # Calculate TermFrequency and generate a matrix
    def __create_tf_matrix(freq_matrix):
        tf_matrix = {}

        for sent, f_table in freq_matrix.items():
            tf_table = {}

            count_words_in_sentence = len(f_table)
            for word, count in f_table.items():
                tf_table[word] = count / count_words_in_sentence

            tf_matrix[sent] = tf_table

        return tf_matrix

    # Create a table for documents per words
    def __create_documents_per_words(freq_matrix):
        word_per_doc_table = {}

        for sent, f_table in freq_matrix.items():
            for word, count in f_table.items():
                if word in word_per_doc_table:
                    word_per_doc_table[word] += 1
                else:
                    word_per_doc_table[word] = 1

        return word_per_doc_table

    # Calculate IDF and generate a matrix
    def __create_idf_matrix(freq_matrix, count_doc_per_words, total_documents):
        idf_matrix = {}

        for sent, f_table in freq_matrix.items():
            idf_table = {}

            for word in f_table.keys():
                idf_table[word] = math.log10(total_documents / float(count_doc_per_words[word]))

            idf_matrix[sent] = idf_table

        return idf_matrix

    # Calculate TF-IDF and generate a matrix
    def __create_tf_idf_matrix(tf_matrix, idf_matrix):
        tf_idf_matrix = {}

        for (sent1, f_table1), (sent2, f_table2) in zip(tf_matrix.items(), idf_matrix.items()):

            tf_idf_table = {}

            for (word1, value1), (word2, value2) in zip(f_table1.items(),
                                                        f_table2.items()):  # here, keys are the same in both the table
                tf_idf_table[word1] = float(value1 * value2)

            tf_idf_matrix[sent1] = tf_idf_table

        return tf_idf_matrix

    # Important Algorithm: score the sentences
    def __score_sentences(tf_idf_matrix) -> dict:
        """
        score a sentence by its word's TF
        Basic algorithm: adding the TF frequency of every non-stop word in a sentence divided by total no of words in a sentence.
        :rtype: dict
        """

        sentenceValue = {}

        for sent, f_table in tf_idf_matrix.items():
            total_score_per_sentence = 0

            count_words_in_sentence = len(f_table)
            for word, score in f_table.items():
                total_score_per_sentence += score

            if count_words_in_sentence != 0:
                sentenceValue[sent] = total_score_per_sentence / count_words_in_sentence
            else:
                sentenceValue[sent] = 0

        return sentenceValue

    # Find the threshold
    def __find_average_score(sentenceValue) -> int:
        """
        Find the average score from the sentence value dictionary
        :rtype: int
        """
        sumValues = 0
        for entry in sentenceValue:
            sumValues += sentenceValue[entry]

        # Average value of a sentence from original summary_text
        average = (sumValues / len(sentenceValue))

        return average

    # Important Algorithm: Generate the summary
    def __generate_summary(sentences, sentenceValue, threshold):
        sentence_count = 0
        summary = ''

        for sentence in sentences:
            if sentence[:15] in sentenceValue and sentenceValue[sentence[:15]] >= threshold:
                summary += " " + sentence
                sentence_count += 1

        return summary.strip()

    if os.path.isfile(my_folder_path):
        print('The input is not a folder!')
        return dict()

    profile_dict = {
        'analysis': {
            'title': 'Profiling Report',
            'date_start': '',
            'date_end': '',
            'duration': '',
            'filenames': []
        },
        'table': {
            'profiler_type': 'Textual',
            'num_texts': 0,
            'num_words': 0,
            'num_sentences': 0,
            'num_distinct_words': 0,
            'num_characters': 0,
            'ratio_uppercase': 0,
            'ratio_digits': 0,
            'ratio_special_characters': 0,
            'language': '',
            'language_distribution': [],
            'sentiment': 0,
            'sentiment_analysis': {
                'compound_mean': 0.0,
                'compound_levels': {
                    '(-1, -0.5)': 0,
                    '(-0.5, 0)': 0,
                    '(0, 0.5)': 0,
                    '(0.5, 1)': 0
                }
            },
            'term_frequency': []

        },
        'variables': [],
        'package': {
            'pandas_profiling_version': 'v3.5.0',
            'pandas_profiling_config': ''
        }
    }

    now = datetime.now()
    start_string = now.strftime("%Y-%m-%d %H:%M:%S.%f")
    profile_dict['analysis']['date_start'] = start_string

    corpus_languages = dict()
    set_of_unique_words = set()
    dict_term_freq = dict()
    compound_scores = {
        '(-1, -0.5)': 0,
        '(-0.5, 0)': 0,
        '(0, 0.5)': 0,
        '(0.5, 1)': 0
    }

    for text_file in os.listdir(my_folder_path):
        if text_file.lower().endswith(text_format.lower()):
            filepath = my_folder_path + '/' + text_file
            profile_dict['analysis']['filenames'].append(filepath)
            with open(filepath, 'r+') as text:
                text_dict = {
                    'name': text_file.split('.')[0],
                    'type': 'Text',
                    'num_words': 0,
                    'num_sentences': 0,
                    'num_distinct_words': 0,
                    'num_characters': 0,
                    'ratio_uppercase': 0,
                    'ratio_digits': 0,
                    'ratio_special_characters': 0,
                    'language': '',
                    'language_distribution': [],
                    'summary': '',
                    'topics': [],
                    'sentiment': 0,
                    'named_entities': [],
                    'term_frequency': [],
                    'special_characters_distribution': [],
                    'sentence_length_distribution': dict(),
                    'word_length_distribution': dict(),
                }

                # key is a special character and how many times is has been found in the text
                special_chars = {}

                # add the length of each word in the list to be used in the calculation of word_length_distribution
                word_length_list = []

                # add the length of each sentence in the list to be used in the calculation of sentence_length_distribution
                sentence_length_list = []

                file_contents = text.read()
                file_contents = ' '.join(file_contents.split())
                string_encode = file_contents.encode("ascii", "ignore")
                file_contents = string_encode.decode()

                if file_contents:
                    profile_dict['table']['num_texts'] += 1

                    # Find number of words
                    words = nltk.word_tokenize(file_contents.lower())
                    words_count = 0
                    for word in words:
                        words_count += 1
                        word_length_list.append(len(word))
                    profile_dict['table']['num_words'] += words_count
                    text_dict['num_words'] = words_count

                    # Find number of sentences
                    sentences = nltk.sent_tokenize(file_contents)
                    sentences_count = 0
                    for sentence in sentences:
                        sentences_count += 1
                        sentence_length_list.append(len(sentence))
                    profile_dict['table']['num_sentences'] += sentences_count
                    text_dict['num_sentences'] = sentences_count

                    # Find Distinct/Unique words
                    unique_words = sorted(set(words))
                    unique_words_count = len(unique_words)
                    set_of_unique_words.update(unique_words)
                    text_dict['num_distinct_words'] = unique_words_count

                    # Find number of characters
                    numCharacters = len(file_contents)
                    text_dict['num_characters'] = numCharacters
                    profile_dict['table']['num_characters'] += numCharacters

                    # ratio_uppercase, ratio_digits, ratio_special_characters
                    ratioUppercase = 0
                    ratioDigits = 0
                    ratioSpecialChars = 0
                    for c in file_contents:
                        if c.isupper():
                            ratioUppercase += 1
                        if c.isdigit():
                            ratioDigits += 1
                        if not c.isalnum():
                            ratioSpecialChars += 1
                            if c not in special_chars:
                                special_chars[c] = 1
                            else:
                                special_chars[c] += 1

                    text_dict['ratio_uppercase'] = ratioUppercase / numCharacters
                    text_dict['ratio_digits'] = ratioDigits / numCharacters
                    text_dict['ratio_special_characters'] = ratioSpecialChars / numCharacters
                    profile_dict['table']['ratio_uppercase'] += ratioUppercase
                    profile_dict['table']['ratio_digits'] += ratioDigits
                    profile_dict['table']['ratio_special_characters'] += ratioSpecialChars

                    # Find languages
                    try:
                        nlp = spacy.load('en_core_web_sm')
                    except OSError:
                        print('Downloading language model for the spaCy POS tagger\n'
                              "(don't worry, this will only happen once)")
                        from spacy.cli import download
                        download('en')
                        nlp = spacy.load('en_core_web_sm')
                    if not Language.has_factory("language_detector"):
                        Language.factory("language_detector", func=__get_lang_detector)
                    nlp.add_pipe('language_detector', last=True)
                    doc = nlp(file_contents)

                    languages = {}
                    cleaned_text = ''
                    lemma_text = ''
                    freq_matrix = Counter()
                    for i, sent in enumerate(doc.sents):
                        if sent.text:
                            sentence = sent.text
                            if pycountry.languages.get(alpha_2=sent._.language['language']) is not None:
                                language = pycountry.languages.get(alpha_2=sent._.language['language']).name.lower()
                            else:
                                language = 'english'
                            length_sent = len(sentence)
                            if language not in languages:
                                languages[language] = float(sent._.language[
                                                                'score'] * length_sent / sentences_count * numCharacters)
                            else:
                                languages[language] += float(sent._.language[
                                                                 'score'] * length_sent / sentences_count * numCharacters)

                            # Clean the sentence using the detecting language
                            # Punctuation Removal
                            cleaned_sentence = sentence.lower()
                            for val in string.punctuation:
                                if val not in "'":
                                    if val in "-":
                                        cleaned_sentence = cleaned_sentence.replace(val, " ")
                                    else:
                                        cleaned_sentence = cleaned_sentence.replace(val, "")
                            cleaned_sentence = ' '.join(cleaned_sentence.split()).strip()

                            words = cleaned_sentence.split()

                            # Stopword Removal
                            if language in stopwords.fileids():
                                stop_words = set(stopwords.words(language))
                                cleaned_words = [w for w in words if not w in stop_words]
                            else:
                                cleaned_words = words

                            # Stemming
                            stemmed_words = []
                            if language in list(SnowballStemmer.languages):
                                stemmer = SnowballStemmer(language=language)
                                for word in cleaned_words:
                                    word = stemmer.stem(word)
                                    stemmed_words.append(word)
                            else:
                                stemmed_words = cleaned_words

                            # Lemma
                            lemmatized_words = []
                            if pycountry.languages.get(name=language) is not None:
                                for word in cleaned_words:
                                    word = lemmatize(word, pycountry.languages.get(name=language).alpha_2)
                                    lemmatized_words.append(word)
                            else:
                                lemmatized_words = cleaned_words

                            # freq_matrix will be used in summary extraction
                            freq_matrix[sentence[:15]] = dict(Counter(stemmed_words))

                            # add stemmed sentence to the cleaned_text
                            cleaned_sentence = " ".join(stemmed_words)
                            cleaned_text += cleaned_sentence.strip()
                            cleaned_text += ' '

                            # lemmatized text will be used in topic extraction
                            lemmatized_text = " ".join(lemmatized_words)
                            lemma_text += lemmatized_text.strip()
                            lemma_text += ' '

                    # Normalize language percentages
                    total = sum(languages.values(), float(0))
                    n_languages = {k: v * 100 / total for k, v in languages.items()}
                    languages = n_languages

                    # Add languages dictionary to the corpus dictionary
                    if corpus_languages is not {}:
                        corpus_languages = dict(Counter(corpus_languages) + Counter(languages))
                    else:
                        corpus_languages = languages

                    # Find language most used in the text
                    text_dict['language'] = max(languages, key=languages.get)

                    # calculate language_distribution where all languages have percentages based on the sentences each language was detected
                    total = sum(languages.values(), float(0))
                    unknown_language_perc = 100
                    for k, v in languages.items():
                        if total >= 100:
                            new_v = v * 100 / total
                            text_dict['language_distribution'].append(
                                {'name': text_dict['name'], 'language': k, "percentage": new_v})
                        else:
                            text_dict['language_distribution'].append(
                                {'name': text_dict['name'], 'language': k, "percentage": v})
                            unknown_language_perc -= v

                    # Summary Extraction
                    if len(file_contents.replace(" ", "")) > 300:
                        '''
                        Term frequency (TF) is how often a word appears in a document, divided by how many words are there in a document.
                        '''
                        # Calculate TermFrequency and generate a matrix
                        tf_matrix = __create_tf_matrix(freq_matrix)
                        # creating table for documents per words
                        count_doc_per_words = __create_documents_per_words(freq_matrix)

                        '''
                        Inverse document frequency (IDF) is how unique or rare a word is.
                        '''
                        # Calculate IDF and generate a matrix
                        idf_matrix = __create_idf_matrix(freq_matrix, count_doc_per_words, sentences_count)

                        # Calculate TF-IDF and generate a matrix
                        tf_idf_matrix = __create_tf_idf_matrix(tf_matrix, idf_matrix)

                        # Important Algorithm: score the sentences
                        sentence_scores = __score_sentences(tf_idf_matrix)

                        # Find the threshold
                        threshold = __find_average_score(sentence_scores)

                        # Important Algorithm: Generate the summary
                        summary = __generate_summary(sentences, sentence_scores, 1.8 * threshold)
                        if not summary:
                            summary = __generate_summary(sentences, sentence_scores, threshold)
                            text_dict['summary'] = summary
                        else:
                            text_dict['summary'] = summary
                    else:
                        text_dict['summary'] = file_contents

                    # Topic Extraction
                    corpus = [lemma_text.split(' ')]

                    dic = gensim.corpora.Dictionary(corpus)
                    bow_corpus = [dic.doc2bow(doc) for doc in corpus]

                    lda_model = gensim.models.LdaModel(bow_corpus,
                                                       num_topics=1,
                                                       id2word=dic,
                                                       passes=100,
                                                       iterations=100,
                                                       random_state=2023,
                                                       alpha='asymmetric')

                    text_dict['topics'] = list(
                        [token for token, score in lda_model.show_topic(i, topn=10)] for i in
                        range(0, lda_model.num_topics))[0]

                    # Sentiment Analysis
                    sia = SentimentIntensityAnalyzer()
                    compound_score = sia.polarity_scores(file_contents)['compound']

                    text_dict['sentiment'] = compound_score
                    profile_dict['table']['sentiment'] += compound_score

                    if compound_score > 0:
                        if compound_score >= 0.5:
                            compound_scores['(0.5, 1)'] += 1
                        else:
                            compound_scores['(0, 0.5)'] += 1
                    elif compound_score < 0:
                        if compound_score <= -0.5:
                            compound_scores['(-1, -0.5)'] += 1
                        else:
                            compound_scores['(-0.5, 0)'] += 1

                    profile_dict['table']['sentiment_analysis']['compound_mean'] += compound_score

                    # Named Entity Extraction
                    named_entities = {}
                    for X in doc.ents:
                        sentence = X.text
                        for val in string.punctuation:
                            if val not in "'":
                                if val in "-":
                                    sentence = sentence.replace(val, " ")
                                else:
                                    sentence = sentence.replace(val, "")
                        sentence = ' '.join(sentence.split()).strip()

                        named_entities[sentence] = X.label_

                    for ne, neType in named_entities.items():
                        text_dict['named_entities'].append({'named_entity': ne, "type": neType})

                    # Term Frequency
                    data_analysis = dict(
                        sorted(nltk.FreqDist(nltk.word_tokenize(cleaned_text)).items(), key=lambda item: item[1],
                               reverse=True))

                    dict_term_freq = dict(Counter(dict_term_freq) + Counter(data_analysis))

                    for term, v in data_analysis.items():
                        text_dict['term_frequency'].append({'term': term, "count": v})

                    # calculate special_characters_distribution (FrequencyDistr)
                    for k, v in special_chars.items():
                        text_dict['special_characters_distribution'].append(
                            {'name': text_dict['name'], 'type': k, "count": v})

                    # calculate sentence_length_distribution
                    s = pd.Series(sentence_length_list)
                    stats = s.describe(percentiles=[.10, .25, .75, .90])

                    text_dict['sentence_length_distribution'] = {
                        'name': text_dict['name'],
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

                    # calculate word_length_distribution
                    s = pd.Series(word_length_list)
                    stats = s.describe(percentiles=[.10, .25, .75, .90])

                    text_dict['word_length_distribution'] = {
                        'name': text_dict['name'],
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

                    profile_dict['variables'].append(text_dict)

    # Calculate number of distinct words in the corpus
    profile_dict['table']['num_distinct_words'] = len(set_of_unique_words)

    # Calculate ratio_uppercase, ratio_digits, ratio_special_characters in the corpus
    profile_dict['table']['ratio_uppercase'] /= profile_dict['table']['num_characters']
    profile_dict['table']['ratio_digits'] /= profile_dict['table']['num_characters']
    profile_dict['table']['ratio_special_characters'] /= profile_dict['table']['num_characters']

    # Calculate language distribution in the corpus
    languages = {k: v / profile_dict['table']['num_texts'] for k, v in corpus_languages.items()}
    total = sum(languages.values(), float(0))
    unknown_language_perc = 100
    for k, v in languages.items():
        if total >= 100:
            new_v = v * 100 / total
            profile_dict['table']['language_distribution'].append({'language': k, "percentage": new_v})
        else:
            profile_dict['table']['language_distribution'].append({'language': k, "percentage": v})
            unknown_language_perc -= v

    if total < 100:
        profile_dict['table']['language_distribution'].append(
            {'language': "unknown", "percentage": unknown_language_perc})

    # Calculate Sentiment analysis for the corpus
    profile_dict['table']['sentiment'] /= profile_dict['table']['num_texts']
    profile_dict['table']['sentiment_analysis']['compound_levels'] = compound_scores
    profile_dict['table']['sentiment_analysis']['compound_mean'] /= profile_dict['table']['num_texts']

    # Calculate term frequency for the corpus
    data_analysis = dict(sorted(dict_term_freq.items(), key=lambda item: item[1], reverse=True))

    for term, v in data_analysis.items():
        profile_dict['table']['term_frequency'].append({'term': term, "count": v})

    now = datetime.now()
    end_string = now.strftime("%Y-%m-%d %H:%M:%S.%f")
    profile_dict['analysis']['date_end'] = end_string

    profile_dict['analysis']['duration'] = str(
        dateutil.parser.parse(profile_dict['analysis']['date_end']) - dateutil.parser.parse(
            profile_dict['analysis']['date_start']))

    return profile_dict


# ----------- MAIN FUNCTION ----------#
def profile_text_with_config(config: dict) -> None:
    """
    This method performs profiling on text data and write the resulting profile dictionary based on a configuration dictionary.

    :param config: a dictionary with all configuration settings.
    :type config: dict
    :return: None.
    :rtype: None

    """
    input_dir_path = config['input']['path']
    input_file_name = config['input']['file']
    output_dir_path = config['output']['path']
    output_json_name = config['output']['json']

    # Create input file path
    only_directory_path = False
    if input_file_name == '':
        my_path = os.path.abspath(input_dir_path)
        only_directory_path = True
    else:
        my_path = os.path.abspath(os.path.join(input_dir_path, input_file_name))

    # Create output file paths
    output_dir_path = os.path.abspath(output_dir_path)
    output_json_path = os.path.abspath(os.path.join(output_dir_path, output_json_name))

    # Run raster profile
    if only_directory_path:
        print('You are running text profile for multiple text files!\n'
              'Please make sure you have the right format for the text files.')
        if 'format' not in config['input']:
            print("No format is specified so the default '.txt' is used.")
            text_format: str = '.txt'
        else:
            text_format: str = str(config['input']['format']).lower()
        profile_dict = profile_text(my_path=my_path, text_format=text_format)
    else:
        profile_dict = profile_text(my_path=my_path)

    # Write resulting profile dictionary
    write_to_json(profile_dict, output_json_path)


def profile_text(my_path: str, text_format: str = '.txt'):
    """
    This method performs profiling and generates a profiling dictionary for either a single text or many texts.

    :param my_path: the path to either a text file or a folder that has text files.
    :type my_path: str
    :param text_format: the suffix of the texts that exist in the folder if the given path is a folder path.
    :type text_format: str, optional
    :return: A dict which contains the results of the profiler for the text or texts.
    :rtype: dict

    """
    if os.path.isfile(my_path):
        profile_dict = profile_single_text(my_path)
    elif os.path.isdir(my_path):
        profile_dict = profile_multiple_texts(my_path, text_format)
    else:
        profile_dict = dict()

    return profile_dict


# ---------- HIERARCHICAL ---------#
def profile_hierarchical_with_config(config: dict) -> None:
    """
    This method performs profiling on hierarchical data and write the resulting profile dictionary based on a configuration dictionary.

    :param config: a dictionary with all configuration settings.
    :type config: dict
    :return: None.
    :rtype: None

    """
    input_dir_path = config['input']['path']
    input_file_name = config['input']['file']
    output_dir_path = config['output']['path']
    output_json_name = config['output']['json']

    # Create input file path
    my_file_path = ''
    if input_file_name == '':
        print('No input file was found for hierarchical profile!')
        return None
    else:
        my_file_path = os.path.abspath(os.path.join(input_dir_path, input_file_name))

    # Create output file paths
    output_dir_path = os.path.abspath(output_dir_path)
    output_json_path = os.path.abspath(os.path.join(output_dir_path, output_json_name))

    # Run raster profile
    profile_dict = profile_hierarchical(my_file_path=my_file_path)

    # Write resulting profile dictionary
    write_to_json(profile_dict, output_json_path)


# TODO: Add num_attributes (number of distinct tags)
def profile_hierarchical(my_file_path: str) -> dict:
    """
    This method performs profiling and generates a profiling dictionary for a given json file that exists in the given path.

    :param my_file_path: the path to a json file.
    :type my_file_path: str
    :return: A dict which contains the results of the profiler for the json.
    :rtype: dict

    """
    profile_dict = {
        'analysis': {
            'title': 'Profiling Report',
            'date_start': '',
            'date_end': '',
            'duration': '',
            'filenames': [my_file_path]
        },
        'table': {
            'profiler_type': 'Hierarchical',
            'byte_size': 0,
            'num_records': 0,
            'depth_distribution': dict()
        },
        'variables': [],
        'package': {
            'pandas_profiling_version': 'v3.5.0',
            'pandas_profiling_config': ''
        }

    }

    now = datetime.now()
    start_string = now.strftime("%Y-%m-%d %H:%M:%S.%f")
    profile_dict['analysis']['date_start'] = start_string

    # File size
    profile_dict['table']['byte_size'] = os.path.getsize(my_file_path)

    data = Data(my_file_path)
    profile = Profiler(data, profiler_type='structured')
    readable_report = profile.report(report_options={'output_format': 'pretty'})

    profile_dict['table']['num_records'] = readable_report['global_stats']['column_count']
    depth = dict()

    variables = readable_report['data_stats']

    for var in variables:
        attr = {
            'name': var['column_name'],
            'type': var['data_type'],
            'uniqueness': var['statistics']['unique_ratio'],
            'nesting_level': 0
        }

        levels = var['column_name'].split('.')

        attr['nesting_level'] = len(levels) - 1

        for level in range(0, attr['nesting_level'] + 1):
            if level in depth.keys():
                depth[level].add(levels[level])
            else:
                depth[level] = {levels[level]}

        profile_dict['variables'].append(attr)

    unique_levels = []
    for level, names in depth.items():
        for name in names:
            unique_levels.append(level)

    s = pd.Series(unique_levels)
    stats = s.describe(percentiles=[.10, .25, .75, .90])

    profile_dict['table']['depth_distribution'] = {
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

    now = datetime.now()
    end_string = now.strftime("%Y-%m-%d %H:%M:%S.%f")
    profile_dict['analysis']['date_end'] = end_string

    profile_dict['analysis']['duration'] = str(
        dateutil.parser.parse(profile_dict['analysis']['date_end']) - dateutil.parser.parse(
            profile_dict['analysis']['date_start']))

    return profile_dict


# ---------- RDF-GRAPH ---------#
def profile_rdfGraph_with_config(config: dict) -> None:
    """
    This method performs profiling on rdfGraph data and write the resulting profile dictionary based on a configuration dictionary.

    :param config: a dictionary with all configuration settings.
    :type config: dict
    :return: None.
    :rtype: None

    """
    input_dir_path = config['input']['path']
    input_file_name = config['input']['file']
    output_dir_path = config['output']['path']
    output_json_name = config['output']['json']

    # Create input file path
    my_file_path = ''
    if input_file_name == '':
        print('No input file was found for rdfGraph profile!')
        return None
    else:
        my_file_path = os.path.abspath(os.path.join(input_dir_path, input_file_name))

    # Create output file paths
    output_dir_path = os.path.abspath(output_dir_path)
    output_json_path = os.path.abspath(os.path.join(output_dir_path, output_json_name))

    # Run raster profile
    if 'serialization' not in config['input']:
        print("No rdflib format is specified so the default 'application/rdf+xml' is used.")
        parse_format: str = 'application/rdf+xml'
    else:
        parse_format: str = str(config['input']['serialization']).lower()
    profile_dict = profile_rdfGraph(my_file_path=my_file_path, parse_format=parse_format)

    # Write resulting profile dictionary
    write_to_json(profile_dict, output_json_path)


def profile_rdfGraph(my_file_path: str, parse_format: str = 'application/rdf+xml'):
    """
    This method performs profiling and generates a profiling dictionary for a given rdf file that exists in the given path.

    :param my_file_path: the path to a rdf file.
    :type my_file_path: str
    :param parse_format: the format of the rdf file. (see rdflib package to find the available formats e.g. 'turtle', 'application/rdf+xml', 'n3', 'nt', etc.)
    :type parse_format: str, optional
    :return: A dict which contains the results of the profiler for the rdf.
    :rtype: dict

    """

    # Calculate the number of nodes
    def __calc_num_nodes(g: Graph):
        return len(g.all_nodes())

    # Calculate the number of edges
    def __calc_num_edges(g: Graph):
        return len(g)

    # Calculate the number of namespaces
    def __calc_num_namespaces(g: Graph):
        v = g.serialize(format="ttl")

        return v.count('@prefix')

    # Calculate the number of classes and a class frequency list
    def __calc_class_features(g: Graph):

        num_classes = set()
        classes_distribution = dict()

        for cl in g.objects(predicate=RDF.type):
            if str(cl) not in classes_distribution:
                classes_distribution[str(cl)] = 0

            classes_distribution[str(cl)] += 1

            num_classes.add(str(cl))

        # List of classes and their frequencies in the graph
        class_distribution_list = []

        for c, v in sorted(classes_distribution.items(), key=lambda x: x[1], reverse=True):
            class_dict = dict({
                'class_name': c,
                'count': v
            })
            class_distribution_list.append(class_dict)

        return len(num_classes), class_distribution_list

    # Calculate the number of object type properties
    def __calc_num_object_properties(g: Graph):
        # Extract set from objects of triples
        object_list = {x for x in g.objects() if isinstance(x, URIRef)}
        # Append set extracted from subjects of triples
        object_list.update({x for x in g.subjects() if isinstance(x, URIRef)})

        return len(object_list)

    # Calculate the number of data type properties
    def __calc_num_datatype_properties(g: Graph):
        data_property_list = {x for x in g.objects() if not isinstance(x, URIRef)}

        return len(data_property_list)

    # Calculate the number of connected components and a list with each connected component and its number of nodes
    def __calc_cc_features(nx_g: nx.MultiDiGraph):
        nx_g_undirected = nx_g.to_undirected()
        cc = list(nx.connected_components(nx_g_undirected))

        cc_list = []

        for i, c in enumerate(cc):
            cc_dict = dict({
                'component_name': i,
                'num_nodes': len(c)
            })
            cc_list.append(cc_dict)

        return len(cc), cc_list

    # Calculate the density of the graph
    def __calc_density(nx_g: nx.MultiDiGraph):
        nx_g_density = nx.density(nx_g)

        return nx_g_density

    # Calculate the degree_centrality_distribution
    def __calc_degree_centrality(nx_g: nx.MultiDiGraph):

        dc = nx.degree_centrality(nx_g)
        degrees_centrality = []
        for _, v in dc.items():
            degrees_centrality.append(v)

        s = pd.Series(degrees_centrality)
        stats = s.describe(percentiles=[.10, .25, .75, .90])

        degree_centrality_distribution = {
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

        return degree_centrality_distribution

    # Calculate the degree_distribution
    def __calc_degree(nx_g: nx.MultiDiGraph):
        degrees = []
        for _, v in nx_g.degree:
            degrees.append(v)

        s = pd.Series(degrees)
        stats = s.describe(percentiles=[.10, .25, .75, .90])

        degree_distribution = {
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

        return degree_distribution

    # Calculate the in_degree_distribution
    def __calc_in_degree(nx_g: nx.MultiDiGraph):
        in_degrees = []
        for _, v in nx_g.in_degree:
            in_degrees.append(v)

        s = pd.Series(in_degrees)
        stats = s.describe(percentiles=[.10, .25, .75, .90])

        in_degrees_distribution = {
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

        return in_degrees_distribution

    # Calculate the out_degree_distribution
    def __calc_out_degree(nx_g: nx.MultiDiGraph):
        out_degrees = []
        for _, v in nx_g.out_degree:
            out_degrees.append(v)

        s = pd.Series(out_degrees)
        stats = s.describe(percentiles=[.10, .25, .75, .90])

        out_degrees_distribution = {
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

        return out_degrees_distribution

    profile_dict = {
        'analysis': {
            'title': 'Profiling Report',
            'date_start': '',
            'date_end': '',
            'duration': '',
            'filenames': [my_file_path]
        },
        'table': {
            'profiler_type': 'RDFGraph',
            'byte_size': 0,
            'num_nodes': 0,
            'num_edges': 0,
            'num_namespaces': 0,
            'num_classes': 0,
            'num_object_properties': 0,
            'num_datatype_properties': 0,
            'density': 0,
            'num_connected_components': 0,
            'connected_components': [],
            'degree_centrality_distribution': dict(),
            'degree_distribution': dict(),
            'in_degree_distribution': dict(),
            'out_degree_distribution': dict(),
            'class_distribution': []

        },
        'variables': [],
        'package': {
            'pandas_profiling_version': 'v3.5.0',
            'pandas_profiling_config': ''
        }

    }

    # Start time
    now = datetime.now()
    start_string = now.strftime("%Y-%m-%d %H:%M:%S.%f")
    profile_dict['analysis']['date_start'] = start_string

    # File size
    profile_dict['table']['byte_size'] = os.path.getsize(my_file_path)

    g = Graph()
    g.parse(my_file_path, format=parse_format)

    # Number of nodes
    profile_dict['table']['num_nodes'] = __calc_num_nodes(g)

    # Number of edges
    profile_dict['table']['num_edges'] = __calc_num_edges(g)

    # Number of namespaces
    profile_dict['table']['num_namespaces'] = __calc_num_namespaces(g)

    # Number of Classes + class_distribution
    profile_dict['table']['num_classes'], profile_dict['table']['class_distribution'] = __calc_class_features(g)

    # Number of Object type properties
    profile_dict['table']['num_object_properties'] = __calc_num_object_properties(g)

    # Number of Data type properties
    profile_dict['table']['num_datatype_properties'] = __calc_num_datatype_properties(g)

    # Create networkx graph
    nx_g = rdflib_to_networkx_multidigraph(g)

    # Number of connected components + List of connected components
    profile_dict['table']['num_connected_components'], profile_dict['table'][
        'connected_components'] = __calc_cc_features(
        nx_g)

    # Density
    profile_dict['table']['density'] = __calc_density(nx_g)

    # Calculate degree_centrality_distribution
    profile_dict['table']['degree_centrality_distribution'] = __calc_degree_centrality(nx_g)

    # Calculate degree_distribution
    profile_dict['table']['degree_distribution'] = __calc_degree(nx_g)

    # Calculate in_degree_distribution
    profile_dict['table']['in_degree_distribution'] = __calc_in_degree(nx_g)

    # Calculate out_degree_distribution
    profile_dict['table']['out_degree_distribution'] = __calc_out_degree(nx_g)

    # End time
    now = datetime.now()
    end_string = now.strftime("%Y-%m-%d %H:%M:%S.%f")
    profile_dict['analysis']['date_end'] = end_string

    # Time Difference
    profile_dict['analysis']['duration'] = str(
        dateutil.parser.parse(profile_dict['analysis']['date_end']) - dateutil.parser.parse(
            profile_dict['analysis']['date_start']))

    return profile_dict


# ------ VISTA (RHD, RAS FILES) ------#
def profile_vista_rasters_with_config(config: dict) -> None:
    """
    This method performs profiling on ras data and write the resulting profile dictionary based on a configuration dictionary.

    :param config: a dictionary with all configuration settings.
    :type config: dict
    :return: None.
    :rtype: None

    """
    input_path = config['input']['path']
    input_ras_file = config['input']['ras_file']
    input_rhd_file = config['input']['rhd_file']
    output_dir_path = config['output']['path']
    output_json_name = config['output']['json']

    # Create input ras and rhd file paths
    my_ras_file_path = ''
    if input_ras_file == '':
        print('No input ras file was found for vista profile!')
        return None
    else:
        my_ras_file_path = os.path.abspath(os.path.join(input_path, input_ras_file))

    my_rhd_file_path = ''
    if input_rhd_file == '':
        print('No input rhd file was found for vista profile!')
        return None
    else:
        my_rhd_file_path = os.path.abspath(os.path.join(input_path, input_rhd_file))

    # Create output file paths
    output_dir_path = os.path.abspath(output_dir_path)
    output_json_path = os.path.abspath(os.path.join(output_dir_path, output_json_name))

    # Run raster profile
    profile_dict = profile_vista_rasters(rhd_datapath=my_rhd_file_path, ras_datapath=my_ras_file_path)

    # Write resulting profile dictionary
    write_to_json(profile_dict, output_json_path)


def profile_vista_rasters(rhd_datapath: str, ras_datapath: str):
    """
    This method performs profiling and generates a profiling dictionary for a given ras file
    that exists in the given path using the contents of a rhd file that exists in the given path.

    :param rhd_datapath: the path to a rhd file.
    :type rhd_datapath: str
    :param ras_datapath: the path to a ras file.
    :type ras_datapath: str
    :return: A dict which contains the results of the profiler for the ras.
    :rtype: dict

    """

    def __read_image_rhd(rhd_datapath: str):
        with open(rhd_datapath, 'r') as f:
            lines = f.readlines()
            vista_data_type = int(lines[0])
            n_of_LAI = int(lines[1])
            split_third_row = " ".join(lines[2].split()).split(' ')
            columns = int(split_third_row[0])
            rows = int(split_third_row[1])
            split_fourth_row = " ".join(lines[3].split()).split(' ')
            resolution = float(split_fourth_row[0])
            upper_left_corner_x = float(split_fourth_row[1])
            upper_left_corner_y = float(split_fourth_row[2])
            UTM_x = float(split_fourth_row[3])
            UTM_y = float(split_fourth_row[4])
            UTM_zone = str(split_fourth_row[5])
            LAI_images = {'vista_data_type': vista_data_type, 'resolution': resolution,
                          'upper_left_corner_x': upper_left_corner_x, 'upper_left_corner_y': upper_left_corner_y,
                          'rows': rows, 'columns': columns, 'UTM_x': UTM_x, 'UTM_y': UTM_y, 'UTM_zone': UTM_zone}
            count_LAI_images = 0
            LAI_images['images'] = {}
            for value_LAI in range(5, n_of_LAI + 5):
                ras_file_name = rhd_datapath.split('/')[-1].split('.')[0]
                img_name = ras_file_name + '_' + str(count_LAI_images)
                prev_img_name = ras_file_name + '_' + str(count_LAI_images - 1)
                split_row = " ".join(lines[value_LAI].split()).split(' ')
                LAI_images['images'][img_name] = {}
                img_bytes = int(split_row[0])
                LAI_images['images'][img_name]['bytes'] = img_bytes
                LAI_images['images'][img_name]['date'] = datetime.strptime(
                    split_row[3] + ' ' + split_row[2] + ' ' + split_row[1], '%d %m %Y').date()

                record_length = img_bytes * columns
                LAI_images['images'][img_name]['record_length_bytes'] = record_length
                if count_LAI_images == 0:
                    LAI_images['images'][img_name]['image_start_pos_bytes'] = 0
                else:
                    LAI_images['images'][img_name]['image_start_pos_bytes'] = LAI_images['images'][prev_img_name][
                                                                                  'image_start_pos_bytes'] + ((
                                                                                                                      record_length / img_bytes) * rows)
                count_LAI_images += 1

            return LAI_images

    ras_dict = __read_image_rhd(rhd_datapath)

    profile_dict = {
        'analysis': {
            'title': 'Profiling Report',
            'date_start': '',
            'date_end': '',
            'duration': '',
            'filenames': [rhd_datapath,
                          ras_datapath]
        },
        'table': {
            'profiler_type': 'Vista_Raster',
            'byte_size': 0,
            'n_of_imgs': len(ras_dict['images']),
            'avg_width': 0,
            'avg_height': 0,
            'combined_bands': []
        },
        'variables': [], 'package': {
            'pandas_profiling_version': 'v3.5.0',
            'pandas_profiling_config': ''
        }
    }

    # initialize .ras NODATA value counts
    ras_zero_count = 0
    ras_missing_count = 0
    ras_forest_count = 0
    ras_urban_count = 0
    ras_water_count = 0
    ras_snow_count = 0
    ras_cloud_shadow_buffer_count = 0
    ras_cloud_shadow_count = 0
    ras_cloud_buffer_count = 0
    ras_cirrus_clouds_count = 0
    ras_clouds_count = 0

    __lai_f = lambda x: float(str(x)) / 1000 if (x > 0) else x

    # Start time
    now = datetime.now()
    start_string = now.strftime("%Y-%m-%d %H:%M:%S.%f")
    profile_dict['analysis']['date_start'] = start_string

    img_names = []
    imgs = []
    lai_in_imgs = []
    with open(ras_datapath, 'r+') as f:
        ras_file_name = ras_datapath.split('/')[-1].split('.')[0]
        if ras_dict['vista_data_type'] == 7:
            ras_file_array = np.fromfile(f, dtype=np.int16).astype(float)
            ras_file_array[np.where(ras_file_array > 0)] = list(
                map(__lai_f, ras_file_array[np.where(ras_file_array > 0)]))
            n_of_imgs = len(ras_dict['images'])

            for n_img in range(0, n_of_imgs):

                # Create image dictionary
                img_dict = {
                    'name': '',
                    'type': 'Raster',
                    'crs': '',
                    'date': '',
                    'spatial_coverage': '',
                    'spatial_resolution': {
                        'pixel_size_x': 0,
                        'pixel_size_y': 0
                    },
                    'no_data_value': '',
                    'format': ''
                }

                img_name = ras_file_name + '_' + str(n_img)
                img_names.append(img_name)

                # image name
                img_dict['name'] = img_name

                next_img_name = ras_file_name + '_' + str(n_img + 1)
                if n_img == n_of_imgs - 1:
                    start_pos = int(ras_dict['images'][img_name]['image_start_pos_bytes'])
                    end_pos = len(ras_file_array)
                else:
                    start_pos = int(ras_dict['images'][img_name]['image_start_pos_bytes'])
                    end_pos = int(ras_dict['images'][next_img_name]['image_start_pos_bytes'])

                # data of the image
                img_data = ras_file_array[start_pos:end_pos]
                img_data = img_data.reshape((ras_dict['rows'], ras_dict['columns']))

                # Find Image General Data
                upper_left_corner_x = ras_dict['upper_left_corner_x']
                upper_left_corner_y = ras_dict['upper_left_corner_y']
                x_res = ras_dict['resolution']
                y_res = ras_dict['resolution']
                transform = from_origin(upper_left_corner_x, upper_left_corner_y, x_res, y_res)

                # create in-memory rasterio image
                mem_file = MemoryFile()

                with mem_file.open(driver='GTiff', height=ras_dict['rows'],
                                   width=ras_dict['columns'], count=1,
                                   dtype=str(ras_file_array.dtype), crs='+proj=utm +zone=' + str(ras_dict['UTM_zone']),
                                   transform=transform) as img:

                    img.update_tags(date=ras_dict['images'][img_name]['date'])

                    # image general metadata
                    img_dict.update(img.meta)

                    # image size
                    profile_dict['table']['byte_size'] += img_dict['width'] * img_dict['height'] * 4

                    # image date
                    img_dict['date'] = ras_dict['images'][img_name]['date'].strftime("%d.%m.%Y")

                    # making transform JSON-serializable
                    img_dict['transform'] = list(img_dict['transform'])

                    profile_dict['table']['avg_width'] += img_dict['width']
                    profile_dict['table']['avg_height'] += img_dict['height']

                    # change nodata and driver keys
                    img_dict['no_data_value'] = img_dict['nodata']
                    del img_dict['nodata']

                    img_dict['format'] = img_dict['driver']
                    del img_dict['driver']

                    # change crs format
                    if img.crs is not None:
                        crs_list = CRS.from_string(str(img_dict['crs']))
                        img_dict['crs'] = 'EPSG:' + str(crs_list.to_epsg())
                    else:
                        img_dict['crs'] = 'EPSG:4326'

                    # calculate spatial resolution
                    pixelSizeX, pixelSizeY = img.res
                    img_dict['spatial_resolution']['pixel_size_x'] = pixelSizeX
                    img_dict['spatial_resolution']['pixel_size_y'] = pixelSizeY

                    # calculate spatial coverage
                    # Bounding box (in the original CRS)
                    bounds = img.bounds

                    xmin, ymin, xmax, ymax = transform_bounds(CRS.from_string(img_dict['crs']), CRS.from_epsg(4326),
                                                              *bounds)

                    geom = box(xmin, ymin, xmax, ymax)

                    img_dict['spatial_coverage'] = geom.wkt

                    img.close()

                # statistics for LAI band
                img_dict['bands'] = []
                s = pd.Series(img_data[np.where(img_data > 0)])
                stats = s.describe(percentiles=[.10, .25, .75, .90])

                band_uuid = str(uuid.uuid4())

                band_dict = {
                    'uuid': band_uuid,
                    'name': 'LAI',
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
                    'no_data_distribution': []
                }

                # percentages of no_data values
                img_no_data = img_data[np.where(img_data < 0)]
                width = img_dict['width']
                height = img_dict['height']

                missing_count = np.count_nonzero(img_no_data == -999)
                forest_count = np.count_nonzero(img_no_data == -961)
                urban_count = np.count_nonzero(img_no_data == -950)
                water_count = np.count_nonzero(img_no_data == -940)
                snow_count = np.count_nonzero(img_no_data == -930)
                cloud_shadow_buffer_count = np.count_nonzero(img_no_data == -923)
                cloud_shadow_count = np.count_nonzero(img_no_data == -920)
                cloud_buffer_count = np.count_nonzero(img_no_data == -913)
                cirrus_clouds_count = np.count_nonzero(img_no_data == -911)
                clouds_count = np.count_nonzero(img_no_data == -910)

                img_zeros = img_data[np.where(img_data == 0)]
                zero_count = img_zeros.size

                # add NODATA value counts to the .ras NODATA value counts
                ras_missing_count += missing_count
                ras_forest_count += forest_count
                ras_urban_count += urban_count
                ras_water_count += water_count
                ras_snow_count += snow_count
                ras_cloud_shadow_buffer_count += cloud_shadow_buffer_count
                ras_cloud_shadow_count += cloud_shadow_count
                ras_cloud_buffer_count += cloud_buffer_count
                ras_cirrus_clouds_count += cirrus_clouds_count
                ras_clouds_count += clouds_count

                # add zero value counts to the .ras zero value counts
                ras_zero_count += zero_count

                no_data_dict = {
                    'LAI': (band_dict['count'] / (width * height)) * 100,
                    'missing': (missing_count / (width * height)) * 100,
                    'forest': (forest_count / (width * height)) * 100,
                    'urban': (urban_count / (width * height)) * 100,
                    'water': (water_count / (width * height)) * 100,
                    'snow': (snow_count / (width * height)) * 100,
                    'cloud_shadow_buffer': (cloud_shadow_buffer_count / (width * height)) * 100,
                    'cloud_shadow': (cloud_shadow_count / (width * height)) * 100,
                    'cloud_buffer': (cloud_buffer_count / (width * height)) * 100,
                    'cirrus_clouds': (cirrus_clouds_count / (width * height)) * 100,
                    'clouds': (clouds_count / (width * height)) * 100,
                    'zeros': (zero_count / (width * height)) * 100
                }

                for k, v in no_data_dict.items():
                    band_dict['no_data_distribution'].append(
                        {'uuid': band_uuid, 'value': k, 'percentage': v}
                    )

                    if k == 'LAI':
                        imgs.append({'raster': img_dict['name'],
                                     'date': img_dict['date'],
                                     'percentage': no_data_dict['LAI']})

                        lai_in_imgs.append(no_data_dict['LAI'])

                img_dict['bands'].append(band_dict)

                profile_dict['variables'].append(img_dict)

            # calculate combined stats
            combined_band_stats_dict = {
                'name': 'LAI',
                'n_of_imgs': profile_dict['table']['n_of_imgs'],
                'img_names': img_names,
                'imgs': imgs,
                'count': 0,
                'min': math.inf,
                'average': 0,
                'max': -math.inf,
                'variance': 0,
                'no_data_distribution': [],
                'lai_distribution': {}
            }

            # calculate LAI numeric distribution for the images of the .ras
            s = pd.Series(lai_in_imgs)
            stats = s.describe(percentiles=[.10, .25, .75, .90])

            lai_dict = {
                'name': 'LAI',
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
                'percentile90': stats[8]
            }

            combined_band_stats_dict['lai_distribution'] = lai_dict

            for image in profile_dict['variables']:
                lai_band = image['bands'][0]
                if lai_band['count'] != 0:
                    combined_band_stats_dict['count'] += lai_band['count']
                    combined_band_stats_dict['average'] += lai_band['average'] * lai_band['count']

                    if lai_band['min'] < combined_band_stats_dict['min']:
                        combined_band_stats_dict['min'] = lai_band['min']

                    if lai_band['max'] > combined_band_stats_dict['max']:
                        combined_band_stats_dict['max'] = lai_band['max']

            combined_band_stats_dict['average'] = combined_band_stats_dict['average'] / combined_band_stats_dict[
                'count']

            # calculate combined_variance
            # comb_var = (n*std1 + n*d_sqrt1 + m*std2 + m*d_sqrt2 + k*std3 + k*d_sqrt3)/ n + m + k
            for image in profile_dict['variables']:
                lai_band = image['bands'][0]
                if lai_band['count'] != 0:
                    count = lai_band['count']
                    std = lai_band['stddev']
                    mean = lai_band['average']
                    comb_mean = combined_band_stats_dict['average']
                    d_sqrt = (mean - comb_mean) * (mean - comb_mean)

                    combined_band_stats_dict['variance'] += count * std + count * d_sqrt

            combined_band_stats_dict['variance'] = combined_band_stats_dict['variance'] / combined_band_stats_dict[
                'count']

            # calculate no_data_distribution for LAI of the .ras
            width_all = profile_dict['table']['avg_width']
            height_all = profile_dict['table']['avg_height']

            no_data_dict = {
                'LAI': ((combined_band_stats_dict['count'] * n_of_imgs) / (width_all * height_all)) * 100,
                'missing': ((ras_missing_count * n_of_imgs) / (width_all * height_all)) * 100,
                'forest': ((ras_forest_count * n_of_imgs) / (width_all * height_all)) * 100,
                'urban': ((ras_urban_count * n_of_imgs) / (width_all * height_all)) * 100,
                'water': ((ras_water_count * n_of_imgs) / (width_all * height_all)) * 100,
                'snow': ((ras_snow_count * n_of_imgs) / (width_all * height_all)) * 100,
                'cloud_shadow_buffer': ((ras_cloud_shadow_buffer_count * n_of_imgs) / (width_all * height_all)) * 100,
                'cloud_shadow': ((ras_cloud_shadow_count * n_of_imgs) / (width_all * height_all)) * 100,
                'cloud_buffer': ((ras_cloud_buffer_count * n_of_imgs) / (width_all * height_all)) * 100,
                'cirrus_clouds': ((ras_cirrus_clouds_count * n_of_imgs) / (width_all * height_all)) * 100,
                'clouds': ((ras_clouds_count * n_of_imgs) / (width_all * height_all)) * 100,
                'zeros': ((ras_zero_count * n_of_imgs) / (width_all * height_all)) * 100
            }

            for k, v in no_data_dict.items():
                combined_band_stats_dict['no_data_distribution'].append(
                    {'name': 'LAI', 'value': k, 'percentage': v}
                )

            profile_dict['table']['combined_bands'].append(combined_band_stats_dict)

            # calculate avg_width and avg_height of .ras file
            profile_dict['table']['avg_width'] = profile_dict['table']['avg_width'] / profile_dict['table']['n_of_imgs']
            profile_dict['table']['avg_height'] = profile_dict['table']['avg_height'] / profile_dict['table'][
                'n_of_imgs']

    # End time
    now = datetime.now()
    end_string = now.strftime("%Y-%m-%d %H:%M:%S.%f")
    profile_dict['analysis']['date_end'] = end_string

    # Time Difference
    profile_dict['analysis']['duration'] = str(
        dateutil.parser.parse(profile_dict['analysis']['date_end']) - dateutil.parser.parse(
            profile_dict['analysis']['date_start']))

    return profile_dict


# ---------- OTHER FUNCTIONS ---------#
def read_config(json_file: str) -> dict:
    """
    This method reads configuration settings from a json file. Configuration includes all parameters for input/output.

    :param json_file: path to .json file that contains the configuration parameters.
    :type json_file: str
    :return: A dictionary with all configuration settings.
    :rtype: dict

    """
    try:
        config_dict: dict = json.loads(json_file)
    except ValueError as e:
        with open(json_file) as f:
            config_dict: dict = json.load(f)
            return config_dict

    return config_dict


def write_to_json(output_dict: dict, output_file: Union[str, Path]) -> None:
    """
    Write the profile dictionary to a file.

    :param output_dict: the profile dictionary that will writen.
    :type output_dict: dict
    :param output_file: The name or the path of the file to generate including the extension (.json).
    :type output_file: Union[str, Path]
    :return: a dict which contains the results of the profiler for the texts.
    :rtype: dict

    """
    if not isinstance(output_file, Path):
        output_file = Path(str(output_file))

    # create image folder if it doesn't exist
    path = Path(str(output_file.parent))
    path.mkdir(parents=True, exist_ok=True)

    if output_file.suffix == ".json":
        with open(output_file, "w") as outfile:
            def encode_it(o: Any) -> Any:
                if isinstance(o, dict):
                    return {encode_it(k): encode_it(v) for k, v in o.items()}
                else:
                    if isinstance(o, (bool, int, float, str)):
                        return o
                    elif isinstance(o, list):
                        return [encode_it(v) for v in o]
                    elif isinstance(o, set):
                        return {encode_it(v) for v in o}
                    elif isinstance(o, (pd.DataFrame, pd.Series)):
                        return encode_it(o.reset_index().to_dict("records"))
                    elif isinstance(o, np.ndarray):
                        return encode_it(o.tolist())
                    elif isinstance(o, np.generic):
                        return o.item()
                    else:
                        return str(o)

            output_dict = encode_it(output_dict)
            json.dump(output_dict, outfile, indent=3)
    else:
        suffix = output_file.suffix
        warnings.warn(
            f"Extension {suffix} not supported. For now we assume .json was intended. "
            f"To remove this warning, please use .json."
        )


# --------------- READ ---------------#
def __read_files(my_file, header=None, sep=',', encoding='UTF-8'):
    try:
        df = pd.read_csv(my_file, header=header, sep=sep, encoding=encoding)
    except:
        return pd.DataFrame()

    return df


def __profile_timeseries_main(my_file_path: str, time_column: str, header: int = 0,
                              sep: str = ',', mode: str = "default", minimal: bool = True):
    df = __read_files(my_file_path, header, sep)
    df[time_column] = pd.to_datetime(df[time_column])

    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception as e:
                pass

    if minimal:
        config_file = get_config("config_minimal.yaml")

        with open(config_file) as f:
            data = yaml.safe_load(f)

        config: Settings = Settings().parse_obj(data)
    else:
        config: Settings = Settings()
    config.progress_bar = False
    config.vars.num.quantiles.append(0.10)
    config.vars.num.quantiles.append(0.90)
    sample_timeseries: Container = None
    html_dict = None
    if mode == 'default' and len(df.columns) > 2:
        sample_time_series = __create_sample_df(df, time_column)
        config_file = get_config("config_minimal.yaml")
        with open(config_file) as f:
            data = yaml.safe_load(f)

        new_config: Settings = Settings().parse_obj(data)

        new_config.progress_bar = False
        new_config.vars.timeseries.active = True
        # if autocorrelation test passes then numeric timeseries else 'real' numeric
        new_config.vars.timeseries.autocorrelation = 0.3
        typeset = ProfilingTypeSet(new_config)
        custom_summarizer = PandasProfilingSummarizer(typeset)
        custom_summarizer.mapping['TimeSeries'].append(__new_numeric_summary)
        profile = ProfileReport(sample_time_series, tsmode=True, title="Profiling Report", sortby=time_column,
                                summarizer=custom_summarizer, config=new_config, progress_bar=False)
        html_dict = profile.description_set
        html_dict['table']['profiler_type'] = 'TimeSeries'
        html_dict['analysis']['title'] = 'Profiling Report'
        html_dict['analysis']['filenames'] = list(my_file_path)

        # Create a container of timeseries samples which will be used in the html
        report = __get_report_structure(new_config, html_dict)
        variables = report.content['body'].content['items'][1]
        item = variables.content['item'].content['items']
        sample_timeseries = Container(
            item,
            sequence_type="accordion",
            name="Sample TimeSeries",
            anchor_id="sample-timeseries-variables",
        )

        # Fill missing values as tsfresh cannot handle them
        time_series_stacked = df.melt(id_vars=[time_column], value_vars=df.columns[1:],
                                      value_name='value', var_name='id')
        time_series_stacked = time_series_stacked.reindex(columns=[time_column, 'value', 'id'])

        time_series_stacked.rename(columns={time_column: 'time'}, inplace=True)

        time_series_stacked['time'] = pd.to_datetime(time_series_stacked['time']).apply(lambda x: x.value)

        if __is_not_finite(time_series_stacked['value']).any():
            time_series_stacked['value'] = __replace_missing_inf_values(time_series_stacked['value'])

        # Run tsfresh
        json_decoded = __read_json_file_tsfresh(tsfresh_json_file)
        ts_fresh_results = __ts_fresh_json(time_series_stacked, json_decoded, no_time=False)

        config.progress_bar = True
        profile = ProfileReport(ts_fresh_results, config=config, title="Profiling Report", minimal=minimal)
        html_dict = profile.description_set
        html_dict['table']['profiler_type'] = 'TimeSeries'
        html_dict['analysis']['title'] = 'Profiling Report'
        html_dict['analysis']['filenames'] = [my_file_path]
        # Files size
        html_dict['table']['byte_size'] = os.path.getsize(my_file_path)
    elif mode == 'verbose' or len(df.columns) == 2:
        config.vars.timeseries.active = True
        config.progress_bar = True
        # if autocorrelation test passes then numeric timeseries else 'real' numeric
        config.vars.timeseries.autocorrelation = 0.3
        typeset = ProfilingTypeSet(config)
        custom_summarizer = PandasProfilingSummarizer(typeset)
        custom_summarizer.mapping['TimeSeries'].append(__new_numeric_summary)
        profile = ProfileReport(df, tsmode=True, title="Profiling Report", sortby=time_column,
                                summarizer=custom_summarizer, config=config, progress_bar=True)
        html_dict = profile.description_set
        html_dict['table']['profiler_type'] = 'TimeSeries'
        html_dict['analysis']['title'] = 'Profiling Report'
        html_dict['analysis']['filenames'] = [my_file_path]
        # Files size
        html_dict['table']['byte_size'] = os.path.getsize(my_file_path)

    texts_column_names = []
    timeseries_columns = []
    for var_name, info in html_dict['variables'].items():
        if info['type'] == 'Categorical' and info['p_unique'] > 0.4:
            texts_column_names.append(var_name)
        if info['type'] == 'TimeSeries':
            timeseries_columns.append(var_name)

    if len(timeseries_columns) != 0:
        df_ts = df[timeseries_columns]
        html_dict = __calculate_gaps(html_dict, df_ts)

    if len(texts_column_names) != 0:
        df = df[texts_column_names]
        profile_dict = __create_profile_dict(html_dict, df)

        html_dict = __extend_textual_html(profile_dict, html_dict)
    else:
        profile_dict = __create_profile_dict(html_dict)

    return profile_dict, config, html_dict, sample_timeseries


def __calculate_gaps(html_dict: dict, df: pd.DataFrame = pd.DataFrame()):
    column_gap_dict = {}
    max_gap_all = -np.Inf
    min_gap_all = np.Inf
    average_gap_all = 0
    count_gaps = 0

    # Dictionary with gap size as key and count as value
    gaps_dict = Counter()

    for column in df:

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

            html_dict['variables'][column]['gaps_distribution'] = gaps_distribution

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

            column_gap_dict[column] = max_gap
        else:
            html_dict['variables'][column]['gaps_distribution'] = {}

    if count_gaps != 0:
        average_gap_all = round(average_gap_all / count_gaps)
    else:
        min_gap_all = 0
        max_gap_all = 0
    if len(gaps_dict) != 0:
        gaps_dict = dict(gaps_dict)

    html_dict['table']['ts_min_gap'] = min_gap_all
    html_dict['table']['ts_max_gap'] = max_gap_all
    html_dict['table']['ts_avg_gap'] = average_gap_all
    html_dict['table']['ts_gaps_frequency_distribution'] = gaps_dict

    return html_dict


def __create_sample_df(df, time_column):
    sample_time_series = df[[time_column]]
    temp_df = df.loc[:, df.columns != time_column]
    sample_count = 3
    if len(temp_df.columns) < sample_count:
        sample_count = len(temp_df.columns)
    for i in range(0, sample_count):
        sample_time_series[temp_df.columns[i]] = temp_df[temp_df.columns[i]]
    return sample_time_series


def __new_numeric_summary(config: Settings, series: pd.Series, summary: dict = None):
    if summary is None:
        summary = {}
    df = pd.DataFrame()
    dates_float = range(len(series))
    df['time'] = dates_float
    df['id'] = series.name
    df['value'] = series.values

    json_decoded = __read_json_file_tsfresh(tsfresh_json_file)
    ts_fresh_results = __ts_fresh_json(df, json_decoded, no_time=False)
    summary['tsfresh_features'] = ts_fresh_results.to_dict(orient='records')[0]
    return config, series, summary


# TODO: Add language distribution
def __extend_textual_attributes(texts_list: list, var_name: str, info: dict):
    # Used in language detection
    DetectorFactory.seed = 2023

    var_dict = {
        'name': var_name,
        'type': 'Textual',
        'count': info['count'],
        'num_missing': info['n_missing'],
        'uniqueness': info['p_unique'],
        'ratio_uppercase': 0,
        'ratio_digits': 0,
        'ratio_special_characters': 0,
        'num_chars_distribution': {},
        'num_words_distribution': {},
        'language_distribution': [],
        'n_distinct': info['n_distinct'],
        'p_distinct': info['p_distinct'],
        'p_missing': info['p_missing'],
        'memory_size': info['memory_size'],
        'n_unique': info['n_unique']
    }

    num_chars = 0
    ratio_uppercase = 0
    ratio_digits = 0
    ratio_special_characters = 0
    num_chars_list = []
    num_words_list = []
    corpus_languages = dict()

    for text in texts_list:
        if not pd.isnull(text):
            text_num_chars = len(text)
            num_chars += text_num_chars
            num_chars_list.append(text_num_chars)
            for c in text:
                if c.isupper():
                    ratio_uppercase += 1
                if c.isdigit():
                    ratio_digits += 1
                if not c.isalnum():
                    ratio_special_characters += 1

            words = nltk.word_tokenize(text.lower())
            words_count = 0
            for word in words:
                num_words_list.append(len(word))

            # Find number of sentences
            sentences = nltk.sent_tokenize(text)
            sentences_count = 0
            for sentence in sentences:
                sentences_count += 1

            # Find languages
            try:
                languages = detect_langs(text)

                for language in languages:
                    if pycountry.languages.get(alpha_2=language.lang) is not None:
                        lang = pycountry.languages.get(alpha_2=language.lang).name.lower()
                    else:
                        lang = 'english'

                    if lang not in corpus_languages:
                        corpus_languages[lang] = language.prob
                    else:
                        corpus_languages[lang] += language.prob

            except:
                language = detect(text)

                if pycountry.languages.get(alpha_2=language['lang']) is not None:
                    lang = pycountry.languages.get(alpha_2=language['lang']).name.lower()
                else:
                    lang = 'english'

                if lang not in corpus_languages:
                    corpus_languages[lang] = language['score']
                else:
                    corpus_languages[lang] += language['score']

    # Calculate language distribution in the corpus

    corpus_languages = {k: v / var_dict['count'] for k, v in corpus_languages.items()}
    total = sum(corpus_languages.values(), float(0)) * 100
    if total < 100:
        corpus_languages['unknown'] = (100 - total) / 100

    corpus_languages = dict(sorted(corpus_languages.items(), key=lambda item: item[1], reverse=True))

    for k, v in corpus_languages.items():
        var_dict['language_distribution'].append({'language': k, "percentage": v * 100})

    if num_chars != 0:
        var_dict['ratio_uppercase'] = ratio_uppercase / num_chars
        var_dict['ratio_digits'] = ratio_digits / num_chars
        var_dict['ratio_special_characters'] = ratio_special_characters / num_chars

    if len(num_chars_list) != 0:
        s = pd.Series(num_chars_list)
        stats = s.describe(percentiles=[.10, .25, .75, .90])

        var_dict['num_chars_distribution'] = {
            'name': var_name,
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

    if len(num_words_list) != 0:
        s = pd.Series(num_words_list)
        stats = s.describe(percentiles=[.10, .25, .75, .90])

        var_dict['num_words_distribution'] = {
            'name': var_name,
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

    return var_dict


def __create_profile_dict(html_dict: dict, df: pd.DataFrame = pd.DataFrame()):
    profile_dict = {
        'analysis': {
            'title': 'Profiling Report',
            'date_start': '',
            'date_end': '',
            'duration': '',
            'filenames': ''
        },
        'table': {
            'profiler_type': '',
            'byte_size': 0,
            'memory_size': 0,
            'record_size': 0,
            'num_rows': 0,
            'num_attributes': 0,
            'n_cells_missing': 0,
            'p_cells_missing': 0.0,
            'types': []
        },
        'variables': [],
        'package': html_dict['package']
    }

    # Fill analysis section
    start_string = str(html_dict['analysis']['date_start'])
    profile_dict['analysis']['date_start'] = start_string

    end_string = str(html_dict['analysis']['date_end'])
    profile_dict['analysis']['date_end'] = end_string

    profile_dict['analysis']['duration'] = str(html_dict['analysis']['duration'])

    profile_dict['analysis']['filenames'] = html_dict['analysis']['filenames']

    # Fill table section
    profile_dict['table']['profiler_type'] = html_dict['table']['profiler_type']
    profile_dict['table']['byte_size'] = html_dict['table']['byte_size']
    profile_dict['table']['num_rows'] = html_dict['table']['n']
    profile_dict['table']['num_attributes'] = html_dict['table']['n_var']
    profile_dict['table']['n_cells_missing'] = html_dict['table']['n_cells_missing']
    profile_dict['table']['p_cells_missing'] = html_dict['table']['p_cells_missing']
    profile_dict['table']['memory_size'] = html_dict['table']['memory_size']
    profile_dict['table']['record_size'] = html_dict['table']['record_size']

    profile_types = {}

    # Pass gaps in timeseries profile
    if profile_dict['table']['profiler_type'] == 'TimeSeries':
        profile_dict['table']['ts_min_gap'] = html_dict['table']['ts_min_gap']
        profile_dict['table']['ts_max_gap'] = html_dict['table']['ts_max_gap']
        profile_dict['table']['ts_avg_gap'] = html_dict['table']['ts_avg_gap']

        profile_dict['table']['ts_gaps_frequency_distribution'] = []
        gaps_freq_distr = []
        for gap, count in html_dict['table']['ts_gaps_frequency_distribution'].items():
            profile_dict['table']['ts_gaps_frequency_distribution'].append({'gap_size': gap,
                                                                            "count": count})

    # Fill variables
    for var_name, info in html_dict['variables'].items():
        if info['type'] == 'DateTime':

            if info['type'] in profile_types:
                profile_types[info['type']] += 1
            else:
                profile_types[info['type']] = 1

            var_dict = {
                'name': var_name,
                'type': 'DateTime',
                'count': info['count'],
                'n_distinct': info['n_distinct'],
                'p_distinct': info['p_distinct'],
                'num_missing': info['n_missing'],
                'uniqueness': info['p_unique'],
                'p_missing': info['p_missing'],
                'memory_size': info['memory_size'],
                'start': str(info['min']),
                'end': str(info['max']),
                'date_range': str(info['range']),
                'histogram_counts': info['histogram'][0],
                'histogram_bins': info['histogram'][1]
            }

            profile_dict['variables'].append(var_dict)
        elif info['type'] == 'TimeSeries':
            if info['type'] in profile_types:
                profile_types[info['type']] += 1
            else:
                profile_types[info['type']] = 1

            var_dict = {
                'name': var_name,
                'type': 'TimeSeries',
                'count': info['count'],
                'num_missing': info['n_missing'],
                'uniqueness': info['p_unique'],
                'min': info['min'],
                'max': info['max'],
                'average': info['mean'],
                'stddev': info['std'],
                'median': info['50%'],
                'kurtosis': info['kurtosis'],
                'skewness': info['skewness'],
                'variance': info['variance'],
                'percentile5': info['5%'],
                'percentile10': info['10%'],
                'percentile25': info['25%'],
                'percentile75': info['75%'],
                'percentile90': info['90%'],
                'percentile95': info['95%'],
                'seasonal': info['seasonal'],
                'stationary': info['stationary'],
                'add_fuller': info['addfuller'],
                'abs_energy': info['tsfresh_features']['abs energy'],
                'abs_sum_changes': info['tsfresh_features']['absolute sum of changes'],
                'len_above_mean': info['tsfresh_features']['count above mean'],
                'len_below_mean': info['tsfresh_features']['count below mean'],
                'num_peaks': info['tsfresh_features']['number cwt peaks  n 10'],
                'n_distinct': info['n_distinct'],
                'p_distinct': info['p_distinct'],
                'p_missing': info['p_missing'],
                'memory_size': info['memory_size'],
                'n_unique': info['n_unique'],
                'n_infinite': info['n_infinite'],
                'p_infinite': info['p_infinite'],
                'n_zeros': info['n_zeros'],
                'p_zeros': info['p_zeros'],
                'n_negative': info['n_negative'],
                'p_negative': info['p_negative'],
                'monotonic': info['monotonic'],
                'range': info['range'],
                'iqr': info['iqr'],
                'cv': info['cv'],
                'mad': info['mad'],
                'sum': info['sum'],
                'gaps_distribution': info['gaps_distribution'],
                'histogram_counts': info['histogram'][0],
                'histogram_bins': info['histogram'][1],
                'value_counts_without_nan': [],
                'value_counts_index_sorted': [],
                'series': []
            }

            for value, count in info['value_counts_without_nan'].items():
                var_dict['value_counts_without_nan'].append({'value': value, "count": count})

            for value, count in info['value_counts_index_sorted'].items():
                var_dict['value_counts_index_sorted'].append({'value': value, "count": count})

            for key, value in info['series'].items():
                var_dict['series'].append({'key': key, "value": value})

            profile_dict['variables'].append(var_dict)
        elif info['type'] == 'Numeric':
            if info['type'] in profile_types:
                profile_types[info['type']] += 1
            else:
                profile_types[info['type']] = 1

            var_dict = {
                'name': var_name,
                'type': 'Numeric',
                'count': info['count'],
                'num_missing': info['n_missing'],
                'uniqueness': info['p_unique'],
                'min': info['min'],
                'max': info['max'],
                'average': info['mean'],
                'stddev': info['std'],
                'median': info['50%'],
                'kurtosis': info['kurtosis'],
                'skewness': info['skewness'],
                'variance': info['variance'],
                'percentile5': info['5%'],
                'percentile10': info['10%'],
                'percentile25': info['25%'],
                'percentile75': info['75%'],
                'percentile90': info['90%'],
                'percentile95': info['95%'],
                'n_distinct': info['n_distinct'],
                'p_distinct': info['p_distinct'],
                'p_missing': info['p_missing'],
                'memory_size': info['memory_size'],
                'n_unique': info['n_unique'],
                'n_infinite': info['n_infinite'],
                'p_infinite': info['p_infinite'],
                'n_zeros': info['n_zeros'],
                'p_zeros': info['p_zeros'],
                'n_negative': info['n_negative'],
                'p_negative': info['p_negative'],
                'monotonic': info['monotonic'],
                'range': info['range'],
                'iqr': info['iqr'],
                'cv': info['cv'],
                'mad': info['mad'],
                'sum': info['sum'],
                'histogram_counts': info['histogram'][0],
                'histogram_bins': info['histogram'][1],
                'value_counts_without_nan': [],
                'value_counts_index_sorted': []
            }

            for value, count in info['value_counts_without_nan'].items():
                var_dict['value_counts_without_nan'].append({'value': value, "count": count})

            for value, count in info['value_counts_index_sorted'].items():
                var_dict['value_counts_index_sorted'].append({'value': value, "count": count})

            profile_dict['variables'].append(var_dict)
        elif info['type'] == 'Categorical':
            if info['p_unique'] > 0.4:
                if 'Textual' in profile_types:
                    profile_types['Textual'] += 1
                else:
                    profile_types['Textual'] = 1

                texts_list = df[var_name].to_list()
                var_dict = __extend_textual_attributes(texts_list, var_name, info)
            else:
                if info['type'] in profile_types:
                    profile_types[info['type']] += 1
                else:
                    profile_types[info['type']] = 1

                var_dict = {
                    'name': var_name,
                    'type': 'Categorical',
                    'count': info['count'],
                    'num_missing': info['n_missing'],
                    'uniqueness': info['p_unique'],
                    'frequency_distribution': [],
                    'n_distinct': info['n_distinct'],
                    'p_distinct': info['p_distinct'],
                    'p_missing': info['p_missing'],
                    'memory_size': info['memory_size'],
                    'n_unique': info['n_unique'],
                    'samples': []
                }

                for cat, count in info['first_rows'].items():
                    var_dict['samples'].append({'row': cat, "cat": count})

                for cat, count in info['value_counts_without_nan'].items():
                    var_dict['frequency_distribution'].append({'name': var_name, 'type': cat, 'count': count})

            profile_dict['variables'].append(var_dict)
        elif info['type'] == 'Geometry':
            if info['type'] in profile_types:
                profile_types[info['type']] += 1
            else:
                profile_types[info['type']] = 1

            var_dict = {
                'name': var_name,
                'type': 'Geometry',
                'count': info['count'],
                'num_missing': info['n_missing'],
                'uniqueness': info['p_unique'],
                'mbr': info['mbr'],
                'centroid': info['centroid'],
                'crs': info['crs'],
                'union_convex_hull': info['union_convex_hull'],
                'length_distribution': info['length_distribution'],
                'area_distribution': info['area_distribution'],
                'geom_type_distribution': [],
                'value_counts_without_nan': [],
                'n_distinct': info['n_distinct'],
                'p_distinct': info['p_distinct'],
                'p_missing': info['p_missing'],
                'memory_size': info['memory_size'],
                'n_unique': info['n_unique'],
                'samples': [],
                'heatmap': info['heatmap']

            }

            for geom_type, frequency in info['geom_types'].items():
                var_dict['geom_type_distribution'].append({'name': var_name, 'type': geom_type, 'count': frequency})

            for value, count in info['value_counts_without_nan'].items():
                var_dict['value_counts_without_nan'].append({'name': var_name, 'value': value, 'count': count})

            for row, value in info['first_rows'].items():
                var_dict['samples'].append({'row': row, "value": value})

            profile_dict['variables'].append(var_dict)
        elif info['type'] == 'Boolean':
            if info['type'] in profile_types:
                profile_types[info['type']] += 1
            else:
                profile_types[info['type']] = 1

            var_dict = {
                'name': var_name,
                'type': 'Boolean',
                'count': info['count'],
                'num_missing': info['n_missing'],
                'uniqueness': info['p_unique'],
                'value_counts_without_nan': [],
                'n_distinct': info['n_distinct'],
                'p_distinct': info['p_distinct'],
                'p_missing': info['p_missing'],
                'memory_size': info['memory_size'],
                'n_unique': info['n_unique'],
            }

            for value, count in info['value_counts_without_nan'].items():
                var_dict['value_counts_without_nan'].append({'name': var_name, 'value': value, 'count': count})

            profile_dict['variables'].append(var_dict)
        else:
            if info['type'] in profile_types:
                profile_types[info['type']] += 1
            else:
                profile_types[info['type']] = 1

            var_dict = {
                'name': var_name,
                'type': info['type'],
                'count': info['count'],
                'num_missing': info['n_missing'],
                'uniqueness': info['p_unique'],
                'p_missing': info['p_missing'],
                'memory_size': info['memory_size']
            }

            profile_dict['variables'].append(var_dict)

    for k, v in sorted(profile_types.items(), key=lambda x: x[1], reverse=True):
        profile_dict['table']['types'].append({'type': k, 'count': v})

    return profile_dict


def __extend_textual_html(profile_dict: dict, html_dict: dict):
    for variable in profile_dict['variables']:
        if variable['type'] == 'Textual':
            var_dict = {
                'type': variable['type'],
                'count': variable['count'],
                'num_missing': variable['num_missing'],
                'uniqueness': variable['uniqueness'],
                'ratio_uppercase': variable['ratio_uppercase'],
                'ratio_digits': variable['ratio_digits'],
                'ratio_special_characters': variable['ratio_special_characters'],
                'num_chars_distribution': variable['num_chars_distribution'],
                'num_words_distribution': variable['num_words_distribution'],
                'language_distribution': {language['language']: language['percentage']
                                          for language in variable['language_distribution']}
            }

            html_dict['variables'][variable['name']].update(var_dict)

            if not html_dict['table']['types'].__contains__('Textual'):
                html_dict['table']['types']['Categorical'] -= 1
                html_dict['table']['types']['Textual'] = 1
            else:
                html_dict['table']['types']['Categorical'] -= 1
                html_dict['table']['types']['Textual'] += 1

    return html_dict


def __profile_tabular_main(my_file_path: str, header: int = 0, sep: str = ',', crs: str = "EPSG:4326",
                           longitude_column: str = None,
                           latitude_column: str = None, wkt_column: str = None, minimal: bool = True):
    if my_file_path.__contains__('.shp'):
        pois = gp.read_file(my_file_path)
        crs = pois.crs
        df = pd.DataFrame(pois)
        df.geometry = df.geometry.astype(str)
    else:
        df = __read_files(my_file_path, header, sep)

    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception as e:
                pass

    if minimal:
        config_file = get_config("config_minimal.yaml")

        with open(config_file) as f:
            data = yaml.safe_load(f)

        config: Settings = Settings().parse_obj(data)
    else:
        config: Settings = Settings()

    config.vars.num.quantiles.append(0.10)
    config.vars.num.quantiles.append(0.90)

    if longitude_column is not None and latitude_column is not None:
        geom_lon_lat = "geometry_" + longitude_column + "_" + latitude_column
        s = gp.GeoSeries.from_xy(df[longitude_column], df[latitude_column], crs=crs)
        s = s.to_crs("EPSG:4326")
        df[geom_lon_lat] = s.to_wkt()

    if wkt_column is not None:
        s = gp.GeoSeries.from_wkt(data=df[wkt_column], crs=crs)
        s = s.to_crs("EPSG:4326")
        df[wkt_column] = s.to_wkt()

    profile = ProfileReport(df, config=config, progress_bar=True)
    html_dict = profile.description_set
    html_dict['table']['profiler_type'] = 'Tabular'
    html_dict['analysis']['filenames'] = [my_file_path]
    html_dict['analysis']['title'] = 'Profiling Report'

    if wkt_column is not None:
        if not html_dict['table']['types'].__contains__('Geometry'):
            html_dict['table']['types']['Categorical'] -= 1
            html_dict['table']['types']['Geometry'] = 1
        else:
            html_dict['table']['types']['Categorical'] -= 1
            html_dict['table']['types']['Geometry'] += 1

        s = gp.GeoSeries.from_wkt(data=df[wkt_column], crs="EPSG:4326")
        html_dict['variables'][wkt_column]['type'] = 'Geometry'
        html_dict['variables'][wkt_column]['mbr'] = box(*s.total_bounds).wkt
        html_dict['variables'][wkt_column]['union_convex_hull'] = s.unary_union.convex_hull.wkt
        html_dict['variables'][wkt_column]['centroid'] = s.unary_union.centroid.wkt
        html_dict['variables'][wkt_column]['length'] = s.unary_union.length
        if len(s) > 1000:
            html_dict['variables'][wkt_column]['heatmap'] = __get_clusters_dict(s[:2000], wkt_column)
        else:
            html_dict['variables'][wkt_column]['heatmap'] = __get_clusters_dict(s, wkt_column)
        missing = s.isna().tolist()
        if any(missing):
            html_dict['variables'][wkt_column]['missing'] = True
            html_dict['variables'][wkt_column]['n_missing'] = sum(missing)
            html_dict['variables'][wkt_column]['p_missing'] = sum(missing) * 100 / len(missing)
        else:
            html_dict['variables'][wkt_column]['missing'] = False
            html_dict['variables'][wkt_column]['n_missing'] = 0
            html_dict['variables'][wkt_column]['p_missing'] = 0.0

        if crs is not None:
            crs_list = CRS.from_string(str(crs))
            html_dict['variables'][wkt_column]['crs'] = 'EPSG:' + str(crs_list.to_epsg())
        else:
            html_dict['variables'][wkt_column]['crs'] = 'EPSG:4326'

        count_geom_types = s.geom_type.value_counts()
        html_dict['variables'][wkt_column]['geom_types'] = count_geom_types

        # calculate area distribution
        s_area = s.area
        stats = s_area.describe(percentiles=[.10, .25, .75, .90])

        html_dict['variables'][wkt_column]['area_distribution'] = {
            'name': wkt_column,
            'count': stats[0],
            'min': stats[3],
            'max': stats[9],
            'average': stats[1],
            'stddev': stats[2],
            'median': stats[6],
            'kurtosis': s_area.kurtosis(),
            'skewness': s_area.skew(),
            'variance': s_area.var(),
            'percentile10': stats[4],
            'percentile25': stats[5],
            'percentile75': stats[7],
            'percentile90': stats[8],
        }

        # calculate length distribution
        s_length = s.length
        stats = s_length.describe(percentiles=[.10, .25, .75, .90])

        html_dict['variables'][wkt_column]['length_distribution'] = {
            'name': wkt_column,
            'count': stats[0],
            'min': stats[3],
            'max': stats[9],
            'average': stats[1],
            'stddev': stats[2],
            'median': stats[6],
            'kurtosis': s_length.kurtosis(),
            'skewness': s_length.skew(),
            'variance': s_length.var(),
            'percentile10': stats[4],
            'percentile25': stats[5],
            'percentile75': stats[7],
            'percentile90': stats[8],
        }

    if longitude_column is not None and latitude_column is not None:
        if not html_dict['table']['types'].__contains__('Geometry'):
            html_dict['table']['types']['Categorical'] -= 1
            html_dict['table']['types']['Geometry'] = 1
        else:
            html_dict['table']['types']['Categorical'] -= 1
            html_dict['table']['types']['Geometry'] += 1
        geom_lon_lat = "geometry_" + longitude_column + "_" + latitude_column
        html_dict['variables'][geom_lon_lat]['type'] = 'Geometry'
        s = gp.GeoSeries.from_wkt(df[geom_lon_lat], crs="EPSG:4326")
        html_dict['variables'][geom_lon_lat]['mbr'] = box(*s.total_bounds).wkt
        html_dict['variables'][geom_lon_lat]['union_convex_hull'] = s.unary_union.convex_hull.wkt
        html_dict['variables'][geom_lon_lat]['centroid'] = s.unary_union.centroid.wkt
        html_dict['variables'][geom_lon_lat]['length'] = s.unary_union.length
        if len(s) > 2000:
            html_dict['variables'][geom_lon_lat]['heatmap'] = __get_clusters_dict(s[:2000], geom_lon_lat)
        else:
            html_dict['variables'][geom_lon_lat]['heatmap'] = __get_clusters_dict(s, geom_lon_lat)
        missing = s.isna().tolist()
        if any(missing):
            html_dict['variables'][geom_lon_lat]['missing'] = True
            html_dict['variables'][geom_lon_lat]['n_missing'] = sum(missing)
            html_dict['variables'][geom_lon_lat]['p_missing'] = sum(missing) * 100 / len(missing)
        else:
            html_dict['variables'][geom_lon_lat]['missing'] = False
            html_dict['variables'][geom_lon_lat]['n_missing'] = 0
            html_dict['variables'][geom_lon_lat]['p_missing'] = 0.0

        if crs is not None:
            crs_list = CRS.from_string(str(crs))
            html_dict['variables'][geom_lon_lat]['crs'] = 'EPSG:' + str(crs_list.to_epsg())
        else:
            html_dict['variables'][geom_lon_lat]['crs'] = 'EPSG:4326'

        count_geom_types = s.geom_type.value_counts()
        html_dict['variables'][geom_lon_lat]['geom_types'] = count_geom_types

        # calculate area distribution
        s_area = s.area
        stats = s_area.describe(percentiles=[.10, .25, .75, .90])

        html_dict['variables'][geom_lon_lat]['area_distribution'] = {
            'name': geom_lon_lat,
            'count': stats[0],
            'min': stats[3],
            'max': stats[9],
            'average': stats[1],
            'stddev': stats[2],
            'median': stats[6],
            'kurtosis': s_area.kurtosis(),
            'skewness': s_area.skew(),
            'variance': s_area.var(),
            'percentile10': stats[4],
            'percentile25': stats[5],
            'percentile75': stats[7],
            'percentile90': stats[8],
        }

        # calculate length distribution
        s_length = s.length

        stats = s_length.describe(percentiles=[.10, .25, .75, .90])

        html_dict['variables'][geom_lon_lat]['length_distribution'] = {
            'name': geom_lon_lat,
            'count': stats[0],
            'min': stats[3],
            'max': stats[9],
            'average': stats[1],
            'stddev': stats[2],
            'median': stats[6],
            'kurtosis': s_length.kurtosis(),
            'skewness': s_length.skew(),
            'variance': s_length.var(),
            'percentile10': stats[4],
            'percentile25': stats[5],
            'percentile75': stats[7],
            'percentile90': stats[8],
        }

    # Files size
    html_dict['table']['byte_size'] = os.path.getsize(my_file_path)

    texts_column_names = []
    for var_name, info in html_dict['variables'].items():
        if info['type'] == 'Categorical' and info['p_unique'] > 0.4:
            texts_column_names.append(var_name)

    if len(texts_column_names) != 0:
        df = df[texts_column_names]
        profile_dict = __create_profile_dict(html_dict, df)

        html_dict = __extend_textual_html(profile_dict, html_dict)
    else:
        profile_dict = __create_profile_dict(html_dict)

    return profile_dict, config, html_dict


# TODO: EPS_DISTANCE MUST BE DATA DRIVEN
def __get_clusters_dict(geo_data: gp.GeoSeries, geometry_column: str = None):
    EPS_DISTANCE = 0.018
    MIN_SAMPLE_POLYGONS = 5
    wkt = gp.GeoDataFrame(geo_data)
    wkt.columns = [geometry_column, *wkt.columns[1:]]

    # preparation for dbscan
    wkt['x'] = wkt[geometry_column].centroid.x
    wkt['y'] = wkt[geometry_column].centroid.y
    coords = wkt[['x', 'y']].values

    # dbscan
    dbscan = DBSCAN(eps=EPS_DISTANCE, min_samples=MIN_SAMPLE_POLYGONS)
    clusters = dbscan.fit(coords)
    # add labels back to dataframe
    labels = pd.Series(clusters.labels_).rename('Clusters')
    wkt = pd.concat([wkt, labels], axis=1)
    data = wkt[['y', 'x', 'Clusters']]
    dict1 = data.to_dict()

    return dict1


def __replace_missing_inf_values(feature_array):
    """
    This method is used to replace the NaN , infinity and -infinity values of an array of numbers.
    The NaN is replaced by the mean of the numbers in the array, the infinity with mean + 3*std (standard deviation)
    and the -infinity with mean - 3*std.

    :param feature_array: An array that contains the values of a feature.
    :type feature_array: numpy.array
    :return:
        -feature_array (numpy.array) - A numpy array with no NaN, infinity and -infinity values.

    """
    feature_array_finite = feature_array[np.isfinite(feature_array)]
    mean_feature_array = np.nanmean(feature_array_finite)
    std_feature_array = np.nanstd(feature_array_finite)
    replace_pos_inf = mean_feature_array + 3 * std_feature_array
    replace_neg_inf = mean_feature_array - 3 * std_feature_array

    feature_array = np.nan_to_num(feature_array, copy=False, nan=mean_feature_array,
                                  posinf=replace_pos_inf, neginf=replace_neg_inf)
    return feature_array


def __is_not_finite(arr):
    """
    This method returns an array of booleans that have 'True' in the positions where we do not have finite numbers.

    :param arr: An array of numbers.
    :type arr: numpy.array
    :return:
         -res (numpy.array) - A numpy array where 'True' if we have non-finite (NaN, infinity and -infinity) values.
    """
    res = np.isfinite(arr)
    np.bitwise_not(res, out=res)  # in-place
    return res


def __read_json_file_tsfresh(json_path: str):
    """
    Read the json file from the given path that contains the features to be calculated by tsfresh package.

    :param json_path: The path containing the json file.
    :type json_path: string
    :return:
        -json_decoded (dict) - A dictionary containing the tsfresh features.
    """
    with open(json_path, "r") as jf:
        json_decoded = json.load(jf)
    return json_decoded


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
