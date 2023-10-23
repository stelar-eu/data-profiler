import os
import sys

sys.path.append('.')
sys.path.append('..')
import calendar  # Core Python Module
from datetime import datetime  # Core Python Module
import plotly.graph_objects as go  # pip install plotly
import streamlit as st  # pip install streamlit
import streamlit.components.v1 as com
from streamlit_option_menu import option_menu  # pip install streamlit-option-menu
from PIL import Image
import urllib.request
from pathlib import Path
import requests
from stelardataprofiler import (
    run_profile,
    write_to_json,
    read_config
)
import json
from minio import Minio
import pandas as pd
import geopandas as gpd
import altair as alt
import numpy as np
from itertools import islice
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium, folium_static
from datetime import datetime
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from matplotlib.collections import PolyCollection
from statsmodels.tsa.stattools import acf
import rasterio as rio
from typing import Union, Any
import plotly.express as px
from collections import Counter
from folium.plugins import Fullscreen


def __float_to_datetime(fl):
    return datetime.fromtimestamp(fl)


def _get_ts_lag(series: pd.Series) -> int:
    lag = 100
    max_lag_size = (len(series) // 2) - 1
    return np.min([lag, max_lag_size])


def __download_file_url(input_file_path: str, url_input: str):
    # Downloads file from the url and save it as filename
    # check if file already exists
    if not os.path.isfile(input_file_path):
        form.write('Downloading File')
    else:
        form.write('File exists and was overwritten!')
    response = requests.get(url_input)
    # Check if the response is ok (200)
    if response.status_code == 200:
        # Open file and write the content
        with open(input_file_path, 'wb') as file:
            # A chunk of 128 bytes
            for chunk in response:
                file.write(chunk)


def __encode_it(o: Any) -> Any:
    if isinstance(o, dict):
        return {__encode_it(k): __encode_it(v) for k, v in o.items()}
    else:
        if isinstance(o, (bool, int, float, str)):
            return o
        elif isinstance(o, list):
            return [__encode_it(v) for v in o]
        elif isinstance(o, set):
            return {__encode_it(v) for v in o}
        elif isinstance(o, (pd.DataFrame, pd.Series)):
            return __encode_it(o.to_dict('records'))
        elif isinstance(o, np.ndarray):
            return __encode_it(o.tolist())
        elif isinstance(o, np.generic):
            return o.item()
        else:
            return str(o)


def __take(n, iterable):
    """Return the first n items of the iterable as a list."""
    return list(islice(iterable, n))


def __reduceCategoricalDict(dictionary: dict, n: int):
    if len(dictionary) > n:
        first_n_pairs = {k: dictionary[k] for k in list(dictionary)[:n]}
        sum_other_pairs = sum(list(dictionary.values())[n:])
        first_n_pairs['Other Values (' + str(len(dictionary) - len(first_n_pairs)) + ')'] = sum_other_pairs
    else:
        first_n_pairs = dictionary
    return first_n_pairs


def __is_integer_num(n):
    if isinstance(n, int):
        return True
    if isinstance(n, float):
        return n.is_integer()
    return False


def __change_font(text: str):
    new_txt = '${\sf \Large '
    split_text = text.split(' ')
    for txt in split_text:
        new_txt += txt + ' \\ '

    new_txt += '}$'

    return new_txt


def __print_metric(value, percentage: bool = False, memory: bool = False, record: bool = False):  # , label: str
    if isinstance(value, str):
        return f"{value}"
        # st.metric(label, f"{value}")
    else:
        if __is_integer_num(value) and not percentage and not memory:
            return f"{int(value)}"
            # st.metric(label, f"{int(value)}")
        else:
            if str(value)[::-1].find('.') > 3:
                value = round(value, 3)
            if percentage or memory or record:
                if percentage:
                    return f"{value * 100:.1f}%"
                    # st.metric(label, f"{value * 100:.1f}%")
                elif record:
                    return f"{value:.1f} B"
                else:
                    return f"{value / 1024:.1f} KiB"
                    # st.metric(label, f"{value / 1024:.1f} KiB")
            else:
                return f"{value}"
                # st.metric(label, f"{value}")


def __calc_hierarchical(records: list) -> int:
    types_count_dict = {}
    uniqueness_names_dict = {}
    uniqueness_counts_dict = {}
    nested_levels_dict = {}
    types_names_dict = {}

    for variable in records:
        v_type = variable['type']
        v_name = variable['name']
        v_uniqueness = variable['uniqueness']
        v_nesting_level = variable['nesting_level']

        if v_nesting_level in nested_levels_dict.keys():
            nested_levels_dict[v_nesting_level] += 1
        else:
            nested_levels_dict[v_nesting_level] = 1

        if v_uniqueness in uniqueness_names_dict.keys():
            uniqueness_names_dict[v_uniqueness] += ", " + v_name
        else:
            uniqueness_names_dict[v_uniqueness] = v_name

        if v_uniqueness in uniqueness_counts_dict.keys():
            uniqueness_counts_dict[v_uniqueness] += 1
        else:
            uniqueness_counts_dict[v_uniqueness] = 1

        if v_type == 'int':
            new_vtype = 'Integer'
        elif v_type == 'string':
            new_vtype = 'Categorical'
        elif v_type == 'float':
            new_vtype = 'Float'
        elif v_type == 'None':
            new_vtype = 'None/Null'
        elif v_type == 'text':
            new_vtype = 'Textual'
        else:
            new_vtype = v_type

        if new_vtype in types_count_dict.keys():
            types_count_dict[new_vtype] += 1
        else:
            types_count_dict[new_vtype] = 1

        if new_vtype in types_names_dict.keys():
            types_names_dict[new_vtype] += ", " + v_name
        else:
            types_names_dict[new_vtype] = v_name

    return uniqueness_counts_dict, types_count_dict, types_names_dict, uniqueness_names_dict, nested_levels_dict


def __fix_length(val):
    new_val = val.split()
    if len(new_val) > 5:
        new_val = np.array_split(new_val, int(len(new_val) / 6))
        new_val = ['<br>'.join(' '.join(x for x in w) for w in new_val)]

    return new_val


# -------------- SETTINGS --------------
incomes = ["Salary", "Blog", "Other Income"]
profiler_types = ["", "Tabular", "Timeseries", "Raster", "Textual", "Hierarchical", "RDFGraph"]
currency = "USD"
page_title = "Stelar Data Profiler"
title_icon = ":mag:"
dirname = os.path.dirname(__file__)
stelar_icon = os.path.join(dirname, 'icons/stelar_icon.jpg')
page_icon = stelar_icon
layout = "wide"
# --------------------------------------

# -------------- INITIALIZE --------------
config_json = {
    "input": {
        "path": "datasets",
        "file": "tabular_vector_example.csv",
        "ras_path": "datasets/vista",
        "ras_file": "32UQV_2002.RAS",
        "rhd_path": "datasets/vista",
        "rhd_file": "32UQV_2002.RHD",
        "format": ".csv",
        "_COMMENT_format_options_": [".csv", ".txt", ".tif", ".rdf", "..."],
        "separator": "|",
        "header": 0,
        "serialization": "turtle",
        "_COMMENT_serialization_options_": ["application/rdf+xml (DEFAULT)", "nquads", "nquads", "turtle", "ntriples",
                                            "trig", "trix", "jsonld", "hdt", "..."],
        "columns": {
            "longitude": "lon",
            "latitude": "lat",
            "wkt": "wkt",
            "time": "date"
        }
    },
    "output": {
        "path": "output",
        "json": "json/tabular_vector_profile.json",
        "html": "html/tabular_vector_profile.html",
        "rdf": "results.rdf",
        "serialization": "nquads",
        "_COMMENT_serialization_options_": ["nquads (DEFAULT)", "turtle", "ntriples", "trig", "trix", "jsonld", "hdt"]
    },
    "profile": {
        "type": "tabular",
        "_COMMENT_profile_type_options_": ["timeseries", "tabular", "raster", "textual", "hierarchical", "rdfGraph",
                                           "vista"]
    }
}

# ----------------------------------------

im = Image.open(page_icon)

st.set_page_config(page_title=page_title, page_icon=im, layout=layout)
st.title(page_title + ' ' + title_icon)

# Include absolute path to directory
if len(sys.argv) == 1:
    dir_path = '.'
else:
    dir_path = sys.argv[1]

# --- HIDE STREAMLIT STYLE ---
# ul.streamlit - expander
# {
#     overflow: scroll;
# }
hide_st_style = """
            <style>
            #the-title {text-align: center}
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            [data-testid="stMetricValue"] {
                font-size: 120%;
            }

            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

if 'show_sidebar' not in st.session_state:
    st.session_state.show_sidebar = False

if 'download_profiling_results' not in st.session_state:
    st.session_state.download_profiling_results = False

if 'profile_dict' not in st.session_state:
    st.session_state.profile_dict = dict()

if 'output_json_name' not in st.session_state:
    st.session_state.output_json_name = ''

if 'button' not in st.session_state:
    st.session_state.button = False

if 'config_placeholder' not in st.session_state:
    st.session_state.config_placeholder = ''

if 'profile_placeholder' not in st.session_state:
    st.session_state.profile_placeholder = ''

if not st.session_state.show_sidebar:
    hide_st_sidebar = """
                    <style>
                        section[data-testid='stSidebar'] {
                            background-color: #ADD8E6;
                            min-width:unset !important;
                            width: unset !important;
                            flex-shrink: unset !important;
                        }

                        section[data-testid='stSidebar'] > div {
                            height: 100%;
                            width: 0px;

                        }
                    </style>

                    """

    st.markdown(hide_st_sidebar, unsafe_allow_html=True)

# sidebar
nav_profiler_type = st.sidebar.selectbox("Select Profiler Type", profiler_types)

with st.sidebar:
    sidebar_selected = option_menu(
        menu_title=None,
        options=["local", "url", "minio"],
        icons=["archive-fill", "wifi", "database-fill"],  # https://icons.getbootstrap.com/
        orientation="horizontal",
    )

    st.session_state.sidebar_select = sidebar_selected

# TODO: Try catch if connection fails or bucket doesn't exist then cancel and throw message
if st.session_state.sidebar_select == 'minio' and nav_profiler_type != '':
    minio_form = st.sidebar.form('minio_form')

    minio_server_input = minio_form.text_input(
        "Enter endpoint",
        placeholder='play.min.io'
    )

    minio_access_key_input = minio_form.text_input(
        "Enter access Key",
        placeholder='Q3AM3UQ867SPQQA43P2F'
    )

    minio_secret_key_input = minio_form.text_input(
        "Enter secret Key",
        placeholder='zuf+tfteSlswRu7BJ86wekitnifILbZam1KYY3TG'
    )

    minio_bucket_input = minio_form.text_input(
        "Enter bucket name",
        placeholder='asiatrip'
    )

    minio_submit = minio_form.form_submit_button('Connect to bucket')

    if minio_submit:
        client = Minio(
            minio_server_input,
            access_key=minio_access_key_input,
            secret_key=minio_secret_key_input,
        )

        if client.bucket_exists(minio_bucket_input):
            minio_form.write('The ' + minio_bucket_input + ' bucket exists')
        else:
            minio_form.write('The ' + minio_bucket_input + ' bucket does not exist')

if 'add_configuration_file' not in st.session_state:
    st.session_state.add_configuration_file = False

if 'form' not in st.session_state:
    st.session_state.form = ''

if nav_profiler_type == 'Tabular':
    config_json = {
        "input": {
            "path": "./datasets",
            "file": "tabular_vector_example.csv",
            "format": ".csv",
            "separator": "|",
            "header": 0,
            "columns": {
                "longitude": "lon",
                "latitude": "lat",
                "wkt": "wkt"
            }
        },
        "output": {
            "path": "./output",
            "json": "tabular_vector_profile.json",
            "rdf": "results.rdf",
            "serialization": "nquads",
            "_COMMENT_serialization_options_": ["nquads (DEFAULT)", "turtle", "ntriples", "trig", "trix", "jsonld",
                                                "hdt"]
        },
        "profile": {
            "type": "tabular",
        }
    }
    st.session_state.form = st.sidebar.form("tabular_form")
elif nav_profiler_type == 'Timeseries':
    config_json = {
        "input": {
            "path": "./datasets/timeseries",
            "file": "one_timeseries_example.csv",
            "format": ".csv",
            "separator": ",",
            "header": 0,
            "columns": {
                "time": "date"
            }
        },
        "output": {
            "path": "./output",
            "json": "one_timeseries_profile.json",
            "rdf": "results.rdf",
            "serialization": "nquads",
            "_COMMENT_serialization_options_": ["nquads (DEFAULT)", "turtle", "ntriples", "trig", "trix", "jsonld",
                                                "hdt"]
        },
        "profile": {
            "type": "timeseries",
        }
    }
    st.session_state.form = st.sidebar.form("timeseries_form")
elif nav_profiler_type == 'Raster':
    config_json = {
        "input": {
            "path": "./datasets/images",
            "file": "image_example.png",
            "format": ".png",
            "ras_file": "32UQV_2002.RAS",
            "rhd_file": "32UQV_2002.RHD"
        },
        "output": {
            "path": "./output",
            "json": "raster_profile.json",
            "rdf": "results.rdf",
            "serialization": "nquads",
            "_COMMENT_serialization_options_": ["nquads (DEFAULT)", "turtle", "ntriples", "trig", "trix", "jsonld",
                                                "hdt"]
        },
        "profile": {
            "type": "raster",
        }
    }
    st.session_state.form = st.sidebar.form("raster_form")
elif nav_profiler_type == 'Textual':
    config_json = {
        "input": {
            "path": "./datasets/text",
            "file": "text_example.txt",
            "format": ".txt"
        },
        "output": {
            "path": "./output",
            "json": "textual_profile.json",
            "rdf": "results.rdf",
            "serialization": "nquads",
            "_COMMENT_serialization_options_": ["nquads (DEFAULT)", "turtle", "ntriples", "trig", "trix", "jsonld",
                                                "hdt"]
        },
        "profile": {
            "type": "textual",
        }
    }
    st.session_state.form = st.sidebar.form("textual_form")
elif nav_profiler_type == 'Hierarchical':
    config_json = {
        "input": {
            "path": "./datasets",
            "file": "json_example.json",
            "format": ".json"
        },
        "output": {
            "path": "./output",
            "json": "hierarchical_profile.json",
            "rdf": "results.rdf",
            "serialization": "nquads",
            "_COMMENT_serialization_options_": ["nquads (DEFAULT)", "turtle", "ntriples", "trig", "trix", "jsonld",
                                                "hdt"]
        },
        "profile": {
            "type": "hierarchical",
        }
    }
    st.session_state.form = st.sidebar.form("hierarchical_form")
elif nav_profiler_type == 'RDFGraph':
    config_json = {
        "input": {
            "path": "./datasets",
            "file": "rdf_example.ttl",
            "format": ".ttl",
            "serialization": "turtle"
        },
        "output": {
            "path": "./output",
            "json": "rdfGraph_profile.json",
            "rdf": "results.rdf",
            "serialization": "nquads",
            "_COMMENT_serialization_options_": ["nquads (DEFAULT)", "turtle", "ntriples", "trig", "trix", "jsonld",
                                                "hdt"]
        },
        "profile": {
            "type": "rdfGraph",
        }
    }
    st.session_state.form = st.sidebar.form("rdfGraph_form")
else:
    st.session_state.form = ''

if st.session_state.form != '':
    form = st.session_state.form

    if st.session_state.sidebar_select == 'url':
        url_input = form.text_input(
            "Enter url 👇",
            help='If the url is valid the file will be downloaded '
                 'to the given local path under the given filename'
        )

    if st.session_state.sidebar_select == 'minio':
        minio_filename_input = form.text_input(
            "Input file name in minio",
            help='If the file with this name exists in the given bucket then '
                 'it is downloaded in the below directory path with the below filename'
        )

    dir_input = form.text_input(
        "Input directory path",
        placeholder=config_json['input']['path']
    )

    filename_input = form.text_input(
        "Input filename",
        placeholder=config_json['input']['file'],
        help='The filename must include the extension of the file e.g. filename.txt\n\r'
             'If it is empty then search the above directory and take only the files '
             'of the below given format'
    )

    format_input = form.text_input(
        "Input files format",
        placeholder=config_json['input']['format'],
        help='Only needed if you have multiple files'
    )

    if nav_profiler_type == 'Tabular' or nav_profiler_type == 'Timeseries':
        separator_input = form.text_input(
            "Input separator",
            placeholder=config_json['input']['separator'],
            help='The seperator of the file only needed for .csv'
        )

        header_input = form.number_input(
            "Input header",
            value=config_json['input']['header'],
            step=1,
            min_value=-1,
            help='If -1 then no header'
        )

        if nav_profiler_type == 'Tabular':
            lon, lat = form.columns(2)
            lon_input = lon.text_input(
                "(Optional) Input name of longitude column",
                placeholder=config_json['input']['columns']['longitude'],
                help='We also need the latitude column'
            )

            lat_input = lat.text_input(
                "(Optional) Input name of latitude column",
                placeholder=config_json['input']['columns']['latitude'],
                help='We also need the longitude column'
            )

            wkt_input = form.text_input(
                "(Optional) Input name of wkt column",
                placeholder=config_json['input']['columns']['wkt']
            )
        elif nav_profiler_type == 'Timeseries':
            time_input = form.text_input(
                "Input name of time column",
                placeholder=config_json['input']['columns']['time'],
                help='We need the name of the time column to successfully run timeseries profile'
            )

    if nav_profiler_type == 'RDFGraph':
        serialization_input = form.text_input(
            "Input the format that represents the RDF data",
            placeholder=config_json['input']['serialization'],
            help='The RDF formats you can serialize data to with rdflib are listed in https://rdflib.readthedocs.io/en/stable/plugin_serializers.html '
        )

    json_output = form.text_input(
        "Output the .JSON filename",
        placeholder=config_json['output']['json'],
        help='Include the .json extension in the filename'
    )

    submitted = form.form_submit_button("Create configuration file")

    if submitted:
        vista = False
        config_json['input']['path'] = dir_input
        config_json['input']['file'] = filename_input
        config_json['input']['format'] = format_input
        if nav_profiler_type == 'Raster':
            file = Path(filename_input)
            if file.suffix == '.RAS':
                config_json['input']['ras_file'] = filename_input
                config_json['input']['rhd_file'] = file.stem + '.RHD'
                config_json['profile']['type'] = 'vista'
                vista = True
                del (config_json['input']['file'])
                del (config_json['input']['format'])
            else:
                config_json['input']['path'] = dir_input
                config_json['input']['file'] = filename_input
                config_json['input']['format'] = format_input
                vista = False
                del (config_json['input']['ras_file'])
                del (config_json['input']['rhd_file'])

        if st.session_state.sidebar_select == 'url':
            if not vista:
                input_file_path = str(os.path.abspath(os.path.join(os.path.abspath(dir_input), filename_input)))
                path = Path(str(Path(input_file_path).parent))
                path.mkdir(parents=True, exist_ok=True)

                # Download file from the url
                __download_file_url(input_file_path, url_input)
            else:
                # Download .RAS file
                input_file_path = str(os.path.abspath(os.path.join(os.path.abspath(dir_input), filename_input)))
                path = Path(str(Path(input_file_path).parent))
                path.mkdir(parents=True, exist_ok=True)

                # Download .RAS file from the url
                __download_file_url(input_file_path, url_input)

                # Download .RHD file

                rhd_file_path = str(os.path.abspath(os.path.join(Path(input_file_path).parent,
                                                                 Path(input_file_path).stem + '.RHD')))
                rhd_url_input = str(Path(os.path.join(Path(url_input).parent,
                                                      Path(url_input).stem + '.RHD')))

                # Download .RHD file from the url
                __download_file_url(rhd_file_path, rhd_url_input)

        # TODO: Try catch if file doesn't exist then cancel and throw message
        if st.session_state.sidebar_select == 'minio':
            client = Minio(
                minio_server_input,
                access_key=minio_access_key_input,
                secret_key=minio_secret_key_input,
            )

            if not vista:
                input_file_path = str(os.path.abspath(os.path.join(os.path.abspath(dir_input), filename_input)))
                path = Path(str(Path(input_file_path).parent))
                path.mkdir(parents=True, exist_ok=True)

                client.fget_object(minio_bucket_input, minio_filename_input, input_file_path)
            else:
                # Download .RAS file
                input_file_path = str(os.path.abspath(os.path.join(os.path.abspath(dir_input), filename_input)))
                path = Path(str(Path(input_file_path).parent))
                path.mkdir(parents=True, exist_ok=True)

                client.fget_object(minio_bucket_input, minio_filename_input, input_file_path)

                # Download .RAS file
                minio_rhd_filename_input = str(Path(minio_filename_input).stem + '.RHD')
                rhd_file_path = str(os.path.abspath(os.path.join(Path(input_file_path).parent,
                                                                 Path(input_file_path).stem + '.RHD')))

                client.fget_object(minio_bucket_input, minio_rhd_filename_input, rhd_file_path)

        if nav_profiler_type == 'Tabular' or nav_profiler_type == 'Timeseries':
            config_json['input']['separator'] = separator_input

            if header_input == -1:
                # noinspection PyTypedDict
                config_json['input']['header'] = None
            else:
                config_json['input']['header'] = header_input
            if nav_profiler_type == 'Tabular':
                if lon_input != '' and lat_input != '':
                    config_json['input']['columns']['longitude'] = lon_input
                    config_json['input']['columns']['latitude'] = lat_input
                else:
                    del (config_json['input']['columns']['longitude'])
                    del (config_json['input']['columns']['latitude'])
                if wkt_input != '':
                    config_json['input']['columns']['wkt'] = wkt_input
                else:
                    del (config_json['input']['columns']['wkt'])
            elif nav_profiler_type == 'Timeseries':
                if time_input != '':
                    config_json['input']['columns']['time'] = time_input

        if nav_profiler_type == 'RDFGraph':
            if serialization_input != '':
                config_json['input']['serialization'] = serialization_input

        config_json['output']['path'] = dir_path + '/profiling_results'
        config_json['output']['json'] = json_output

        if not vista:
            name = Path(config_json['input']['file']).stem
        else:
            name = Path(config_json['input']['ras_file']).stem

        if name == '':
            name = 'multi_' + nav_profiler_type

        write_to_json(config_json, dir_path + '/config_files/' + name + '_config.json')
        st.warning('The configuration file was saved at ' + str(
            'config_files/' + name + '_config.json'))
        st.session_state.config_placeholder = str(
            'config_files/' + name + '_config.json')
        json_string = json.dumps(__encode_it(config_json), indent=3)

        st.download_button(
            label="Download JSON Config",
            data=json_string,
            file_name=config_json['profile']['type'] + '_config.json',
            mime='application/json',
        )
        st.session_state.add_configuration_file = True
    else:
        st.session_state.add_configuration_file = False

if st.session_state.add_configuration_file:
    st.markdown(
        """
        <style>
        section[data-testid='stSidebar'] {
            background-color: #ADD8E6;
            min-width:unset !important;
            width: unset !important;
            flex-shrink: unset !important;
        }
        
        section[data-testid='stSidebar'] > div {
            height: 100%;
            width: 0px;

        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.session_state.show_sidebar = False
    create_configuration_file = False

selected = option_menu(
    menu_title=None,
    options=["Run Profiler", "Data Visualization", "Compare Profiles"],
    icons=["pencil-fill", "bar-chart-fill", "gear-fill"],  # https://icons.getbootstrap.com/
    orientation="horizontal",
)

if selected == "Run Profiler":
    create_configuration_file = st.button('Create new configuration file')

    if create_configuration_file:
        st.markdown("""
                    <style>
                    section[data-testid='stSidebar'] > div {
                        height: 100%;
                        width: 400px;
                        position: relative;
                        z-index: 1;
                        top: 0;
                        left: -5;
                        background-color: #ADD8E6;
                        overflow-x: hidden;
                        transition: 0.5s ease;
                        padding-top: -10px;
                        white-space: nowrap;
                    <style>
                        """, unsafe_allow_html=True)

        st.session_state.show_sidebar = True

    with st.form("my_form"):
        path_to_config, local = st.tabs(['Write the config JSON file path', 'Choose local config JSON file'])
        with local:
            configuration_file = st.file_uploader("Choose a configuration file",
                                                  help='Please see config_template.json in https://github.com/stelar-eu/data-profiler/blob/main/config_template.json')

        with path_to_config:
            path_to_config_input = st.text_input(
                "Input the path to the config json which was "
                "displayed after the creation of the config",
                value=st.session_state.config_placeholder,
                help='The path is pasted automatically after the creation of the config file.\n\n '
                     'It searches for the config JSON file in the local/server '
                     'storage that has been given when the app was executed.'
            )

        submitted = st.form_submit_button("Execute Profiling")
        if submitted and (configuration_file is not None or path_to_config_input != ''):
            if configuration_file is not None:
                config_dict = read_config(configuration_file.read())
            else:
                absolute_path = os.path.abspath(os.path.join(dir_path, path_to_config_input))
                config_dict = read_config(absolute_path)

            with st.spinner('Running Profiler....'):
                run_profile(config_dict)

            output_dir_path = config_dict['output']['path']
            output_dir_path = os.path.abspath(output_dir_path)
            output_json_name = config_dict['output']['json']
            output_json_path = os.path.abspath(os.path.join(output_dir_path, output_json_name))

            st.session_state.profile_placeholder = 'profiling_results/' + output_json_name

            st.write(
                'Successfully executed Profiler and the resulting JSON file can be found in profiling_results/' + output_json_name)

            st.session_state.profile_dict = read_config(str(output_json_path))
            st.session_state.output_json_name = output_json_name
            st.session_state.download_profiling_results = True
            st.session_state.button = False

        elif submitted and configuration_file is None and path_to_config_input == '':
            st.write("No configuration file was uploaded!")

    if st.session_state.button:
        st.session_state.download_profiling_results = False

    if st.session_state.download_profiling_results:
        profile_dict = st.session_state.profile_dict
        profile_json_string = json.dumps(__encode_it(profile_dict), indent=3)

        st.session_state.button = st.download_button(
            label="Download JSON Profile",
            data=profile_json_string,
            file_name=st.session_state.output_json_name,
            mime='application/json'
        )

        st.session_state.button = True

if selected == "Data Visualization":
    with st.form("my_form"):
        path_to_config, local = st.tabs(['Write the profile JSON file path', 'Choose local profile JSON file'])
        with local:
            profile_results_file = st.file_uploader("Choose a JSON file",
                                                    help='The chosen JSON file must have been produced by the Profiler')

        with path_to_config:
            path_to_profile_json_input = st.text_input(
                "Input the path to the json which was displayed "
                "after the execution of the profiler",
                value=st.session_state.profile_placeholder,
                help='The path is pasted automatically after the execution of the profiler.\n\n '
                     'It searches for the profiler JSON file in the local/server '
                     'storage that has been given when the app was executed.'
            )

        submitted = st.form_submit_button("Visualize Results")
        if submitted and (profile_results_file is not None or path_to_profile_json_input != ''):
            if profile_results_file is not None:
                config_dict = read_config(profile_results_file.read())
            else:
                absolute_path = os.path.abspath(os.path.join(dir_path, path_to_profile_json_input))
                config_dict = read_config(absolute_path)

            if config_dict['table']['profiler_type'] in ['Tabular', 'TimeSeries']:
                overview, variables = st.tabs(["Overview", "Variables"])

                with overview:
                    table_stats = config_dict['table']
                    if config_dict['table']['profiler_type'] == 'TimeSeries':
                        dataset_stats, gap_stats = st.columns([0.4, 0.6])
                        with dataset_stats:
                            st.subheader("Dataset Statistics")
                            st.metric(__change_font("Number of timeseries"),
                                      __print_metric(table_stats['num_attributes'] - 1))
                            st.metric(__change_font("Timestamps"),
                                      __print_metric(table_stats['num_rows']))
                            st.metric(__change_font("Missing values"),
                                      __print_metric(table_stats['n_cells_missing']))
                            st.metric(__change_font("Missing values (\%)"),
                                      __print_metric(table_stats['p_cells_missing'], percentage=True))
                            # st.metric("Profiler Type", __print_metric(table_stats['profiler_type']))
                            st.metric(__change_font("Total size in memory"),
                                      __print_metric(table_stats['memory_size'], memory=True))
                            st.metric(__change_font("Average record size in memory"),
                                      __print_metric(table_stats['record_size'], record=True))

                        if len(table_stats['ts_gaps_frequency_distribution']) != 0:
                            with gap_stats:
                                st.subheader('Total Gap Length')
                                cols = st.columns(2)
                                # cols[0].subheader('Statistics')
                                gaps_dict = {'Minimum': __print_metric(table_stats['ts_min_gap']),
                                             'Mean': __print_metric(table_stats['ts_avg_gap']),
                                             'Maximum': __print_metric(table_stats['ts_max_gap'])
                                             }

                                gaps_df = pd.DataFrame(gaps_dict.items(), columns=['Statistics', 'Values'])
                                cols[0].dataframe(gaps_df, use_container_width=True, hide_index=True)

                                # cols[0].metric(__change_font("Minimum"), __print_metric(table_stats['ts_min_gap']))
                                # cols[0].metric(__change_font("Mean"), __print_metric(table_stats['ts_avg_gap']))
                                # cols[0].metric(__change_font("Maximum"), __print_metric(table_stats['ts_max_gap']))

                                # cols[1].subheader('Distribution')

                                gaps = {gap['gap_size']: gap['count']
                                        for gap in table_stats['ts_gaps_frequency_distribution']}
                                gaps = dict(
                                    sorted(gaps.items(), key=lambda item: item[1], reverse=True))

                                gaps_df = pd.DataFrame(gaps.items(), columns=['Gap Length', 'Count'])
                                gaps_df = gaps_df[gaps_df.Count != 0]

                                fig = px.pie(gaps_df, values='Count', names='Gap Length')
                                fig.update_traces(textposition='inside', textfont_size=14)
                                fig.update_layout(title_text='Distribution', title_x=0.5,
                                                  title_y=1, title_xanchor='center',
                                                  title_yanchor='top', legend_font_size=14,
                                                  title_font_size=18
                                                  )
                                cols[1].plotly_chart(fig, use_container_width=True)
                        else:
                            with gap_stats:
                                st.subheader('There are no gaps in the timeseries of the dataset!')
                    else:
                        dataset_stats, var_types = st.columns(2)
                        with dataset_stats:
                            st.subheader("Dataset Statistics")
                            st.metric(__change_font("Number of Columns"), __print_metric(table_stats['num_attributes']))
                            st.metric(__change_font("Number of Rows"), __print_metric(table_stats['num_rows']))
                            st.metric(__change_font("Missing cells"), __print_metric(table_stats['n_cells_missing']))
                            st.metric(__change_font("Missing cells (\%)"),
                                      __print_metric(table_stats['p_cells_missing'], percentage=True))
                            # st.metric("Profiler Type", __print_metric(table_stats['profiler_type']))
                            st.metric(__change_font("Total size in memory"),
                                      __print_metric(table_stats['memory_size'], memory=True))
                            st.metric(__change_font("Average record size in memory"),
                                      __print_metric(table_stats['record_size'], record=True))

                            with var_types:
                                variable_types = table_stats['types']
                                st.subheader("Variable Types")

                                for vtype in variable_types:
                                    st.metric(__change_font(vtype['type']), __print_metric(vtype['count']))

                with variables:
                    variable_stats = config_dict['variables']

                    for variable in variable_stats:
                        with st.expander(variable['name'] + ' ( ' + variable['type'] + ' ) '):
                            if variable['type'] == 'Textual':
                                stats, plot = st.columns(2)
                                with stats:
                                    st.metric(__change_font("Distinct"),
                                              __print_metric(variable['n_distinct']))
                                    st.metric(__change_font("Distinct (\%)"),
                                              __print_metric(variable['p_distinct'], percentage=True))
                                    st.metric(__change_font("Missing values"),
                                              __print_metric(variable['num_missing']))
                                    st.metric(__change_font("Missing values (\%)"),
                                              __print_metric(variable['p_missing'],
                                                             percentage=True))
                                    st.metric(__change_font("Memory size"),
                                              __print_metric(variable['memory_size'],
                                                             memory=True))
                                with plot:
                                    languages = {language['language']: language['percentage']
                                                 for language in variable['language_distribution']}
                                    languages = dict(
                                        sorted(languages.items(), key=lambda item: item[1], reverse=True))

                                    languages = __reduceCategoricalDict(languages, 10)
                                    languages_df = pd.DataFrame(languages.items(), columns=['Language', 'Percentage'])
                                    languages_df = languages_df[languages_df.Percentage != 0]
                                    languages_df['Language'] = languages_df['Language'].apply(__fix_length)

                                    fig = px.pie(languages_df, values='Percentage', names='Language')
                                    fig.update_traces(textposition='inside', textfont_size=14)
                                    fig.update_layout(title_text='Language Distribution', title_x=0.5,
                                                      title_y=1, title_xanchor='center',
                                                      title_yanchor='top', legend_font_size=14,
                                                      title_font_size=18)
                                    st.plotly_chart(fig, use_container_width=True)

                                tab1, tab2 = st.tabs(["Overview", "Distributions"])

                                with tab1:
                                    unique, ratios = st.columns(2)

                                    with unique:
                                        st.subheader("Unique")
                                        st.metric(__change_font("Unique"), __print_metric(variable['n_unique']))
                                        st.metric(__change_font("Unique (\%)"),
                                                  __print_metric(variable['uniqueness'], percentage=True))

                                    with ratios:
                                        st.subheader("Ratios")
                                        st.metric(__change_font("Ratio Uppercase (\%)"),
                                                  __print_metric(variable['ratio_uppercase'], percentage=True))
                                        st.metric(__change_font("Ratio Digits (\%)"),
                                                  __print_metric(variable['ratio_digits'], percentage=True))
                                        st.metric(__change_font("Ratio Special Characters (\%)"),
                                                  __print_metric(variable['ratio_special_characters'], percentage=True))

                                with tab2:
                                    characters, words = st.columns(2)

                                    with characters:
                                        tmp = variable['num_chars_distribution']
                                        st.subheader("Characters Length")
                                        characters_dict = {'Mean': __print_metric(tmp['average']),
                                                           'Standard Deviation': __print_metric(tmp['stddev']),
                                                           'Kurtosis': __print_metric(tmp['kurtosis']),
                                                           'Skewness': __print_metric(tmp['skewness']),
                                                           'Variance': __print_metric(tmp['variance'])}

                                        characters_df = pd.DataFrame(characters_dict.items(),
                                                                     columns=['Statistics', 'Values'])
                                        st.dataframe(characters_df, use_container_width=True, hide_index=True)

                                        # st.metric(__change_font("Mean"), __print_metric(tmp['average']))
                                        # st.metric(__change_font("Standard Deviation"), __print_metric(tmp['stddev']))
                                        # st.metric(__change_font("Kurtosis"), __print_metric(tmp['kurtosis']))
                                        # st.metric(__change_font("Skewness"), __print_metric(tmp['skewness']))
                                        # st.metric(__change_font("Variance"), __print_metric(tmp['variance']))
                                        fig = go.Figure()
                                        fig.add_trace(go.Box(
                                            q1=[tmp['percentile25']],
                                            median=[tmp['median']],
                                            mean=[tmp['average']],
                                            q3=[tmp['percentile75']],
                                            lowerfence=[tmp['min']],
                                            upperfence=[tmp['max']],
                                            boxmean='sd',
                                            boxpoints=False,
                                            sd=[tmp['stddev']],
                                            showlegend=False,
                                            x0=tmp['name']
                                        ))

                                        fig.update_layout(title_text="Values Distribution", title_x=0.5,
                                                          title_y=0.9, title_xanchor='center',
                                                          title_yanchor='top',
                                                          title_font_size=18)

                                        fig.update_xaxes(
                                            title_font_size=16,
                                            tickfont_size=16
                                        )

                                        fig.update_yaxes(
                                            tickfont_size=16
                                        )

                                        st.plotly_chart(fig, use_container_width=True)

                                    with words:
                                        tmp = variable['num_words_distribution']
                                        st.subheader("Words Length")
                                        words_dict = {'Mean': __print_metric(tmp['average']),
                                                      'Standard Deviation': __print_metric(tmp['stddev']),
                                                      'Kurtosis': __print_metric(tmp['kurtosis']),
                                                      'Skewness': __print_metric(tmp['skewness']),
                                                      'Variance': __print_metric(tmp['variance'])}

                                        words_df = pd.DataFrame(words_dict.items(),
                                                                columns=['Statistics', 'Values'])
                                        st.dataframe(words_df, use_container_width=True, hide_index=True)

                                        # st.metric(__change_font("Mean"), __print_metric(tmp['average']))
                                        # st.metric(__change_font("Standard Deviation"), __print_metric(tmp['stddev']))
                                        # st.metric(__change_font("Kurtosis"), __print_metric(tmp['kurtosis']))
                                        # st.metric(__change_font("Skewness"), __print_metric(tmp['skewness']))
                                        # st.metric(__change_font("Variance"), __print_metric(tmp['variance']))
                                        fig = go.Figure()
                                        fig.add_trace(go.Box(
                                            q1=[tmp['percentile25']],
                                            median=[tmp['median']],
                                            mean=[tmp['average']],
                                            q3=[tmp['percentile75']],
                                            lowerfence=[tmp['min']],
                                            upperfence=[tmp['max']],
                                            boxmean='sd',
                                            boxpoints=False,
                                            sd=[tmp['stddev']],
                                            showlegend=False,
                                            x0=tmp['name']
                                        ))

                                        fig.update_layout(title_text="Values Distribution", title_x=0.5,
                                                          title_y=0.9, title_xanchor='center',
                                                          title_yanchor='top',
                                                          title_font_size=18)

                                        fig.update_xaxes(
                                            title_font_size=16,
                                            tickfont_size=16
                                        )

                                        fig.update_yaxes(
                                            tickfont_size=16
                                        )
                                        st.plotly_chart(fig, use_container_width=True)

                            elif variable['type'] == 'Categorical':
                                stats, plot = st.columns(2)
                                with stats:
                                    st.metric(__change_font("Distinct"), __print_metric(variable['n_distinct']))
                                    st.metric(__change_font("Distinct (\%)"),
                                              __print_metric(variable['p_distinct'], percentage=True))
                                    st.metric(__change_font("Missing values"), __print_metric(variable['num_missing']))
                                    st.metric(__change_font("Missing values (\%)"),
                                              __print_metric(variable['p_missing'], percentage=True))
                                    st.metric(__change_font("Memory size"),
                                              __print_metric(variable['memory_size'], memory=True))

                                with plot:
                                    categories = {category['type']: category['count']
                                                  for category in variable['frequency_distribution']}

                                    categories = dict(
                                        sorted(categories.items(), key=lambda item: item[1], reverse=True))
                                    categories = __reduceCategoricalDict(categories, 10)
                                    categories_df = pd.DataFrame(categories.items(), columns=['Category', 'Count'])
                                    categories_df = categories_df[categories_df.Count != 0]

                                    categories_df['Category'] = categories_df['Category'].apply(__fix_length)
                                    fig = px.pie(categories_df, values='Count', names='Category')
                                    fig.update_traces(textposition='inside', textfont_size=14)
                                    fig.update_layout(title_text='Category Distribution', title_x=0.5,
                                                      title_y=1, title_xanchor='center',
                                                      title_yanchor='top', legend_font_size=14,
                                                      title_font_size=18)
                                    st.plotly_chart(fig, use_container_width=True)

                                tab1, tab2 = st.tabs(["Overview", "Categories"])
                                with tab1:
                                    unique, sample = st.columns(2)

                                    with unique:
                                        st.subheader("Unique")
                                        st.metric(__change_font("Unique"), __print_metric(variable['n_unique']))
                                        st.metric(__change_font("Unique (\%)"),
                                                  __print_metric(variable['uniqueness'], percentage=True))
                                    with sample:
                                        st.subheader("Sample")
                                        samples = {sample['row']: sample['cat']
                                                   for sample in variable['samples']}
                                        count = 0
                                        for row, cat in samples.items():
                                            with st.container():
                                                col1, col2 = st.columns(2)
                                                if count == 0:
                                                    col1.write(__change_font("1st row "))
                                                elif count == 1:
                                                    col1.write(__change_font("2nd row"))
                                                elif count == 2:
                                                    col1.write(__change_font("3rd row"))
                                                elif count == 3:
                                                    col1.write(__change_font("4th row"))
                                                elif count == 4:
                                                    col1.write(__change_font("5th row"))

                                                col2.write(__print_metric(cat))
                                            count += 1

                                with tab2:
                                    categories = {category['type']: category['count']
                                                  for category in variable['frequency_distribution']}

                                    categories = dict(
                                        sorted(categories.items(), key=lambda item: item[1], reverse=True))

                                    categories_df = pd.DataFrame(categories.items(), columns=['Category', 'Count'])
                                    if variable['num_missing'] > 0:
                                        categories_df.loc[len(categories_df.index)] = ['Missing',
                                                                                       variable['num_missing']]

                                    categories_df['Frequency'] = (categories_df['Count'] /
                                                                  (variable['count'] + variable['num_missing']))
                                    categories_df['Frequency'] = categories_df['Frequency'].map('{:.5%}'.format)
                                    st.dataframe(categories_df, use_container_width=True, hide_index=True)

                            elif variable['type'] == 'Unsupported':
                                stats, plot = st.columns(2)
                                with stats:
                                    st.metric(__change_font("Status"), f"{'Rejected'}")
                                    st.metric(__change_font("Missing values"), __print_metric(variable['num_missing']))
                                    st.metric(__change_font("Missing values (\%)"),
                                              __print_metric(variable['p_missing'], percentage=True))
                                    st.metric(__change_font("Memory size"),
                                              __print_metric(variable['memory_size'], memory=True))

                            elif variable['type'] == 'Numeric':
                                with st.container():
                                    cols = st.columns(3)
                                    cols[0].metric(__change_font("Distinct"), __print_metric(variable['n_distinct']))
                                    cols[0].metric(__change_font("Distinct (\%)"),
                                                   __print_metric(variable['p_distinct'], percentage=True))
                                    cols[0].metric(__change_font("Missing values"),
                                                   __print_metric(variable['num_missing']))
                                    cols[0].metric(__change_font("Missing values (\%)"),
                                                   __print_metric(variable['p_missing'], percentage=True))
                                    cols[0].metric(__change_font("Infinite"), __print_metric(variable['n_infinite']))
                                    cols[0].metric(__change_font("Infinite (\%)"),
                                                   __print_metric(variable['p_infinite'], percentage=True))
                                    cols[0].metric(__change_font("Mean"), __print_metric(variable['average']))

                                    cols[1].metric(__change_font("Minimum"), __print_metric(variable['min']))
                                    cols[1].metric(__change_font("Maximum"), __print_metric(variable['max']))
                                    cols[1].metric(__change_font("Zeros"), __print_metric(variable['n_zeros']))
                                    cols[1].metric(__change_font("Zeros (\%)"),
                                                   __print_metric(variable['p_zeros'], percentage=True))
                                    cols[1].metric(__change_font("Negative"), __print_metric(variable['n_negative']))
                                    cols[1].metric(__change_font("Negative (\%)"),
                                                   __print_metric(variable['p_negative'], percentage=True))
                                    cols[1].metric(__change_font("Memory size"),
                                                   __print_metric(variable['memory_size'], memory=True))

                                    bar_df = pd.DataFrame({'counts': variable['histogram_counts']})

                                    binned = bar_df.groupby(
                                        pd.cut(bar_df.counts, bins=variable['histogram_bins'])).count()
                                    binned['start_value'] = variable['histogram_bins'][:-1]
                                    binned['end_value'] = variable['histogram_bins'][1:]
                                    binned['counts'] = variable['histogram_counts']
                                    binned = binned.reset_index(drop=True)

                                    bar_freq = alt.Chart(binned).mark_bar().encode(
                                        x=alt.X('start_value', bin='binned',
                                                title='Histogram with fixed size bins (bins=' + str(len(
                                                    variable['histogram_bins']) - 1) + ')'),
                                        x2=alt.X2('end_value', title=''),
                                        y=alt.Y('counts', title='Frequency'),
                                        tooltip=['start_value', 'end_value', 'counts']
                                    ).configure_axis(
                                        labelFontSize=16,
                                        titleFontSize=16
                                    ).interactive()

                                    cols[2].altair_chart(bar_freq, use_container_width=True)

                                tab1, tab2, tab3 = st.tabs(["Statistics", "Common Values", "Extreme Values"])

                                with tab1:
                                    with st.container():
                                        stat_cols = st.columns(2)
                                        stat_cols[0].subheader('Quantile statistics')
                                        quant_stats_dict = {'Mean': __print_metric(variable['average']),
                                                            'Range': __print_metric(variable['range']),
                                                            'Interquartile range (IQR)': __print_metric(
                                                                variable['iqr'])}

                                        quan_stats_df = pd.DataFrame(quant_stats_dict.items(),
                                                                     columns=['Statistics', 'Values'])

                                        stat_cols[0].dataframe(quan_stats_df, use_container_width=True, hide_index=True)

                                        # stat_cols[0].metric(__change_font("Mean"), __print_metric(variable['average']))
                                        # stat_cols[0].metric(__change_font("Range"), __print_metric(variable['range']))
                                        # stat_cols[0].metric(__change_font("Interquartile range (IQR)"),
                                        #                     __print_metric(variable['iqr']))
                                        fig = go.Figure()
                                        fig.add_trace(go.Box(
                                            q1=[variable['percentile25']],
                                            median=[variable['median']],
                                            mean=[variable['average']],
                                            q3=[variable['percentile75']],
                                            lowerfence=[variable['min']],
                                            upperfence=[variable['max']],
                                            boxmean='sd',
                                            boxpoints=False,
                                            sd=[variable['stddev']],
                                            showlegend=False,
                                            x0=variable['name']
                                        ))
                                        fig.update_layout(title_text="Values Distribution", title_x=0.5,
                                                          title_y=0.9, title_xanchor='center',
                                                          title_yanchor='top',
                                                          title_font_size=18)

                                        fig.update_xaxes(
                                            title_font_size=16,
                                            tickfont_size=16
                                        )

                                        fig.update_yaxes(
                                            tickfont_size=16
                                        )

                                        stat_cols[0].plotly_chart(fig, use_container_width=True)

                                        stat_cols[1].subheader('Descriptive statistics')
                                        desc_stats_dict = {'Standard deviation': __print_metric(variable['stddev']),
                                                           'Coefficient of variation (CV)': __print_metric(
                                                               variable['cv']),
                                                           'Kurtosis': __print_metric(variable['kurtosis']),
                                                           'Median Absolute Deviation (MAD)': __print_metric(
                                                               variable['mad']),
                                                           'Skewness': __print_metric(variable['skewness']),
                                                           'Sum': __print_metric(variable['sum']),
                                                           'Variance': __print_metric(variable['variance'])
                                                           }

                                        if variable['monotonic'] == 0:
                                            # stat_cols[1].metric(__change_font("Monotonicity"), f"Not monotonic")
                                            desc_stats_dict['Monotonicity'] = 'Not monotonic'
                                        else:
                                            # stat_cols[1].metric(__change_font("Monotonicity"), f"Monotonic")
                                            desc_stats_dict['Monotonicity'] = 'Monotonic'

                                        quan_stats_df = pd.DataFrame(desc_stats_dict.items(),
                                                                     columns=['Statistics', 'Values'])

                                        stat_cols[1].dataframe(quan_stats_df, use_container_width=True,
                                                               hide_index=True)

                                        # stat_cols[1].metric(__change_font("Standard deviation"),
                                        #                     __print_metric(variable['stddev']))
                                        # stat_cols[1].metric(__change_font("Coefficient of variation (CV)"),
                                        #                     __print_metric(variable['cv']))
                                        # stat_cols[1].metric(__change_font("Kurtosis"),
                                        #                     __print_metric(variable['kurtosis']))
                                        # stat_cols[1].metric(__change_font("Median Absolute Deviation (MAD)"),
                                        #                     __print_metric(variable['mad']))
                                        # stat_cols[1].metric(__change_font("Skewness"),
                                        #                     __print_metric(variable['skewness']))
                                        # stat_cols[1].metric(__change_font("Sum"), __print_metric(variable['sum']))
                                        # stat_cols[1].metric(__change_font("Variance"),
                                        #                     __print_metric(variable['variance']))
                                        # if variable['monotonic'] == 0:
                                        #     stat_cols[1].metric(__change_font("Monotonicity"), f"Not monotonic")
                                        # else:
                                        #     stat_cols[1].metric(__change_font("Monotonicity"), f"Monotonic")

                                with tab2:

                                    value_counts = {value_counts['value']: value_counts['count']
                                                    for value_counts in variable['value_counts_without_nan']}

                                    value_counts = dict(
                                        sorted(value_counts.items(), key=lambda item: item[1], reverse=True))
                                    value_counts = __reduceCategoricalDict(value_counts, 10)
                                    values, counts, frequency = st.columns(3)
                                    with values:
                                        st.subheader("Value")
                                    with counts:
                                        st.subheader("Count")
                                    with frequency:
                                        st.subheader("Frequency (%)")
                                    for val, count in value_counts.items():
                                        with st.container():
                                            cols = st.columns(3)
                                            cols[0].write(__print_metric(val))
                                            cols[1].write(__print_metric(count))
                                            value = (count / (variable['count'] + variable['num_missing']))
                                            cols[2].write(__print_metric(value, percentage=True))

                                    if variable['num_missing'] > 0:
                                        with st.container():
                                            cols = st.columns(3)
                                            cols[0].write(f"{'(Missing value)'}")
                                            cols[1].write(__print_metric(variable['num_missing']))
                                            value = (variable['num_missing'] / (variable['count'] +
                                                                                variable['num_missing']))
                                            cols[2].write(__print_metric(value, percentage=True))

                                with tab3:
                                    min_tab, max_tab = st.tabs(['Minimum 5 values', 'Maximum 5 values'])
                                    with min_tab:
                                        min_value_counts = {value_counts['value']: value_counts['count']
                                                            for value_counts in variable['value_counts_index_sorted']}
                                        five_min_value_counts = {value: min_value_counts[value] for value in
                                                                 list(min_value_counts)[:5]}
                                        values, counts, frequency = st.columns(3)
                                        with values:
                                            st.subheader("Value")
                                        with counts:
                                            st.subheader("Count")
                                        with frequency:
                                            st.subheader("Frequency (%)")
                                        for val, count in five_min_value_counts.items():
                                            with st.container():
                                                cols = st.columns(3)
                                                cols[0].write(__print_metric(val))
                                                cols[1].write(__print_metric(count))
                                                value = (count / (variable['count'] + variable['num_missing']))
                                                cols[2].write(__print_metric(value, percentage=True))

                                    with max_tab:
                                        max_value_counts = {value_counts['value']: value_counts['count']
                                                            for value_counts in variable['value_counts_index_sorted']}
                                        five_max_value_counts = {value: max_value_counts[value] for value in
                                                                 list(reversed(list(max_value_counts)))[:5]}
                                        values, counts, frequency = st.columns(3)
                                        with values:
                                            st.subheader("Value")
                                        with counts:
                                            st.subheader("Count")
                                        with frequency:
                                            st.subheader("Frequency (%)")
                                        for val, count in five_max_value_counts.items():
                                            with st.container():
                                                cols = st.columns(3)
                                                cols[0].write(__print_metric(val))
                                                cols[1].write(__print_metric(count))
                                                value = (count / (variable['count'] + variable['num_missing']))
                                                cols[2].write(__print_metric(value, percentage=True))

                            elif variable['type'] == 'Geometry':
                                stats, plot = st.columns(2)
                                with stats:
                                    st.metric(__change_font("Distinct"), __print_metric(variable['n_distinct']))
                                    st.metric(__change_font("Distinct (\%)"),
                                              __print_metric(variable['p_distinct'], percentage=True))
                                    st.metric(__change_font("Missing values"), __print_metric(variable['num_missing']))
                                    st.metric(__change_font("Missing values (\%)"),
                                              __print_metric(variable['p_missing'], percentage=True))
                                    st.metric(__change_font("Memory size"),
                                              __print_metric(variable['memory_size'], memory=True))

                                with plot:
                                    categories = {category['type']: category['count']
                                                  for category in variable['geom_type_distribution']}

                                    categories = dict(
                                        sorted(categories.items(), key=lambda item: item[1], reverse=True))
                                    categories = __reduceCategoricalDict(categories, 10)
                                    categories_df = pd.DataFrame(categories.items(), columns=['Geometry', 'Count'])
                                    categories_df = categories_df[categories_df.Count != 0]
                                    categories_df['Geometry'] = categories_df['Geometry'].apply(__fix_length)

                                    fig = px.pie(categories_df, values='Count', names='Geometry')
                                    fig.update_traces(textposition='inside', textfont_size=14)
                                    fig.update_layout(title_text='Geometry Distribution', title_x=0.5,
                                                      title_y=1, title_xanchor='center',
                                                      title_yanchor='top', legend_font_size=14,
                                                      title_font_size=18)
                                    st.plotly_chart(fig, use_container_width=True)

                                tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Overview", "Sample", "Mbr",
                                                                                    "Convex Hull", "Heat Map",
                                                                                    "Frequencies", "Distributions"])
                                with tab1:
                                    unique, geometry = st.columns(2)

                                    with unique:
                                        st.subheader("Unique")
                                        st.metric(__change_font("Unique"), __print_metric(variable['n_unique']))
                                        st.metric(__change_font("Unique (\%)"),
                                                  __print_metric(variable['uniqueness'], percentage=True))
                                    with geometry:
                                        st.subheader("Geometry")
                                        st.metric(__change_font("Crs"), __print_metric(variable['crs']))
                                        st.metric(__change_font("Centroid"), __print_metric(variable['centroid']))

                                with tab2:
                                    samples = {sample['row']: sample['value']
                                               for sample in variable['samples']}

                                    sample_df = pd.DataFrame.from_dict(samples, orient="index",
                                                                       columns=["geometry"])
                                    sample_gdf = gpd.GeoDataFrame(
                                        sample_df, geometry=gpd.GeoSeries.from_wkt(sample_df.geometry))
                                    centroid = sample_gdf.geometry.unary_union.centroid
                                    lon = centroid.x
                                    lat = centroid.y
                                    m1 = folium.Map(location=[lat, lon], tiles="CartoDB positron")
                                    for _, r in sample_gdf.iterrows():
                                        gjson = folium.GeoJson(data=gpd.GeoSeries(r[0]).to_json())
                                        gjson.add_to(m1)

                                    folium_static(m1)

                                with tab3:
                                    mbr_box = gpd.GeoSeries.from_wkt([variable["mbr"]])
                                    gjson = folium.GeoJson(data=mbr_box.to_json(),
                                                           style_function=lambda x: {"fillColor": "orange"})

                                    centroid = mbr_box.centroid
                                    lon = centroid.x
                                    lat = centroid.y
                                    m2 = folium.Map(location=[lat, lon], tiles="CartoDB positron")
                                    gjson.add_to(m2)

                                    folium_static(m2)

                                with tab4:
                                    convex_hull = gpd.GeoSeries.from_wkt([variable["union_convex_hull"]])
                                    gjson2 = folium.GeoJson(data=convex_hull.to_json(),
                                                            style_function=lambda x: {"fillColor": "orange"})
                                    centroid = convex_hull.centroid
                                    lon = centroid.x
                                    lat = centroid.y
                                    m3 = folium.Map(location=[lat, lon], tiles="CartoDB positron")
                                    gjson2.add_to(m3)

                                    folium_static(m3)

                                with tab5:
                                    heatmap_df = pd.DataFrame(variable["heatmap"]).values
                                    centroid = gpd.GeoSeries.from_wkt([variable["centroid"]])
                                    lon = centroid.x
                                    lat = centroid.y
                                    m4 = folium.Map(location=[lat, lon], tiles="CartoDB positron")
                                    HeatMap(heatmap_df).add_to(folium.FeatureGroup(name="Heat Map").add_to(m4))
                                    folium.LayerControl().add_to(m4)

                                    m4.fit_bounds(m4.get_bounds(), padding=(30, 30))
                                    folium_static(m4)

                                with tab6:
                                    value_counts = {value_counts['value']: value_counts['count']
                                                    for value_counts in variable['value_counts_without_nan']}

                                    value_counts = dict(
                                        sorted(value_counts.items(), key=lambda item: item[1], reverse=True))
                                    value_counts = __reduceCategoricalDict(value_counts, 10)
                                    values, counts, frequency = st.columns(3)
                                    with values:
                                        st.subheader("Value")
                                    with counts:
                                        st.subheader("Count")
                                    with frequency:
                                        st.subheader("Frequency (%)")
                                    for val, count in value_counts.items():
                                        with st.container():
                                            cols = st.columns(3)
                                            cols[0].write(__print_metric(val))
                                            cols[1].write(__print_metric(count))
                                            value = (count / (variable['count'] + variable['num_missing']))
                                            cols[2].write(__print_metric(value, percentage=True))

                                    if variable['num_missing'] > 0:
                                        with st.container():
                                            cols = st.columns(3)
                                            cols[0].write(f"{'(Missing value)'}")
                                            cols[1].write(__print_metric(variable['num_missing']))
                                            value = (variable['num_missing'] / (variable['count'] +
                                                                                variable['num_missing']))
                                            cols[2].write(__print_metric(value, percentage=True))

                                with tab7:
                                    length, area = st.columns(2)

                                    with length:
                                        tmp = variable['length_distribution']
                                        st.subheader("Length")
                                        length_dict = {'Mean': __print_metric(tmp['average']),
                                                       'Standard Deviation': __print_metric(tmp['stddev']),
                                                       'Kurtosis': __print_metric(tmp['kurtosis']),
                                                       'Skewness': __print_metric(tmp['skewness']),
                                                       'Variance': __print_metric(tmp['variance'])}

                                        length_df = pd.DataFrame(length_dict.items(),
                                                                 columns=['Statistics', 'Values'])
                                        st.dataframe(length_df, use_container_width=True, hide_index=True)

                                        # st.metric(__change_font("Mean"), __print_metric(tmp['average']))
                                        # st.metric(__change_font("Standard Deviation"), __print_metric(tmp['stddev']))
                                        # st.metric(__change_font("Kurtosis"), __print_metric(tmp['kurtosis']))
                                        # st.metric(__change_font("Skewness"), __print_metric(tmp['skewness']))
                                        # st.metric(__change_font("Variance"), __print_metric(tmp['variance']))
                                        fig = go.Figure()
                                        fig.add_trace(go.Box(
                                            q1=[tmp['percentile25']],
                                            median=[tmp['median']],
                                            mean=[tmp['average']],
                                            q3=[tmp['percentile75']],
                                            lowerfence=[tmp['min']],
                                            upperfence=[tmp['max']],
                                            boxmean='sd',
                                            boxpoints=False,
                                            sd=[tmp['stddev']],
                                            showlegend=False,
                                            x0=tmp['name']
                                        ))
                                        fig.update_layout(title_text="Values Distribution", title_x=0.5,
                                                          title_y=0.9, title_xanchor='center',
                                                          title_yanchor='top',
                                                          title_font_size=18)

                                        fig.update_xaxes(
                                            title_font_size=16,
                                            tickfont_size=16
                                        )

                                        fig.update_yaxes(
                                            tickfont_size=16
                                        )
                                        st.plotly_chart(fig, use_container_width=True)

                                    with area:
                                        tmp = variable['area_distribution']
                                        st.subheader("Area")
                                        area_dict = {'Mean': __print_metric(tmp['average']),
                                                     'Standard Deviation': __print_metric(tmp['stddev']),
                                                     'Kurtosis': __print_metric(tmp['kurtosis']),
                                                     'Skewness': __print_metric(tmp['skewness']),
                                                     'Variance': __print_metric(tmp['variance'])}

                                        area_df = pd.DataFrame(area_dict.items(),
                                                               columns=['Statistics', 'Values'])
                                        st.dataframe(area_df, use_container_width=True, hide_index=True)
                                        # st.metric(__change_font("Mean"), __print_metric(tmp['average']))
                                        # st.metric(__change_font("Standard Deviation"), __print_metric(tmp['stddev']))
                                        # st.metric(__change_font("Kurtosis"), __print_metric(tmp['kurtosis']))
                                        # st.metric(__change_font("Skewness"), __print_metric(tmp['skewness']))
                                        # st.metric(__change_font("Variance"), __print_metric(tmp['variance']))
                                        fig = go.Figure()
                                        fig.add_trace(go.Box(
                                            q1=[tmp['percentile25']],
                                            median=[tmp['median']],
                                            mean=[tmp['average']],
                                            q3=[tmp['percentile75']],
                                            lowerfence=[tmp['min']],
                                            upperfence=[tmp['max']],
                                            boxmean='sd',
                                            boxpoints=False,
                                            sd=[tmp['stddev']],
                                            showlegend=False,
                                            x0=tmp['name']
                                        ))
                                        fig.update_layout(title_text="Values Distribution", title_x=0.5,
                                                          title_y=0.9, title_xanchor='center',
                                                          title_yanchor='top',
                                                          title_font_size=18)

                                        fig.update_xaxes(
                                            title_font_size=16,
                                            tickfont_size=16
                                        )

                                        fig.update_yaxes(
                                            tickfont_size=16
                                        )
                                        st.plotly_chart(fig, use_container_width=True)

                            elif variable['type'] == 'DateTime':
                                stats1, stats2, plot = st.columns(3)
                                with stats1:
                                    st.metric(__change_font("Distinct"), __print_metric(variable['n_distinct']))
                                    st.metric(__change_font("Distinct (\%)"),
                                              __print_metric(variable['p_distinct'], percentage=True))
                                    st.metric(__change_font("Missing values"), __print_metric(variable['num_missing']))
                                    st.metric(__change_font("Missing values (\%)"),
                                              __print_metric(variable['p_missing'], percentage=True))
                                    st.metric(__change_font("Memory size"),
                                              __print_metric(variable['memory_size'], memory=True))

                                with stats2:
                                    st.metric(__change_font("Minimum"), __print_metric(variable['start']))
                                    st.metric(__change_font("Maximum"), __print_metric(variable['end']))

                                with plot:
                                    bar_df = pd.DataFrame({'counts': variable['histogram_counts']})
                                    bar_df['Start'] = variable['histogram_bins'][:-1]
                                    bar_df['End'] = variable['histogram_bins'][1:]

                                    bar_df['Start'] = bar_df['Start'].apply(__float_to_datetime)
                                    bar_df['End'] = bar_df['End'].apply(__float_to_datetime)

                                    bar_freq = alt.Chart(bar_df).mark_bar().encode(
                                        x=alt.X('Start:T', bin='binned',
                                                title='Histogram with fixed size bins (bins=' + str(len(
                                                    variable['histogram_bins']) - 1) + ')', type='temporal'),
                                        x2=alt.X2('End:T', title=''),
                                        y=alt.Y('counts', title='Frequency'),
                                        tooltip=['Start', 'End', 'counts']
                                    ).configure_axis(
                                        labelFontSize=16,
                                        titleFontSize=16
                                    ).interactive()

                                    plot.altair_chart(bar_freq, use_container_width=True)

                            elif variable['type'] == 'TimeSeries':
                                with st.container():
                                    cols = st.columns(3)
                                    cols[0].metric(__change_font("Distinct"), __print_metric(variable['n_distinct']))
                                    cols[0].metric(__change_font("Distinct (\%)"),
                                                   __print_metric(variable['p_distinct'], percentage=True))
                                    cols[0].metric(__change_font("Missing values"),
                                                   __print_metric(variable['num_missing']))
                                    cols[0].metric(__change_font("Missing values (\%)"),
                                                   __print_metric(variable['p_missing'], percentage=True))
                                    cols[0].metric(__change_font("Infinite"), __print_metric(variable['n_infinite']))
                                    cols[0].metric(__change_font("Infinite (\%)"),
                                                   __print_metric(variable['p_infinite'], percentage=True))
                                    cols[0].metric(__change_font("Mean"), __print_metric(variable['average']))

                                    cols[0].metric(__change_font("Minimum"), __print_metric(variable['min']))
                                    cols[1].metric(__change_font("Maximum"), __print_metric(variable['max']))
                                    cols[1].metric(__change_font("Zeros"), __print_metric(variable['n_zeros']))
                                    cols[1].metric(__change_font("Zeros (\%)"),
                                                   __print_metric(variable['p_zeros'], percentage=True))
                                    cols[1].metric(__change_font("Negative"), __print_metric(variable['n_negative']))
                                    cols[1].metric(__change_font("Negative (\%)"),
                                                   __print_metric(variable['p_negative']))
                                    cols[1].metric(__change_font("Memory size"),
                                                   __print_metric(variable['memory_size'], memory=True))

                                    series = {key_value['key']: key_value['value']
                                              for key_value in variable['series']}

                                    data_df = pd.DataFrame.from_dict(series, orient='index')

                                    data_df.rename(columns={data_df.columns[0]: 'values'}, inplace=True)

                                    data_df['keys'] = data_df.index
                                    data_df.reset_index(inplace=True)

                                    ts_line_chart = alt.Chart(data_df).mark_line().encode(
                                        x=alt.X('keys', title='Date Position'),
                                        y=alt.Y('values', title='Value')
                                    ).configure_axis(
                                        labelFontSize=16,
                                        titleFontSize=16
                                    ).interactive()

                                    cols[2].altair_chart(ts_line_chart, use_container_width=True)

                                    st.divider()
                                    if len(variable['gaps_distribution']) != 0:
                                        st.subheader("Gap Length")
                                        gaps_col = st.columns(2)
                                        gaps_distr = variable['gaps_distribution']
                                        gaps_col[0].subheader('Statistics')

                                        gaps_dict = {'Mean': __print_metric(gaps_distr['average']),
                                                     'Standard Deviation': __print_metric(gaps_distr['stddev']),
                                                     'Kurtosis': __print_metric(gaps_distr['kurtosis']),
                                                     'Skewness': __print_metric(gaps_distr['skewness']),
                                                     'Variance': __print_metric(gaps_distr['variance'])}

                                        gaps_df = pd.DataFrame(gaps_dict.items(), columns=['Statistics', 'Values'])
                                        gaps_col[0].dataframe(gaps_df, use_container_width=True, hide_index=True)

                                        # gaps_col[0].metric(__change_font("Mean"), __print_metric(gaps_distr['average']))
                                        # gaps_col[0].metric(__change_font("Standard Deviation"),
                                        #                    __print_metric(gaps_distr['stddev']))
                                        # gaps_col[0].metric(__change_font("Kurtosis"),
                                        #                    __print_metric(gaps_distr['kurtosis']))
                                        # gaps_col[0].metric(__change_font("Skewness"),
                                        #                    __print_metric(gaps_distr['skewness']))
                                        # gaps_col[0].metric(__change_font("Variance"),
                                        #                    __print_metric(gaps_distr['variance']))
                                        fig = go.Figure()
                                        fig.add_trace(go.Box(
                                            q1=[gaps_distr['percentile25']],
                                            median=[gaps_distr['median']],
                                            mean=[gaps_distr['average']],
                                            q3=[gaps_distr['percentile75']],
                                            lowerfence=[gaps_distr['min']],
                                            upperfence=[gaps_distr['max']],
                                            boxmean='sd',
                                            boxpoints=False,
                                            sd=[variable['stddev']],
                                            showlegend=False,
                                            x0=variable['name']
                                        ))
                                        # gaps_col[1].subheader('Gap Type Distribution')
                                        fig.update_layout(title_text="Gap Type Distribution", title_x=0.5,
                                                          title_y=0.9, title_xanchor='center',
                                                          title_yanchor='top',
                                                          title_font_size=18
                                                          )

                                        fig.update_xaxes(
                                            title_font_size=16,
                                            tickfont_size=16
                                        )

                                        fig.update_yaxes(
                                            tickfont_size=16
                                        )

                                        gaps_col[1].plotly_chart(fig, use_container_width=True)
                                    else:
                                        st.subheader('There are no gaps in the timeseries')

                                tab1, tab2, tab3, tab4, tab5 = st.tabs(["Statistics", "Histogram", "Common Values",
                                                                        "Extreme Values", "Autocorrelation"])

                                with tab1:
                                    with st.container():
                                        stat_cols = st.columns(3)
                                        stat_cols[0].subheader('Quantile statistics')

                                        quant_stats_dict = {'Mean': __print_metric(variable['average']),
                                                            'Range': __print_metric(variable['range']),
                                                            'Interquartile range (IQR)': __print_metric(
                                                                variable['iqr'])}

                                        quan_stats_df = pd.DataFrame(quant_stats_dict.items(),
                                                                     columns=['Statistics', 'Values'])

                                        stat_cols[0].dataframe(quan_stats_df, use_container_width=True, hide_index=True)

                                        #
                                        # stat_cols[0].metric(__change_font("Mean"), __print_metric(variable['average']))
                                        # stat_cols[0].metric(__change_font("Range"), __print_metric(variable['range']))
                                        # stat_cols[0].metric(__change_font("Interquartile range (IQR)"),
                                        #                     __print_metric(variable['iqr']))
                                        fig = go.Figure()
                                        fig.add_trace(go.Box(
                                            q1=[variable['percentile25']],
                                            median=[variable['median']],
                                            mean=[variable['average']],
                                            q3=[variable['percentile75']],
                                            lowerfence=[variable['min']],
                                            upperfence=[variable['max']],
                                            boxmean='sd',
                                            boxpoints=False,
                                            sd=[variable['stddev']],
                                            showlegend=False,
                                            x0=variable['name']
                                        ))
                                        fig.update_layout(title_text="Values Distribution", title_x=0.5,
                                                          title_y=0.9, title_xanchor='center',
                                                          title_yanchor='top',
                                                          title_font_size=18)

                                        fig.update_xaxes(
                                            title_font_size=16,
                                            tickfont_size=16
                                        )

                                        fig.update_yaxes(
                                            tickfont_size=16
                                        )

                                        stat_cols[0].plotly_chart(fig, use_container_width=True)

                                        stat_cols[1].subheader('Descriptive statistics')
                                        desc_stats_dict = {'Standard deviation': __print_metric(variable['stddev']),
                                                           'Coefficient of variation (CV)': __print_metric(
                                                               variable['cv']),
                                                           'Kurtosis': __print_metric(variable['kurtosis']),
                                                           'Median Absolute Deviation (MAD)': __print_metric(
                                                               variable['mad']),
                                                           'Skewness': __print_metric(variable['skewness']),
                                                           'Sum': __print_metric(variable['sum']),
                                                           'Variance': __print_metric(variable['variance'])
                                                           }

                                        # stat_cols[1].metric(__change_font("Standard deviation"),
                                        #                     __print_metric(variable['stddev']))
                                        # stat_cols[1].metric(__change_font("Coefficient of variation (CV)"),
                                        #                     __print_metric(variable['cv']))
                                        # stat_cols[1].metric(__change_font("Kurtosis"),
                                        #                     __print_metric(variable['kurtosis']))
                                        #
                                        # stat_cols[1].metric(__change_font("Median Absolute Deviation (MAD)"),
                                        #                     __print_metric(variable['mad']))
                                        # stat_cols[1].metric(__change_font("Skewness"),
                                        #                     __print_metric(variable['skewness']))
                                        # stat_cols[1].metric(__change_font("Sum"), __print_metric(variable['sum']))
                                        # stat_cols[1].metric(__change_font("Variance"),
                                        #                     __print_metric(variable['variance']))

                                        if variable['monotonic'] == 0:
                                            # stat_cols[1].metric(__change_font("Monotonicity"), f"Not monotonic")
                                            desc_stats_dict['Monotonicity'] = 'Not monotonic'
                                        else:
                                            # stat_cols[1].metric(__change_font("Monotonicity"), f"Monotonic")
                                            desc_stats_dict['Monotonicity'] = 'Monotonic'

                                        desc_stats_dict['Augmented Dickey-Fuller test p-value'] = __print_metric(
                                            variable['add_fuller'])
                                        # stat_cols[1].metric(__change_font("Augmented Dickey-Fuller test p-value"),
                                        #                     __print_metric(variable['add_fuller']))

                                        quan_stats_df = pd.DataFrame(desc_stats_dict.items(),
                                                                     columns=['Statistics', 'Values'])

                                        stat_cols[1].dataframe(quan_stats_df, use_container_width=True, hide_index=True)

                                        stat_cols[2].subheader('TsFresh statistics')
                                        tsfresh_stats_dict = {'Absolute Energy': __print_metric(variable['abs_energy']),
                                                              'Sum over the abs value of consecutive changes':
                                                                  __print_metric(variable['abs_sum_changes']),
                                                              'Number of values above mean': __print_metric(
                                                                  variable['len_above_mean']),
                                                              'Number of values below mean': __print_metric(
                                                                  variable['len_below_mean']),
                                                              'Number of peaks': __print_metric(variable['num_peaks'])
                                                              }

                                        # stat_cols[2].metric(__change_font("Absolute Energy"),
                                        #                     __print_metric(variable['abs_energy']))
                                        # stat_cols[2].metric(
                                        #     __change_font("Sum over the absolute value of consecutive changes"),
                                        #     __print_metric(variable['abs_sum_changes']))
                                        # stat_cols[2].metric(__change_font("Number of values above mean"),
                                        #                     __print_metric(variable['len_above_mean']))
                                        # stat_cols[2].metric(__change_font("Number of values below mean"),
                                        #                     __print_metric(variable['len_below_mean']))
                                        # stat_cols[2].metric(__change_font("Number of peaks"),
                                        #                     __print_metric(variable['num_peaks']))

                                        tsfresh_stats_df = pd.DataFrame(tsfresh_stats_dict.items(),
                                                                        columns=['Statistics', 'Values'])

                                        stat_cols[2].dataframe(tsfresh_stats_df, use_container_width=True,
                                                               hide_index=True)

                                with tab2:
                                    bar_df = pd.DataFrame({'counts': variable['histogram_counts']})

                                    binned = bar_df.groupby(
                                        pd.cut(bar_df.counts, bins=variable['histogram_bins'])).count()
                                    binned['start_value'] = variable['histogram_bins'][:-1]
                                    binned['end_value'] = variable['histogram_bins'][1:]
                                    binned['counts'] = variable['histogram_counts']
                                    binned = binned.reset_index(drop=True)

                                    bar_freq = alt.Chart(binned).mark_bar().encode(
                                        x=alt.X('start_value', bin='binned',
                                                title='Histogram with fixed size bins (bins=' + str(len(
                                                    variable['histogram_bins']) - 1) + ')'),
                                        x2=alt.X2('end_value', title=''),
                                        y=alt.Y('counts', title='Frequency'),
                                        tooltip=['start_value', 'end_value', 'counts']
                                    ).configure_axis(
                                        labelFontSize=16,
                                        titleFontSize=16
                                    ).interactive()

                                    st.altair_chart(bar_freq, use_container_width=True)

                                with tab3:

                                    value_counts = {value_counts['value']: value_counts['count']
                                                    for value_counts in variable['value_counts_without_nan']}

                                    value_counts = dict(
                                        sorted(value_counts.items(), key=lambda item: item[1], reverse=True))
                                    value_counts = __reduceCategoricalDict(value_counts, 10)
                                    values, counts, frequency = st.columns(3)
                                    with values:
                                        st.subheader("Value")
                                    with counts:
                                        st.subheader("Count")
                                    with frequency:
                                        st.subheader("Frequency (%)")
                                    for val, count in value_counts.items():
                                        with st.container():
                                            cols = st.columns(3)
                                            cols[0].write(__print_metric(val))
                                            cols[1].write(__print_metric(count))
                                            value = (count / (variable['count'] + variable['num_missing']))
                                            cols[2].write(__print_metric(value, percentage=True))

                                    if variable['num_missing'] > 0:
                                        with st.container():
                                            cols = st.columns(3)
                                            cols[0].write(f"{'(Missing value)'}")
                                            cols[1].write(__print_metric(variable['num_missing']))
                                            value = (variable['num_missing'] / (variable['count'] +
                                                                                variable['num_missing']))
                                            cols[2].write(__print_metric(value, percentage=True))

                                with tab4:
                                    min_tab, max_tab = st.tabs(['Minimum 10 values', 'Maximum 10 values'])
                                    with min_tab:
                                        min_value_counts = {value_counts['value']: value_counts['count']
                                                            for value_counts in variable['value_counts_index_sorted']}
                                        five_min_value_counts = {value: min_value_counts[value] for value in
                                                                 list(min_value_counts)[:10]}
                                        values, counts, frequency = st.columns(3)
                                        with values:
                                            st.subheader("Value")
                                        with counts:
                                            st.subheader("Count")
                                        with frequency:
                                            st.subheader("Frequency (%)")
                                        for val, count in five_min_value_counts.items():
                                            with st.container():
                                                cols = st.columns(3)
                                                cols[0].write(__print_metric(val))
                                                cols[1].write(__print_metric(count))
                                                value = (count / (variable['count'] + variable['num_missing']))
                                                cols[2].write(__print_metric(value, percentage=True))

                                    with max_tab:
                                        max_value_counts = {value_counts['value']: value_counts['count']
                                                            for value_counts in variable['value_counts_index_sorted']}
                                        five_max_value_counts = {value: max_value_counts[value] for value in
                                                                 list(reversed(list(max_value_counts)))[:10]}
                                        values, counts, frequency = st.columns(3)
                                        with values:
                                            st.subheader("Value")
                                        with counts:
                                            st.subheader("Count")
                                        with frequency:
                                            st.subheader("Frequency (%)")
                                        for val, count in five_max_value_counts.items():
                                            with st.container():
                                                cols = st.columns(3)
                                                cols[0].write(__print_metric(val))
                                                cols[1].write(__print_metric(count))
                                                value = (count / (variable['count'] + variable['num_missing']))
                                                cols[2].write(__print_metric(value, percentage=True))
                                with tab5:
                                    series = {key_value['key']: key_value['value']
                                              for key_value in variable['series']}

                                    pd_series = pd.Series(series)
                                    lag = _get_ts_lag(pd_series)
                                    fig_size: tuple = (15, 5)
                                    color = "#377eb8"
                                    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=fig_size)

                                    plot_acf(
                                        pd_series.dropna(),
                                        lags=lag,
                                        ax=axes[0],
                                        title="ACF",
                                        fft=True,
                                        color=color,
                                        vlines_kwargs={"colors": color},
                                    )

                                    plot_pacf(
                                        pd_series.dropna(),
                                        lags=lag,
                                        ax=axes[1],
                                        title="PACF",
                                        method="ywm",
                                        color=color,
                                        vlines_kwargs={"colors": color},
                                    )

                                    # Change Colors
                                    axes[0].set_facecolor('#002b36')
                                    axes[0].spines['bottom'].set_color('white')
                                    axes[0].spines['top'].set_color('white')
                                    axes[0].spines['left'].set_color('white')
                                    axes[0].spines['right'].set_color('white')
                                    axes[0].xaxis.label.set_color('white')
                                    axes[0].tick_params(axis='both', colors='white')
                                    axes[0].set_title('ACF', color='white')
                                    axes[1].set_facecolor('#002b36')
                                    axes[1].spines['bottom'].set_color('white')
                                    axes[1].spines['top'].set_color('white')
                                    axes[1].spines['left'].set_color('white')
                                    axes[1].spines['right'].set_color('white')
                                    axes[1].xaxis.label.set_color('white')
                                    axes[1].tick_params(axis='both', colors='white')
                                    axes[1].set_title('PACF', color='white')
                                    for ax in axes:
                                        for item in ax.collections:
                                            if type(item) == PolyCollection:
                                                item.set_facecolor(color)
                                    fig.patch.set_facecolor('#002b36')
                                    st.pyplot(fig)
                            elif variable['type'] == 'Boolean':
                                cols = st.columns(2)
                                cols[0].metric(__change_font("Distinct"), __print_metric(variable['n_distinct']))
                                cols[0].metric(__change_font("Distinct (\%)"),
                                               __print_metric(variable['p_distinct'], percentage=True))
                                cols[0].metric(__change_font("Missing values"),
                                               __print_metric(variable['num_missing']))
                                cols[0].metric(__change_font("Missing values (\%)"),
                                               __print_metric(variable['p_missing'], percentage=True))
                                cols[0].metric(__change_font("Memory size"),
                                               __print_metric(variable['memory_size'], memory=True))

                                categories_dict = {str(cat['value']): cat['count']
                                                   for cat in variable['value_counts_without_nan']}
                                categories_dict = dict(
                                    sorted(categories_dict.items(), key=lambda item: item[1], reverse=True))
                                categories_dict = __reduceCategoricalDict(categories_dict, 10)

                                categories_df = pd.DataFrame(categories_dict.items(),
                                                             columns=['Value', 'Count'])
                                if variable['num_missing'] > 0:
                                    categories_df.loc[len(categories_df.index)] = ['Missing',
                                                                                   variable['num_missing']]

                                categories_df['Frequency'] = (categories_df['Count'] /
                                                              (variable['count'] + variable['num_missing']))

                                categories_df.drop('Count', axis=1, inplace=True)

                                fig = px.pie(categories_df, values='Frequency', names='Value')
                                fig.update_traces(textposition='inside', textfont_size=14)
                                fig.update_layout(title_text='Boolean Types Distribution', title_x=0.5,
                                                  title_y=1, title_xanchor='center',
                                                  title_yanchor='top', legend_font_size=14,
                                                  title_font_size=18)
                                cols[1].plotly_chart(fig, use_container_width=True)

                            else:
                                st.metric(__change_font("Type"), f"{variable['type']}")

            elif config_dict['table']['profiler_type'] == 'Hierarchical':

                uniqueness_counts_dict, types_count_dict, types_names_dict, uniqueness_names_dict, nested_levels_dict = \
                    __calc_hierarchical(config_dict['variables'])

                table_stats = config_dict['table']
                dataset_stats, rec_types = st.columns(2)
                with dataset_stats:
                    st.subheader("Dataset Statistics")
                    st.metric(__change_font("Number of records"), __print_metric(table_stats['num_records']))
                    # st.metric("Profiler Type", __print_metric(table_stats['profiler_type']))
                    st.metric(__change_font("Total file size"), __print_metric(table_stats['byte_size'], memory=True))

                with rec_types:
                    st.subheader("Record Types")
                    for v_type, count in types_count_dict.items():
                        st.metric(__change_font(v_type), __print_metric(count), help=types_names_dict[v_type])

                tab1, tab2 = st.tabs(["Depth", "Uniqueness"])

                with tab1:
                    depth_distr, plot = st.columns(2)

                    with depth_distr:
                        tmp = table_stats['depth_distribution']
                        st.subheader("Distribution")
                        depth_dict = {'Mean': __print_metric(tmp['average']),
                                      'Standard Deviation': __print_metric(tmp['stddev']),
                                      'Kurtosis': __print_metric(tmp['kurtosis']),
                                      'Skewness': __print_metric(tmp['skewness']),
                                      'Variance': __print_metric(tmp['variance'])}

                        depth_df = pd.DataFrame(depth_dict.items(),
                                                columns=['Statistics', 'Values'])
                        st.dataframe(depth_df, use_container_width=True, hide_index=True)

                        # st.metric(__change_font("Mean"), __print_metric(tmp['average']))
                        # st.metric(__change_font("Standard Deviation"), __print_metric(tmp['stddev']))
                        # st.metric(__change_font("Kurtosis"), __print_metric(tmp['kurtosis']))
                        # st.metric(__change_font("Skewness"), __print_metric(tmp['skewness']))
                        # st.metric(__change_font("Variance"), __print_metric(tmp['variance']))
                        fig = go.Figure()
                        fig.add_trace(go.Box(
                            q1=[tmp['percentile25']],
                            median=[tmp['median']],
                            mean=[tmp['average']],
                            q3=[tmp['percentile75']],
                            lowerfence=[tmp['min']],
                            upperfence=[tmp['max']],
                            boxmean='sd',
                            boxpoints=False,
                            sd=[tmp['stddev']],
                            showlegend=False
                        ))

                        fig.update_layout(title_text="Values Distribution", title_x=0.5,
                                          title_y=0.9, title_xanchor='center',
                                          title_yanchor='top',
                                          title_font_size=18)

                        fig.update_xaxes(
                            showticklabels=False,
                            title_font_size=16,
                            tickfont_size=16
                        )

                        fig.update_yaxes(
                            tickfont_size=16
                        )

                        st.plotly_chart(fig, use_container_width=True)
                    with plot:
                        # st.subheader("Nested Level Distribution")
                        nested_levels_dict = dict(
                            sorted(nested_levels_dict.items(), key=lambda item: item[1], reverse=True))
                        nested_levels_df = pd.DataFrame(nested_levels_dict.items(), columns=['Nested Level', 'Count'])
                        nested_levels_df = nested_levels_df[nested_levels_df.Count != 0]
                        # nested_levels_df['Nested Level'] = nested_levels_df['Nested Level'].apply(__fix_length)

                        fig = px.pie(nested_levels_df, values='Count', names='Nested Level')
                        fig.update_traces(textposition='inside', textfont_size=14)
                        fig.update_layout(title_text='Nested Level Distribution', title_x=0.5,
                                          title_y=1, title_xanchor='center',
                                          title_yanchor='top', legend_font_size=14,
                                          title_font_size=18)
                        st.plotly_chart(fig, use_container_width=True)
                with tab2:
                    unique, counts, records = st.columns(3)
                    with unique:
                        st.subheader("Uniqueness")
                    with counts:
                        st.subheader("Record Percentage")
                    with records:
                        st.subheader("Record Names")

                    uniqueness_counts_dict = dict(
                        sorted(uniqueness_counts_dict.items(), key=lambda item: item[0]))

                    for uniq, count in uniqueness_counts_dict.items():
                        with st.container():
                            cols = st.columns(3)
                            cols[0].write(__print_metric(uniq))
                            cols[1].write(__print_metric(count / table_stats['num_records'], percentage=True))
                            cols[2].write(__print_metric(uniqueness_names_dict[uniq]))

            elif config_dict['table']['profiler_type'] == 'RDFGraph':
                table_stats = config_dict['table']
                dataset_stats, conn_comp = st.columns(2)
                with dataset_stats:
                    st.subheader("Dataset Statistics")
                    dataset_stats1, dataset_stats2 = st.columns(2)
                    with dataset_stats1:
                        st.metric(__change_font("Number of Nodes"), __print_metric(table_stats['num_nodes']))
                        st.metric(__change_font("Number of Edges"), __print_metric(table_stats['num_edges']))
                        # st.metric("Profiler Type", __print_metric(table_stats['profiler_type']))
                        st.metric(__change_font("Total file size"),
                                  __print_metric(table_stats['byte_size'], memory=True))

                    with dataset_stats2:
                        st.metric(__change_font("Number of Namespaces"), __print_metric(table_stats['num_namespaces']))
                        st.metric(__change_font("Number of Classes"), __print_metric(table_stats['num_classes']))
                        st.metric(__change_font("Number of Object Properties"),
                                  __print_metric(table_stats['num_object_properties']))
                        st.metric(__change_font("Number of Datatype Properties"),
                                  __print_metric(table_stats['num_datatype_properties']))
                        st.metric(__change_font("Density"), __print_metric(table_stats['density']))

                with conn_comp:
                    # st.subheader("Connected Component Types")
                    comps = {comp['component_name']: comp['num_nodes']
                             for comp in table_stats['connected_components']}
                    comps = dict(
                        sorted(comps.items(), key=lambda item: item[1], reverse=True))
                    comps = __reduceCategoricalDict(comps, 10)
                    comps_df = pd.DataFrame(comps.items(), columns=['Connected Component', 'Number of Nodes'])
                    comps_df = comps_df[comps_df.loc[:, 'Number of Nodes'] != 0]

                    fig = px.pie(comps_df, values='Number of Nodes', names='Connected Component')
                    fig.update_traces(textposition='inside', textfont_size=14)
                    fig.update_layout(title_text='Connected Component Types Distribution', title_x=0.5,
                                      title_y=1, title_xanchor='center',
                                      title_yanchor='top', legend_font_size=14,
                                      title_font_size=18)
                    st.plotly_chart(fig, use_container_width=True)
                st.divider()
                st.subheader("Distributions")
                degree_cent = False
                degree = False
                in_degree = False
                out_degree = False
                class_distr = False
                tabs = []

                if 'degree_centrality_distribution' in table_stats:
                    degree_cent = True
                    tabs.append('Degree Centrality')

                if 'degree_distribution' in table_stats:
                    degree = True
                    tabs.append('Degree')

                if 'in_degree_distribution' in table_stats:
                    in_degree = True
                    tabs.append('In-Degree')

                if 'out_degree_distribution' in table_stats:
                    out_degree = True
                    tabs.append('Out-Degree')

                if 'class_distribution' in table_stats:
                    class_distr = True
                    tabs.append('Class')

                tabs = st.tabs(tabs)
                count = 0
                if degree_cent:
                    with tabs[count]:
                        tmp = table_stats['degree_centrality_distribution']
                        cols = st.columns(2)
                        degree_dict = {'Mean': __print_metric(tmp['average']),
                                       'Standard Deviation': __print_metric(tmp['stddev']),
                                       'Kurtosis': __print_metric(tmp['kurtosis']),
                                       'Skewness': __print_metric(tmp['skewness']),
                                       'Variance': __print_metric(tmp['variance'])
                                       }

                        degree_df = pd.DataFrame(degree_dict.items(), columns=['Statistics', 'Values'])
                        cols[0].dataframe(degree_df, use_container_width=True, hide_index=True)

                        # cols[0].metric(__change_font("Mean"), __print_metric(tmp['average']))
                        # cols[0].metric(__change_font("Standard Deviation"), __print_metric(tmp['stddev']))
                        # cols[0].metric(__change_font("Kurtosis"), __print_metric(tmp['kurtosis']))
                        # cols[0].metric(__change_font("Skewness"), __print_metric(tmp['skewness']))
                        # cols[0].metric(__change_font("Variance"), __print_metric(tmp['variance']))
                        fig = go.Figure()
                        fig.add_trace(go.Box(
                            q1=[tmp['percentile25']],
                            median=[tmp['median']],
                            mean=[tmp['average']],
                            q3=[tmp['percentile75']],
                            lowerfence=[tmp['min']],
                            upperfence=[tmp['max']],
                            boxmean='sd',
                            boxpoints=False,
                            sd=[tmp['stddev']],
                            showlegend=False
                        ))

                        fig.update_layout(title_text="Values Distribution", title_x=0.5,
                                          title_y=1, title_xanchor='center',
                                          title_yanchor='top',
                                          title_font_size=18)

                        fig.update_xaxes(
                            showticklabels=False,
                            title_font_size=16,
                            tickfont_size=16
                        )

                        fig.update_yaxes(
                            tickfont_size=16
                        )

                        cols[1].plotly_chart(fig, use_container_width=True)
                        count += 1

                if degree:
                    with tabs[count]:
                        tmp = table_stats['degree_distribution']
                        cols = st.columns(2)
                        degree_dict = {'Mean': __print_metric(tmp['average']),
                                       'Standard Deviation': __print_metric(tmp['stddev']),
                                       'Kurtosis': __print_metric(tmp['kurtosis']),
                                       'Skewness': __print_metric(tmp['skewness']),
                                       'Variance': __print_metric(tmp['variance'])
                                       }

                        degree_df = pd.DataFrame(degree_dict.items(), columns=['Statistics', 'Values'])
                        cols[0].dataframe(degree_df, use_container_width=True, hide_index=True)

                        # cols[0].metric(__change_font("Mean"), __print_metric(tmp['average']))
                        # cols[0].metric(__change_font("Standard Deviation"), __print_metric(tmp['stddev']))
                        # cols[0].metric(__change_font("Kurtosis"), __print_metric(tmp['kurtosis']))
                        # cols[0].metric(__change_font("Skewness"), __print_metric(tmp['skewness']))
                        # cols[0].metric(__change_font("Variance"), __print_metric(tmp['variance']))
                        fig = go.Figure()
                        fig.add_trace(go.Box(
                            q1=[tmp['percentile25']],
                            median=[tmp['median']],
                            mean=[tmp['average']],
                            q3=[tmp['percentile75']],
                            lowerfence=[tmp['min']],
                            upperfence=[tmp['max']],
                            boxmean='sd',
                            boxpoints=False,
                            sd=[tmp['stddev']],
                            showlegend=False
                        ))

                        fig.update_layout(title_text="Values Distribution", title_x=0.5,
                                          title_y=1, title_xanchor='center',
                                          title_yanchor='top',
                                          title_font_size=18)

                        fig.update_xaxes(
                            showticklabels=False,
                            title_font_size=16,
                            tickfont_size=16
                        )

                        fig.update_yaxes(
                            tickfont_size=16
                        )

                        cols[1].plotly_chart(fig, use_container_width=True)
                        count += 1

                if in_degree:
                    with tabs[count]:
                        tmp = table_stats['in_degree_distribution']
                        cols = st.columns(2)
                        degree_dict = {'Mean': __print_metric(tmp['average']),
                                       'Standard Deviation': __print_metric(tmp['stddev']),
                                       'Kurtosis': __print_metric(tmp['kurtosis']),
                                       'Skewness': __print_metric(tmp['skewness']),
                                       'Variance': __print_metric(tmp['variance'])
                                       }

                        degree_df = pd.DataFrame(degree_dict.items(), columns=['Statistics', 'Values'])
                        cols[0].dataframe(degree_df, use_container_width=True, hide_index=True)

                        # cols[0].metric(__change_font("Mean"), __print_metric(tmp['average']))
                        # cols[0].metric(__change_font("Standard Deviation"), __print_metric(tmp['stddev']))
                        # cols[0].metric(__change_font("Kurtosis"), __print_metric(tmp['kurtosis']))
                        # cols[0].metric(__change_font("Skewness"), __print_metric(tmp['skewness']))
                        # cols[0].metric(__change_font("Variance"), __print_metric(tmp['variance']))
                        fig = go.Figure()
                        fig.add_trace(go.Box(
                            q1=[tmp['percentile25']],
                            median=[tmp['median']],
                            mean=[tmp['average']],
                            q3=[tmp['percentile75']],
                            lowerfence=[tmp['min']],
                            upperfence=[tmp['max']],
                            boxmean='sd',
                            boxpoints=False,
                            sd=[tmp['stddev']],
                            showlegend=False
                        ))

                        fig.update_layout(title_text="Values Distribution", title_x=0.5,
                                          title_y=1, title_xanchor='center',
                                          title_yanchor='top',
                                          title_font_size=18)

                        fig.update_xaxes(
                            showticklabels=False,
                            title_font_size=16,
                            tickfont_size=16
                        )

                        fig.update_yaxes(
                            tickfont_size=16
                        )

                        cols[1].plotly_chart(fig, use_container_width=True)
                        count += 1

                if out_degree:
                    with tabs[count]:
                        tmp = table_stats['out_degree_distribution']
                        cols = st.columns(2)
                        degree_dict = {'Mean': __print_metric(tmp['average']),
                                       'Standard Deviation': __print_metric(tmp['stddev']),
                                       'Kurtosis': __print_metric(tmp['kurtosis']),
                                       'Skewness': __print_metric(tmp['skewness']),
                                       'Variance': __print_metric(tmp['variance'])
                                       }

                        degree_df = pd.DataFrame(degree_dict.items(), columns=['Statistics', 'Values'])
                        cols[0].dataframe(degree_df, use_container_width=True, hide_index=True)

                        # cols[0].metric(__change_font("Mean"), __print_metric(tmp['average']))
                        # cols[0].metric(__change_font("Standard Deviation"), __print_metric(tmp['stddev']))
                        # cols[0].metric(__change_font("Kurtosis"), __print_metric(tmp['kurtosis']))
                        # cols[0].metric(__change_font("Skewness"), __print_metric(tmp['skewness']))
                        # cols[0].metric(__change_font("Variance"), __print_metric(tmp['variance']))
                        fig = go.Figure()
                        fig.add_trace(go.Box(
                            q1=[tmp['percentile25']],
                            median=[tmp['median']],
                            mean=[tmp['average']],
                            q3=[tmp['percentile75']],
                            lowerfence=[tmp['min']],
                            upperfence=[tmp['max']],
                            boxmean='sd',
                            boxpoints=False,
                            sd=[tmp['stddev']],
                            showlegend=False
                        ))

                        fig.update_layout(title_text="Values Distribution", title_x=0.5,
                                          title_y=1, title_xanchor='center',
                                          title_yanchor='top',
                                          title_font_size=18)

                        fig.update_xaxes(
                            showticklabels=False,
                            title_font_size=16,
                            tickfont_size=16
                        )

                        fig.update_yaxes(
                            tickfont_size=16
                        )

                        cols[1].plotly_chart(fig, use_container_width=True)
                        count += 1

                if class_distr:
                    with tabs[-1]:
                        class_dict = {tmp_class['class_name']: tmp_class['count']
                                      for tmp_class in table_stats['class_distribution']}

                        class_dict = dict(
                            sorted(class_dict.items(), key=lambda item: item[1], reverse=True))
                        class_dict = __reduceCategoricalDict(class_dict, 10)
                        classes, counts = st.columns(2)
                        with classes:
                            st.subheader("Class Name")
                        with counts:
                            st.subheader("Percentage")
                        total = sum(class_dict.values())

                        for val, count in class_dict.items():
                            with st.container():
                                cols = st.columns(2)
                                cols[0].write(__print_metric(val))
                                cols[1].write(__print_metric(count / total, percentage=True))

            elif config_dict['table']['profiler_type'] == 'Textual':
                overview, texts = st.tabs(["Overview", "Texts"])

                with overview:
                    table_stats = config_dict['table']
                    sent_analysis = False
                    if 'sentiment_analysis' in table_stats.keys():
                        sent_analysis = True

                    corpus_stats, plot = st.columns(2)

                    with corpus_stats:
                        st.subheader("Corpus Statistics")
                        st.metric(__change_font("Number of Texts"), __print_metric(table_stats['num_texts']))
                        st.metric(__change_font("Number of Words"), __print_metric(table_stats['num_words']))
                        st.metric(__change_font("Number of Distinct Words"),
                                  __print_metric(table_stats['num_distinct_words']))
                        st.metric(__change_font("Number of Sentences"), __print_metric(table_stats['num_sentences']))
                        st.metric(__change_font("Number of Characters"), __print_metric(table_stats['num_characters']))
                        # st.metric("Profiler Type", __print_metric(table_stats['profiler_type']))
                        tmp = 'Neutral'
                        if table_stats['sentiment'] < -0.5:
                            tmp = 'Strongly Negative'
                        elif -0.05 > table_stats['sentiment'] > -0.5:
                            tmp = 'Negative'
                        elif 0.5 > table_stats['sentiment'] > 0.05:
                            tmp = 'Positive'
                        elif 1 >= table_stats['sentiment'] > 0.5:
                            tmp = 'Strongly Positive'

                        st.metric(__change_font("Emotional tone"), __print_metric(tmp))
                        # if sent_analysis:
                        #     st.metric("Compound Mean",
                        #               __print_metric(table_stats['sentiment_analysis']['compound_mean']))
                        st.metric(__change_font("Ratio Uppercase (\%)"),
                                  __print_metric(table_stats['ratio_uppercase'], percentage=True))
                        st.metric(__change_font("Ratio Digits (\%)"),
                                  __print_metric(table_stats['ratio_digits'], percentage=True))
                        st.metric(__change_font("Ratio Special Characters (\%)"),
                                  __print_metric(table_stats['ratio_special_characters'], percentage=True))

                    with plot:
                        languages = {language['language']: language['percentage']
                                     for language in table_stats['language_distribution']}
                        languages = dict(
                            sorted(languages.items(), key=lambda item: item[1], reverse=True))
                        languages = __reduceCategoricalDict(languages, 10)
                        languages_df = pd.DataFrame(languages.items(), columns=['Language', 'Percentage'])
                        languages_df = languages_df[languages_df.Percentage != 0]
                        languages_df['Language'] = languages_df['Language'].apply(__fix_length)

                        fig = px.pie(languages_df, values='Percentage', names='Language')
                        fig.update_traces(textposition='inside', textfont_size=14)
                        fig.update_layout(title_text='Corpus Language Distribution', title_x=0.5,
                                          title_y=1, title_xanchor='center',
                                          title_yanchor='top', legend_font_size=14,
                                          title_font_size=18)
                        st.plotly_chart(fig, use_container_width=True)

                        if sent_analysis:
                            compound_levels = table_stats['sentiment_analysis']['compound_levels']
                            compound_levels = dict(
                                sorted(compound_levels.items(), key=lambda item: item[1], reverse=True))
                            compound_levels_df = pd.DataFrame(compound_levels.items(),
                                                              columns=['Compound Levels', 'Number of Texts'])

                            compound_levels_df['Sentiment'] = np.select(
                                [compound_levels_df['Compound Levels'] == '(-1, -0.5)',
                                 compound_levels_df['Compound Levels'] == '(-0.5, 0)',
                                 compound_levels_df['Compound Levels'] == '(0, 0.5)',
                                 compound_levels_df['Compound Levels'] == '(0.5, 1)'],
                                ['Strongly Negative', 'Negative', 'Positive',
                                 'Strongly Positive'],
                                default='Neutral')

                            compound_levels_df.drop('Compound Levels', axis=1, inplace=True)

                            compound_levels_df = compound_levels_df[compound_levels_df.loc[:, 'Number of Texts'] != 0]
                            compound_levels_df['Sentiment'] = compound_levels_df['Sentiment'].apply(__fix_length)

                            fig = px.pie(compound_levels_df, values='Number of Texts', names='Sentiment')
                            fig.update_traces(textposition='inside', textfont_size=14)
                            fig.update_layout(title_text='Sentiment Distribution', title_x=0.5,
                                              title_y=1, title_xanchor='center',
                                              title_yanchor='top', legend_font_size=14,
                                              title_font_size=18)
                            st.plotly_chart(fig, use_container_width=True)
                    st.divider()
                    st.header('Term Frequency Distribution')

                    term_freq_df = pd.DataFrame(table_stats['term_frequency'], columns=['term', 'count'])
                    term_freq_df.columns = ['Term', 'Count']
                    term_freq_df['Frequency'] = (term_freq_df['Count'] / term_freq_df['Count'].sum())
                    term_freq_df['Frequency'] = term_freq_df['Frequency'].map('{:.5%}'.format)
                    st.dataframe(term_freq_df, use_container_width=True, hide_index=True)

                with texts:
                    variable_stats = config_dict['variables']
                    for variable in variable_stats:
                        with st.expander(variable['name'] + ' ( Text ) '):
                            corpus_stats, plot = st.columns(2)

                            with corpus_stats:
                                corpus_stats1, corpus_stats2 = st.columns(2)
                                with corpus_stats1:
                                    st.metric(__change_font("Number of Words"), __print_metric(variable['num_words']))
                                    st.metric(__change_font("Number of Distinct Words"),
                                              __print_metric(variable['num_distinct_words']))
                                    st.metric(__change_font("Number of Sentences"),
                                              __print_metric(variable['num_sentences']))
                                    st.metric(__change_font("Number of Characters"),
                                              __print_metric(variable['num_characters']))
                                with corpus_stats2:
                                    tmp = 'Neutral'
                                    if variable['sentiment'] < -0.5:
                                        tmp = 'Strongly Negative'
                                    elif -0.05 > variable['sentiment'] > -0.5:
                                        tmp = 'Negative'
                                    elif 0.5 > variable['sentiment'] > 0.05:
                                        tmp = 'Positive'
                                    elif 1 >= variable['sentiment'] > 0.5:
                                        tmp = 'Strongly Positive'

                                    st.metric(__change_font("Emotional tone"),
                                              __print_metric(tmp))
                                    # st.metric("Compound Mean", __print_metric(variable['sentiment']))
                                    st.metric(__change_font("Ratio Uppercase (\%)"),
                                              __print_metric(variable['ratio_uppercase'], percentage=True))
                                    st.metric(__change_font("Ratio Digits (\%)"),
                                              __print_metric(variable['ratio_digits'], percentage=True))
                                    st.metric(__change_font("Ratio Special Characters (\%)"),
                                              __print_metric(variable['ratio_special_characters'], percentage=True))
                            with plot:
                                languages = {language['language']: language['percentage']
                                             for language in variable['language_distribution']}
                                languages = dict(
                                    sorted(languages.items(), key=lambda item: item[1], reverse=True))
                                languages = __reduceCategoricalDict(languages, 10)
                                languages_df = pd.DataFrame(languages.items(), columns=['Language', 'Percentage'])
                                languages_df = languages_df[languages_df.Percentage != 0]
                                languages_df['Language'] = languages_df['Language'].apply(__fix_length)

                                fig = px.pie(languages_df, values='Percentage', names='Language')
                                fig.update_traces(textposition='inside', textfont_size=14)
                                fig.update_layout(title_text='Corpus Language Distribution', title_x=0.5,
                                                  title_y=1, title_xanchor='center',
                                                  title_yanchor='top', legend_font_size=14,
                                                  title_font_size=18)
                                st.plotly_chart(fig, use_container_width=True)

                            tab1, tab2, tab3, tab4, tab5 = st.tabs(["Summary", "Topics", "Term Frequency",
                                                                    "Special Characters Frequency", "Distributions"])

                            with tab1:
                                st.write(__print_metric(variable['summary']))

                            with tab2:
                                for topic in variable['topics']:
                                    st.write(__print_metric(topic))

                            with tab3:
                                term_freq_df = pd.DataFrame(variable['term_frequency'], columns=['term', 'count'])
                                term_freq_df.columns = ['Term', 'Count']
                                term_freq_df['Frequency'] = (term_freq_df['Count'] / term_freq_df['Count'].sum())
                                term_freq_df['Frequency'] = term_freq_df['Frequency'].map('{:.5%}'.format)
                                st.dataframe(term_freq_df, use_container_width=True, hide_index=True)

                            with tab4:
                                term_freq_df = pd.DataFrame(variable['special_characters_distribution'],
                                                            columns=['type', 'count'])
                                term_freq_df.columns = ['Special Character', 'Count']
                                term_freq_df['Special Character'].replace(' ', 'Whitespace', inplace=True)
                                term_freq_df.sort_values(by=['Count'], ascending=False, inplace=True)
                                term_freq_df['Frequency'] = (term_freq_df['Count'] / variable['num_characters'])
                                term_freq_df['Frequency'] = term_freq_df['Frequency'].map('{:.5%}'.format)
                                st.dataframe(term_freq_df, use_container_width=True, hide_index=True)

                            with tab5:
                                words, sentences = st.columns(2)

                                with words:
                                    tmp = variable['word_length_distribution']
                                    st.subheader("Words Length")

                                    words_dict = {'Mean': __print_metric(tmp['average']),
                                                  'Standard Deviation': __print_metric(tmp['stddev']),
                                                  'Kurtosis': __print_metric(tmp['kurtosis']),
                                                  'Skewness': __print_metric(tmp['skewness']),
                                                  'Variance': __print_metric(tmp['variance'])}

                                    words_df = pd.DataFrame(words_dict.items(),
                                                            columns=['Statistics', 'Values'])
                                    st.dataframe(words_df, use_container_width=True, hide_index=True)

                                    # st.metric(__change_font("Mean"), __print_metric(tmp['average']))
                                    # st.metric(__change_font("Standard Deviation"), __print_metric(tmp['stddev']))
                                    # st.metric(__change_font("Kurtosis"), __print_metric(tmp['kurtosis']))
                                    # st.metric(__change_font("Skewness"), __print_metric(tmp['skewness']))
                                    # st.metric(__change_font("Variance"), __print_metric(tmp['variance']))
                                    fig = go.Figure()
                                    fig.add_trace(go.Box(
                                        q1=[tmp['percentile25']],
                                        median=[tmp['median']],
                                        mean=[tmp['average']],
                                        q3=[tmp['percentile75']],
                                        lowerfence=[tmp['min']],
                                        upperfence=[tmp['max']],
                                        boxmean='sd',
                                        boxpoints=False,
                                        sd=[tmp['stddev']],
                                        showlegend=False,
                                        x0=tmp['name']
                                    ))

                                    fig.update_layout(title_text="Values Distribution", title_x=0.5,
                                                      title_y=0.9, title_xanchor='center',
                                                      title_yanchor='top',
                                                      title_font_size=18)

                                    fig.update_xaxes(
                                        title_font_size=16,
                                        tickfont_size=16
                                    )

                                    fig.update_yaxes(
                                        tickfont_size=16
                                    )

                                    st.plotly_chart(fig, use_container_width=True)

                                with sentences:
                                    tmp = variable['sentence_length_distribution']
                                    st.subheader("Sentences Length")

                                    sentences_dict = {'Mean': __print_metric(tmp['average']),
                                                      'Standard Deviation': __print_metric(tmp['stddev']),
                                                      'Kurtosis': __print_metric(tmp['kurtosis']),
                                                      'Skewness': __print_metric(tmp['skewness']),
                                                      'Variance': __print_metric(tmp['variance'])}

                                    sentences_df = pd.DataFrame(sentences_dict.items(),
                                                                columns=['Statistics', 'Values'])
                                    st.dataframe(sentences_df, use_container_width=True, hide_index=True)

                                    # st.metric(__change_font("Mean"), __print_metric(tmp['average']))
                                    # st.metric(__change_font("Standard Deviation"), __print_metric(tmp['stddev']))
                                    # st.metric(__change_font("Kurtosis"), __print_metric(tmp['kurtosis']))
                                    # st.metric(__change_font("Skewness"), __print_metric(tmp['skewness']))
                                    # st.metric(__change_font("Variance"), __print_metric(tmp['variance']))
                                    fig = go.Figure()
                                    fig.add_trace(go.Box(
                                        q1=[tmp['percentile25']],
                                        median=[tmp['median']],
                                        mean=[tmp['average']],
                                        q3=[tmp['percentile75']],
                                        lowerfence=[tmp['min']],
                                        upperfence=[tmp['max']],
                                        boxmean='sd',
                                        boxpoints=False,
                                        sd=[tmp['stddev']],
                                        showlegend=False,
                                        x0=tmp['name']
                                    ))

                                    fig.update_layout(title_text="Values Distribution", title_x=0.5,
                                                      title_y=0.9, title_xanchor='center',
                                                      title_yanchor='top',
                                                      title_font_size=18)

                                    fig.update_xaxes(
                                        title_font_size=16,
                                        tickfont_size=16
                                    )

                                    fig.update_yaxes(
                                        tickfont_size=16
                                    )

                                    st.plotly_chart(fig, use_container_width=True)
            elif config_dict['table']['profiler_type'] in ['Raster', 'Vista_Raster']:
                overview, variables = st.tabs(["Overview", "Images"])

                with overview:
                    table_stats = config_dict['table']
                    if config_dict['table']['profiler_type'] == 'Vista_Raster':
                        st.subheader("RAS file Statistics")
                    else:
                        if table_stats['n_of_imgs'] == 1:
                            st.subheader("Image Statistics")
                        else:
                            st.subheader("Multiple Image Statistics")
                    st.metric(__change_font("Number of images"), __print_metric(table_stats['n_of_imgs']))
                    st.metric(__change_font("Average image width"), __print_metric(table_stats['avg_width']))
                    st.metric(__change_font("Average image height"), __print_metric(table_stats['avg_height']))
                    # st.metric("Profiler Type", __print_metric(table_stats['profiler_type']))
                    st.metric(__change_font("Total file size"),
                              __print_metric(table_stats['byte_size'], memory=True))

                    # combined_bands only for Vista
                    if config_dict['table']['profiler_type'] == 'Vista_Raster':
                        st.divider()
                        if 'combined_bands' in table_stats:
                            combined_bands = table_stats['combined_bands']

                            band_tabs = []
                            for band in combined_bands:
                                band_tabs.append(band['name'])

                            if len(band_tabs) == 1:
                                band = combined_bands[0]
                                st.header("Combined " + __print_metric(band['name']) + " band")
                                stats, plot = st.columns(2)
                                data_df = pd.DataFrame()
                                enter_imgs = False
                                with stats:
                                    with st.container():
                                        # st.subheader("Statistics")
                                        cols = st.columns(2)

                                        comb_band_dict = {
                                            'Number of images with this band': __print_metric(band['n_of_imgs']),
                                            'Minimum': __print_metric(band['min']),
                                            'Mean': __print_metric(band['average']),
                                            'Maximum': __print_metric(band['max'])
                                        }

                                        comb_band_df = pd.DataFrame(comb_band_dict.items(),
                                                                    columns=['Statistics', 'Values'])
                                        cols[0].dataframe(comb_band_df, use_container_width=True, hide_index=True)

                                        # cols[0].metric(__change_font("Number of rasters with this band"),
                                        #                __print_metric(band['n_of_imgs']))
                                        # cols[0].metric("Count",
                                        #                __print_metric(band['count']))
                                        # cols[0].metric(__change_font("Minimum"),
                                        #                __print_metric(band['min']))
                                        # cols[0].metric(__change_font("Mean"),
                                        #                __print_metric(band['average']))
                                        # cols[0].metric(__change_font("Maximum"),
                                        #                __print_metric(band['max']))
                                        # cols[0].metric(__change_font("Variance"),
                                        #                __print_metric(band['variance']))

                                        if 'imgs' in band:
                                            enter_imgs = True
                                            data_df = pd.DataFrame(band['imgs'])

                                            cols[1].data_editor(
                                                data_df.loc[:, ['raster', 'date']],
                                                column_config={
                                                    "raster": st.column_config.ListColumn(
                                                        "Images",
                                                        help="The names of the images that have this particular band in their bands",
                                                        width="medium",
                                                    ),
                                                    "date": st.column_config.ListColumn(
                                                        "Dates",
                                                        help="The dates of the images that have this particular band in their bands",
                                                        width="medium",
                                                    )
                                                },
                                                hide_index=True,
                                                key=band['name']
                                            )
                                        else:
                                            data_df = pd.DataFrame(
                                                {
                                                    "rasters": band['img_names'],
                                                }
                                            )

                                            cols[1].data_editor(
                                                data_df,
                                                column_config={
                                                    "rasters": st.column_config.ListColumn(
                                                        "Images",
                                                        help="The names of the images that have this particular band in their bands",
                                                        width="medium",
                                                    ),
                                                },
                                                hide_index=True,
                                                key=band['name']
                                            )

                                if 'no_data_distribution' in band:
                                    with plot:
                                        # st.subheader("Pixel Type Distribution")

                                        no_data_dict = {no_data['value']: no_data['percentage']
                                                        for no_data in band['no_data_distribution']}
                                        no_data_dict = dict(
                                            sorted(no_data_dict.items(), key=lambda item: item[1], reverse=True))

                                        no_data_df = pd.DataFrame(no_data_dict.items(),
                                                                  columns=['Value', 'Percentage'])
                                        no_data_df = no_data_df[no_data_df.Percentage != 0]
                                        no_data_df['Value'] = no_data_df['Value'].apply(__fix_length)

                                        fig = px.pie(no_data_df, values='Percentage', names='Value')
                                        fig.update_traces(textposition='inside', textfont_size=14)
                                        fig.update_layout(title_text='Pixel Type Distribution', title_x=0.5,
                                                          title_y=1, title_xanchor='center',
                                                          title_yanchor='top', legend_font_size=14,
                                                          title_font_size=18)
                                        st.plotly_chart(fig, use_container_width=True)

                                if enter_imgs:
                                    st.divider()
                                    st.subheader("Image-LAI Percentage Histogram")
                                    data_df.columns = ['Image', 'TimeStamp', 'LAI Percentage']
                                    bar_freq = alt.Chart(data_df).mark_bar().encode(
                                        x=alt.X('TimeStamp:O', title='TimeStamp', sort=None),
                                        y=alt.Y('LAI Percentage', title='LAI Percentage', sort=None),
                                        tooltip=['Image', 'TimeStamp', 'LAI Percentage']
                                    ).configure_axis(
                                        labelFontSize=16,
                                        titleFontSize=16
                                    ).interactive()

                                    st.altair_chart(bar_freq, use_container_width=True)

                                if 'lai_distribution' in band:
                                    st.divider()
                                    st.subheader("Image-LAI Percentage Distribution")
                                    cols = st.columns(2)
                                    tmp = band['lai_distribution']
                                    lai_dict = {'Mean': __print_metric(tmp['average']),
                                                'Standard Deviation': __print_metric(tmp['stddev']),
                                                'Kurtosis': __print_metric(tmp['kurtosis']),
                                                'Skewness': __print_metric(tmp['skewness']),
                                                'Variance': __print_metric(tmp['variance'])}

                                    lai_df = pd.DataFrame(lai_dict.items(), columns=['Statistics', 'Values'])
                                    cols[0].dataframe(lai_df, use_container_width=True, hide_index=True)

                                    # cols[0].metric(__change_font("Mean"), __print_metric(tmp['average']))
                                    # cols[0].metric(__change_font("Standard Deviation"), __print_metric(tmp['stddev']))
                                    # cols[0].metric(__change_font("Kurtosis"), __print_metric(tmp['kurtosis']))
                                    # cols[0].metric(__change_font("Skewness"), __print_metric(tmp['skewness']))
                                    # cols[0].metric(__change_font("Variance"), __print_metric(tmp['variance']))
                                    fig = go.Figure()
                                    fig.add_trace(go.Box(
                                        q1=[tmp['percentile25']],
                                        median=[tmp['median']],
                                        mean=[tmp['average']],
                                        q3=[tmp['percentile75']],
                                        lowerfence=[tmp['min']],
                                        upperfence=[tmp['max']],
                                        boxmean='sd',
                                        boxpoints=False,
                                        sd=[tmp['stddev']],
                                        showlegend=False,
                                        x0=tmp['name']
                                    ))

                                    fig.update_layout(title_text="LAI Values Distribution", title_x=0.5,
                                                      title_y=1, title_xanchor='center',
                                                      title_yanchor='top',
                                                      title_font_size=18)

                                    fig.update_xaxes(
                                        title_font_size=16,
                                        tickfont_size=16
                                    )

                                    fig.update_yaxes(
                                        tickfont_size=16
                                    )

                                    cols[1].plotly_chart(fig, use_container_width=True)
                            else:
                                st.header("Combined bands")
                                tabs = st.tabs(band_tabs)
                                count = 0
                                for band in combined_bands:
                                    with tabs[count]:
                                        stats, plot = st.columns(2)
                                        with stats:
                                            with st.container():
                                                st.subheader("Statistics")

                                                cols = st.columns(2)
                                                comb_band_dict = {'Number of images with this band': __print_metric(
                                                    band['n_of_imgs']),
                                                    'Minimum': __print_metric(band['min']),
                                                    'Mean': __print_metric(band['average']),
                                                    'Maximum': __print_metric(band['max'])
                                                }

                                                comb_band_df = pd.DataFrame(comb_band_dict.items(),
                                                                            columns=['Statistics', 'Values'])
                                                cols[0].dataframe(comb_band_df, use_container_width=True,
                                                                  hide_index=True)

                                                # cols[0].metric(__change_font("Number of rasters with this band"),
                                                #                __print_metric(band['n_of_imgs']))
                                                # cols[0].metric("Count",
                                                #                __print_metric(band['count']))
                                                # cols[0].metric(__change_font("Minimum"),
                                                #                __print_metric(band['min']))
                                                # cols[0].metric(__change_font("Mean"),
                                                #                __print_metric(band['average']))
                                                # cols[0].metric(__change_font("Maximum"),
                                                #                __print_metric(band['max']))
                                                # cols[0].metric(__change_font("Variance"),
                                                #                __print_metric(band['variance']))

                                                data_df = pd.DataFrame(
                                                    {
                                                        "rasters": band['img_names'],
                                                    }
                                                )

                                                if 'imgs' in band:
                                                    enter_imgs = True
                                                    data_df = pd.DataFrame(band['imgs'])

                                                    cols[1].data_editor(
                                                        data_df.loc[:, ['raster', 'date']],
                                                        column_config={
                                                            "raster": st.column_config.ListColumn(
                                                                "Images",
                                                                help="The names of the images that have this particular band in their bands",
                                                                width="medium",
                                                            ),
                                                            "date": st.column_config.ListColumn(
                                                                "Dates",
                                                                help="The dates of the images that have this particular band in their bands",
                                                                width="medium",
                                                            )
                                                        },
                                                        hide_index=True,
                                                        key=band['name']
                                                    )
                                                else:
                                                    data_df = pd.DataFrame(
                                                        {
                                                            "rasters": band['img_names'],
                                                        }
                                                    )

                                                    cols[1].data_editor(
                                                        data_df,
                                                        column_config={
                                                            "rasters": st.column_config.ListColumn(
                                                                "Images",
                                                                help="The names of the images that have this particular band in their bands",
                                                                width="medium",
                                                            ),
                                                        },
                                                        hide_index=True,
                                                        key=band['name']
                                                    )

                                        if 'no_data_distribution' in band:
                                            with plot:
                                                # st.subheader("Pixel Type Distribution")

                                                no_data_dict = {no_data['value']: no_data['percentage']
                                                                for no_data in band['no_data_distribution']}
                                                no_data_dict = dict(
                                                    sorted(no_data_dict.items(), key=lambda item: item[1],
                                                           reverse=True))

                                                no_data_df = pd.DataFrame(no_data_dict.items(),
                                                                          columns=['Value', 'Percentage'])
                                                no_data_df = no_data_df[no_data_df.Percentage != 0]
                                                no_data_df['Value'] = no_data_df['Value'].apply(__fix_length)

                                                fig = px.pie(no_data_df, values='Percentage', names='Value')
                                                fig.update_traces(textposition='inside', textfont_size=14)
                                                fig.update_layout(title_text='Pixel Type Distribution', title_x=0.5,
                                                                  title_y=1, title_xanchor='center',
                                                                  title_yanchor='top', legend_font_size=14,
                                                                  title_font_size=18)
                                                st.plotly_chart(fig, use_container_width=True)

                                        if enter_imgs:
                                            st.divider()
                                            st.subheader("Image-LAI Percentage Histogram")
                                            data_df.columns = ['Image', 'TimeStamp', 'LAI Percentage']
                                            bar_freq = alt.Chart(data_df).mark_bar().encode(
                                                x=alt.X('TimeStamp:O', title='TimeStamp', sort=None),
                                                y=alt.Y('LAI Percentage', title='LAI Percentage', sort=None),
                                                tooltip=['Image', 'TimeStamp', 'LAI Percentage']
                                            ).configure_axis(
                                                labelFontSize=16,
                                                titleFontSize=16
                                            ).interactive()

                                            st.altair_chart(bar_freq, use_container_width=True)

                                        if 'lai_distribution' in band:
                                            st.divider()
                                            st.subheader("Image-LAI Percentage Distribution")
                                            cols = st.columns(2)
                                            tmp = band['lai_distribution']
                                            lai_dict = {'Mean': __print_metric(tmp['average']),
                                                        'Standard Deviation': __print_metric(tmp['stddev']),
                                                        'Kurtosis': __print_metric(tmp['kurtosis']),
                                                        'Skewness': __print_metric(tmp['skewness']),
                                                        'Variance': __print_metric(tmp['variance'])}

                                            lai_df = pd.DataFrame(lai_dict.items(), columns=['Statistics', 'Values'])
                                            cols[0].dataframe(lai_df, use_container_width=True, hide_index=True)

                                            # cols[0].metric(__change_font("Mean"), __print_metric(tmp['average']))
                                            # cols[0].metric(__change_font("Standard Deviation"),
                                            #                __print_metric(tmp['stddev']))
                                            # cols[0].metric(__change_font("Kurtosis"), __print_metric(tmp['kurtosis']))
                                            # cols[0].metric(__change_font("Skewness"), __print_metric(tmp['skewness']))
                                            # cols[0].metric(__change_font("Variance"), __print_metric(tmp['variance']))
                                            fig = go.Figure()
                                            fig.add_trace(go.Box(
                                                q1=[tmp['percentile25']],
                                                median=[tmp['median']],
                                                mean=[tmp['average']],
                                                q3=[tmp['percentile75']],
                                                lowerfence=[tmp['min']],
                                                upperfence=[tmp['max']],
                                                boxmean='sd',
                                                boxpoints=False,
                                                sd=[tmp['stddev']],
                                                showlegend=False,
                                                x0=tmp['name']
                                            ))

                                            fig.update_layout(title_text="LAI Values Distribution", title_x=0.5,
                                                              title_y=1, title_xanchor='center',
                                                              title_yanchor='top',
                                                              title_font_size=18)

                                            fig.update_xaxes(
                                                title_font_size=16,
                                                tickfont_size=16
                                            )

                                            fig.update_yaxes(
                                                tickfont_size=16
                                            )

                                            cols[1].plotly_chart(fig, use_container_width=True)
                                        count += 1

                with variables:
                    variable_stats = config_dict['variables']
                    for variable in variable_stats:
                        with st.expander(variable['name'] + ' ( Image ) '):
                            with st.container():
                                cols = st.columns(2)
                                if 'date' in variable:
                                    cols[0].metric(__change_font("date"), __print_metric(variable['date']))
                                cols[0].metric(__change_font("format"), __print_metric(variable['format']))
                                cols[0].metric(__change_font("dType"), __print_metric(variable['dtype']))
                                cols[0].metric(__change_font("width"), __print_metric(variable['width']))
                                cols[0].metric(__change_font("height"), __print_metric(variable['height']))
                                cols[0].metric(__change_font("Number of bands"), __print_metric(variable['count']))
                                cols[0].metric(__change_font("CRS"), __print_metric(variable['crs']))
                                if 'no_data_value' in variable:
                                    if variable['no_data_value'] not in ['', 'None']:
                                        cols[0].metric(__change_font("NODATA value"),
                                                       __print_metric(variable['no_data_value']))
                                # cols[0].subheader('Spatial Resolution')
                                # cols[0].metric("Pixel size of x",
                                #                __print_metric(variable['spatial_resolution']['pixel_size_x']))
                                # cols[0].metric("Pixel size of y",
                                #                __print_metric(variable['spatial_resolution']['pixel_size_y']))

                                transform = variable['transform']
                                affine_transform = rio.Affine(transform[0], transform[1], transform[2],
                                                              transform[3], transform[4], transform[5],
                                                              transform[6], transform[7], transform[8])
                                cols[0].subheader('Affine Transform')
                                cols[0].write(affine_transform)

                                cols[1].subheader('Spatial Coverage')
                                mbr_box = gpd.GeoSeries.from_wkt([variable['spatial_coverage']])
                                gjson = folium.GeoJson(data=mbr_box.to_json(),
                                                       style_function=lambda x: {"fillColor": "orange"})

                                centroid = mbr_box.centroid
                                lon = centroid.x
                                lat = centroid.y
                                m = folium.Map(location=[lat, lon], tiles="CartoDB positron")
                                gjson.add_to(m)
                                with cols[1]:
                                    m.fit_bounds(gjson.get_bounds())
                                    st_folium(m, use_container_width=True, key=variable['name'])

                            st.divider()

                            bands = variable['bands']

                            band_tabs = []
                            for band in bands:
                                band_tabs.append(band['name'])

                            if len(band_tabs) == 1:
                                band = bands[0]
                                st.header(__print_metric(band['name']) + " band")
                                with st.container():
                                    stats, plot = st.columns(2)
                                    if 'no_data_distribution' in band:
                                        with plot:
                                            # st.subheader("Pixel Type Distribution")

                                            no_data_dict = {no_data['value']: no_data['percentage']
                                                            for no_data in band['no_data_distribution']}
                                            no_data_dict = dict(
                                                sorted(no_data_dict.items(), key=lambda item: item[1], reverse=True))

                                            no_data_df = pd.DataFrame(no_data_dict.items(),
                                                                      columns=['Value', 'Percentage'])
                                            no_data_df = no_data_df[no_data_df.Percentage != 0]
                                            no_data_df['Value'] = no_data_df['Value'].apply(__fix_length)
                                            fig = px.pie(no_data_df, values='Percentage', names='Value')
                                            fig.update_traces(textposition='inside', textfont_size=14)
                                            fig.update_layout(title_text='Pixel Type Distribution', title_x=0.5,
                                                              title_y=1, title_xanchor='center',
                                                              title_yanchor='top', legend_font_size=14,
                                                              title_font_size=18)
                                            st.plotly_chart(fig, use_container_width=True)

                                    with stats:
                                        if band['count'] == 0:
                                            if config_dict['table']['profiler_type'] == 'Vista_Raster':
                                                st.subheader('There are no LAI values!')
                                            else:
                                                st.subheader('There are no values!')
                                        else:
                                            # st.subheader("Statistics")
                                            lai_dict = {'Mean': __print_metric(band['average']),
                                                        'Standard Deviation': __print_metric(band['stddev']),
                                                        'Kurtosis': __print_metric(band['kurtosis']),
                                                        'Skewness': __print_metric(band['skewness']),
                                                        'Variance': __print_metric(band['variance'])}

                                            lai_df = pd.DataFrame(lai_dict.items(), columns=['Statistics', 'Values'])
                                            st.dataframe(lai_df, use_container_width=True, hide_index=True)

                                            if band['name'] == 'undefined':
                                                # st.subheader('uuid')
                                                # st.write(__print_metric(band['uuid']))
                                                lai_dict['uuid'] = __print_metric(band['uuid'])
                                            # st.metric(__change_font("Mean"), __print_metric(band['average']))
                                            # st.metric(__change_font("Standard Deviation"),
                                            #           __print_metric(band['stddev']))
                                            # st.metric(__change_font("Kurtosis"), __print_metric(band['kurtosis']))
                                            # st.metric(__change_font("Skewness"), __print_metric(band['skewness']))
                                            # st.metric(__change_font("Variance"), __print_metric(band['variance']))
                                            fig = go.Figure()
                                            fig.add_trace(go.Box(
                                                q1=[band['percentile25']],
                                                median=[band['median']],
                                                mean=[band['average']],
                                                q3=[band['percentile75']],
                                                lowerfence=[band['min']],
                                                upperfence=[band['max']],
                                                boxmean='sd',
                                                boxpoints=False,
                                                sd=[band['stddev']],
                                                showlegend=False,
                                                x0=band['name']
                                            ))
                                            if config_dict['table']['profiler_type'] == 'Vista_Raster':
                                                fig.update_layout(title_text="LAI Values Distribution", title_x=0.5,
                                                                  title_y=0.9, title_xanchor='center',
                                                                  title_yanchor='top',
                                                                  title_font_size=18)

                                                fig.update_xaxes(
                                                    title_font_size=16,
                                                    tickfont_size=16
                                                )

                                                fig.update_yaxes(
                                                    tickfont_size=16
                                                )
                                            else:
                                                fig.update_layout(title_text="Values Distribution", title_x=0.5,
                                                                  title_y=0.9, title_xanchor='center',
                                                                  title_yanchor='top',
                                                                  title_font_size=18)

                                                fig.update_xaxes(
                                                    title_font_size=16,
                                                    tickfont_size=16
                                                )

                                                fig.update_yaxes(
                                                    tickfont_size=16
                                                )
                                            st.plotly_chart(fig, use_container_width=True)

                            else:
                                st.subheader("Bands")
                                tabs = st.tabs(band_tabs)
                                count = 0
                                for band in bands:
                                    with tabs[count]:
                                        with st.container():
                                            stats, plot = st.columns(2)
                                            if 'no_data_distribution' in band:
                                                with plot:
                                                    no_data_dict = {no_data['value']: no_data['percentage']
                                                                    for no_data in band['no_data_distribution']}
                                                    no_data_dict = dict(
                                                        sorted(no_data_dict.items(), key=lambda item: item[1],
                                                               reverse=True))

                                                    no_data_df = pd.DataFrame(no_data_dict.items(),
                                                                              columns=['Value', 'Percentage'])
                                                    no_data_df = no_data_df[no_data_df.Percentage != 0]
                                                    no_data_df['Value'] = no_data_df['Value'].apply(__fix_length)

                                                    fig = px.pie(no_data_df, values='Percentage', names='Value')
                                                    fig.update_traces(textposition='inside', textfont_size=14)
                                                    fig.update_layout(title_text='Pixel Type Distribution', title_x=0.5,
                                                                      title_y=1, title_xanchor='center',
                                                                      title_yanchor='top', legend_font_size=14,
                                                                      title_font_size=18)
                                                    st.plotly_chart(fig, use_container_width=True)

                                            with stats:
                                                if band['count'] == 0:
                                                    if config_dict['table']['profiler_type'] == 'Vista_Raster':
                                                        st.subheader('There are no LAI values!')
                                                    else:
                                                        st.subheader('There are no values!')
                                                else:
                                                    # st.subheader("Statistics")
                                                    lai_dict = {'Mean': __print_metric(band['average']),
                                                                'Standard Deviation': __print_metric(band['stddev']),
                                                                'Kurtosis': __print_metric(band['kurtosis']),
                                                                'Skewness': __print_metric(band['skewness']),
                                                                'Variance': __print_metric(band['variance'])}

                                                    lai_df = pd.DataFrame(lai_dict.items(),
                                                                          columns=['Statistics', 'Values'])
                                                    st.dataframe(lai_df, use_container_width=True, hide_index=True)

                                                    if band['name'] == 'undefined':
                                                        # st.subheader('uuid')
                                                        # st.write(__print_metric(band['uuid']))
                                                        lai_dict['uuid'] = __print_metric(band['uuid'])

                                                    # st.metric(__change_font("Mean"), __print_metric(band['average']))
                                                    # st.metric(__change_font("Standard Deviation"),
                                                    #           __print_metric(band['stddev']))
                                                    # st.metric(__change_font("Kurtosis"),
                                                    #           __print_metric(band['kurtosis']))
                                                    # st.metric(__change_font("Skewness"),
                                                    #           __print_metric(band['skewness']))
                                                    # st.metric(__change_font("Variance"),
                                                    #           __print_metric(band['variance']))
                                                    fig = go.Figure()
                                                    fig.add_trace(go.Box(
                                                        q1=[band['percentile25']],
                                                        median=[band['median']],
                                                        mean=[band['average']],
                                                        q3=[band['percentile75']],
                                                        lowerfence=[band['min']],
                                                        upperfence=[band['max']],
                                                        boxmean='sd',
                                                        boxpoints=False,
                                                        sd=[band['stddev']],
                                                        showlegend=False,
                                                        x0=band['name']
                                                    ))

                                                    if config_dict['table']['profiler_type'] == 'Vista_Raster':
                                                        fig.update_layout(title_text="LAI Values Distribution",
                                                                          title_x=0.5,
                                                                          title_y=0.9, title_xanchor='center',
                                                                          title_yanchor='top',
                                                                          title_font_size=18)

                                                        fig.update_xaxes(
                                                            title_font_size=16,
                                                            tickfont_size=16
                                                        )

                                                        fig.update_yaxes(
                                                            tickfont_size=16
                                                        )
                                                    else:
                                                        fig.update_layout(title_text="Values Distribution", title_x=0.5,
                                                                          title_y=0.9, title_xanchor='center',
                                                                          title_yanchor='top',
                                                                          title_font_size=18)

                                                        fig.update_xaxes(
                                                            title_font_size=16,
                                                            tickfont_size=16
                                                        )

                                                        fig.update_yaxes(
                                                            tickfont_size=16
                                                        )

                                                    st.plotly_chart(fig, use_container_width=True)

                                            count += 1
            else:
                st.write('Visualize Profile')
        elif submitted and profile_results_file is None and path_to_profile_json_input == '':
            st.write("No configuration file was uploaded!")
if selected == "Compare Profiles":
    with st.form("my_form"):
        json1, json2 = st.columns(2)

        with json1:
            profile_results_file1 = st.file_uploader("Choose the first JSON file",
                                                     help='The chosen JSON file must have been produced by the Profiler')

        with json2:
            profile_results_file2 = st.file_uploader("Choose the second JSON file",
                                                     help='The chosen JSON file must have been produced by the Profiler')

        submitted = st.form_submit_button("Compare & Visualize")

        if submitted and profile_results_file1 is not None and profile_results_file2 is not None:
            st.write('Compare Profiling Results Between Datasets')
