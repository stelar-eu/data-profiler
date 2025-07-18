import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import json
import streamlit as st
import numpy as np
import pandas as pd
import geopandas as gpd
import altair as alt
import plotly.graph_objects as go
from datetime import datetime
from itertools import islice
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from matplotlib.collections import PolyCollection
from statsmodels.tsa.stattools import acf
from rasterio import Affine
from typing import Union
import plotly.express as px
from collections import Counter


def read_json_profile(json_file: str) -> dict:
    """
    This method reads a json file and deserializes it into a dictionary.

    :param json_file: path to .json file.
    :type json_file: str
    :return: A dictionary.
    :rtype: dict

    """
    try:
        json_dict: dict = json.loads(json_file)
    except ValueError as e:
        with open(json_file) as f:
            json_dict: dict = json.load(f)
            return json_dict

    return json_dict


def __transform_list_of_dicts(list_of_dicts, chosen_key):
    result = {}
    for d in list_of_dicts:
        if chosen_key in d:
            # Create a new dictionary excluding the chosen key
            new_dict = {k: v for k, v in d.items()}
            # Use the value of the chosen key as the new key in the result
            result[d[chosen_key]] = new_dict
    return result


def __recreate_plots(corr_data, ax1=None, ax2=None):
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


def __float_to_datetime(fl):
    return datetime.fromtimestamp(fl)


def _get_ts_lag(series: pd.Series) -> int:
    lag = 100
    max_lag_size = (len(series) // 2) - 1
    return np.min([lag, max_lag_size])


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
    new_txt = '${\\sf '
    # new_txt = '${\\sf \\Large '
    split_text = text.split(' ')
    for txt in split_text:
        new_txt += txt + ' \\ '

    new_txt += '}$'

    return new_txt


def __print_metric(value, percentage: bool = False, memory: bool = False, record: bool = False):  # , label: str
    if isinstance(value, str):
        return f"{value}"
    else:
        if __is_integer_num(value) and not percentage and not memory:
            return f"{int(value)}"
        else:
            if str(value)[::-1].find('.') > 3:
                value = round(value, 3)
            if percentage or memory or record:
                if percentage:
                    return f"{value * 100:.1f}%"
                elif record:
                    return f"{value:.1f} B"
                else:
                    return f"{value / 1024:.1f} KiB"
            else:
                return f"{value}"


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


def profiler_visualization(config_dict: dict) -> None:
    if config_dict['table']['profiler_type'] in ['Tabular', 'TimeSeries']:
        profiler_tabular_timeseries(config_dict)
    elif config_dict['table']['profiler_type'] == 'Hierarchical':
        profiler_hierarchical(config_dict)
    elif config_dict['table']['profiler_type'] == 'RDFGraph':
        profiler_rdfGraph(config_dict)
    elif config_dict['table']['profiler_type'] == 'Textual':
        profiler_textual(config_dict)
    elif config_dict['table']['profiler_type'] in ['Raster', 'Vista_Raster']:
        profiler_raster(config_dict)


def profiler_raster(config_dict: dict) -> None:
    overview, variables = st.tabs(["Overview", "Images"])
    i = 0
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
                            cols = st.columns(2)

                            comb_band_dict = {
                                'Number of images with this band': __print_metric(band['n_of_imgs']),
                                'Minimum': __print_metric(band['min']),
                                'Mean': __print_metric(band['average']),
                                'Maximum': __print_metric(band['max'])
                            }

                            comb_band_df = pd.DataFrame(comb_band_dict.items(),
                                                        columns=['Statistics', 'Values'])
                            cols[0].dataframe(comb_band_df, use_container_width=True, hide_index=True, key=i)
                            i = i + 1

                            if 'imgs' in band:
                                enter_imgs = True
                                data_df = pd.DataFrame(band['imgs'], columns=['raster', 'date', 'percentage'])
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
                            st.plotly_chart(fig, use_container_width=True, key=i)
                            i = i + 1

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

                        st.altair_chart(bar_freq, use_container_width=True, key=i)
                        i = i + 1

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
                        cols[0].dataframe(lai_df, use_container_width=True, hide_index=True, key=i)
                        i = i + 1

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

                        cols[1].plotly_chart(fig, use_container_width=True, key=i)
                        i = i + 1
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
                                                      hide_index=True, key=i)
                                    i = i + 1

                                    data_df = pd.DataFrame(
                                        {
                                            "rasters": band['img_names'],
                                        }
                                    )

                                    if 'imgs' in band:
                                        enter_imgs = True
                                        data_df = pd.DataFrame(band['imgs'], columns=['raster', 'date', 'percentage'])
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
                                    st.plotly_chart(fig, use_container_width=True, key=i)
                                    i = i + 1

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

                                st.altair_chart(bar_freq, use_container_width=True, key=i)
                                i = i + 1

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
                                cols[0].dataframe(lai_df, use_container_width=True, hide_index=True, key=i)
                                i = i + 1

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

                                cols[1].plotly_chart(fig, use_container_width=True, key=i)
                                i = i + 1
                            count += 1

    with variables:
        variable_stats = config_dict['variables']

        variable_stats = __transform_list_of_dicts(variable_stats, 'name')

        chosen_variables = list(variable_stats.keys())

        variables = st.multiselect(
            "View images",
            chosen_variables,
            chosen_variables[:5],
            key='variables'
        )

        for variable_name in st.session_state.variables:
            variable = variable_stats[variable_name]
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

                    transform = variable['transform']
                    affine_transform = Affine(transform[0], transform[1], transform[2],
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
                    m = folium.Map(location=[lat, lon], tiles='OpenStreetMap', min_zoom=2, max_bounds=True,
                                   zoom_start=3)
                    gjson.add_to(m)
                    with cols[1]:
                        m.fit_bounds(gjson.get_bounds())
                        folium_static(m)

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
                                st.plotly_chart(fig, use_container_width=True, key=i)
                                i = i + 1

                        with stats:
                            if band['count'] == 0:
                                if config_dict['table']['profiler_type'] == 'Vista_Raster':
                                    st.subheader('There are no LAI values!')
                                else:
                                    st.subheader('There are no values!')
                            else:
                                lai_dict = {'Mean': __print_metric(band['average']),
                                            'Standard Deviation': __print_metric(band['stddev']),
                                            'Kurtosis': __print_metric(band['kurtosis']),
                                            'Skewness': __print_metric(band['skewness']),
                                            'Variance': __print_metric(band['variance'])}

                                lai_df = pd.DataFrame(lai_dict.items(), columns=['Statistics', 'Values'])
                                st.dataframe(lai_df, use_container_width=True, hide_index=True, key=i)
                                i = i + 1

                                if band['name'] == 'undefined':
                                    lai_dict['uuid'] = __print_metric(band['uuid'])

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
                                st.plotly_chart(fig, use_container_width=True, key=i)
                                i = i + 1

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
                                        st.plotly_chart(fig, use_container_width=True, key=i)
                                        i = i + 1

                                with stats:
                                    if band['count'] == 0:
                                        if config_dict['table']['profiler_type'] == 'Vista_Raster':
                                            st.subheader('There are no LAI values!')
                                        else:
                                            st.subheader('There are no values!')
                                    else:
                                        lai_dict = {'Mean': __print_metric(band['average']),
                                                    'Standard Deviation': __print_metric(band['stddev']),
                                                    'Kurtosis': __print_metric(band['kurtosis']),
                                                    'Skewness': __print_metric(band['skewness']),
                                                    'Variance': __print_metric(band['variance'])}

                                        lai_df = pd.DataFrame(lai_dict.items(),
                                                              columns=['Statistics', 'Values'])
                                        st.dataframe(lai_df, use_container_width=True, hide_index=True, key=i)
                                        i = i + 1

                                        if band['name'] == 'undefined':
                                            lai_dict['uuid'] = __print_metric(band['uuid'])

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

                                        st.plotly_chart(fig, use_container_width=True, key=i)
                                        i = i + 1

                                count += 1


def profiler_textual(config_dict: dict) -> None:
    overview, texts = st.tabs(["Overview", "Texts"])
    i = 0
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
            st.metric(__change_font("Ratio Uppercase (\\%)"),
                      __print_metric(table_stats['ratio_uppercase'], percentage=True))
            st.metric(__change_font("Ratio Digits (\\%)"),
                      __print_metric(table_stats['ratio_digits'], percentage=True))
            st.metric(__change_font("Ratio Special Characters (\\%)"),
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
            st.plotly_chart(fig, use_container_width=True, key=i)
            i = i + 1

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
                st.plotly_chart(fig, use_container_width=True, key=i)
                i = i + 1
        st.divider()
        st.header('Term Frequency Distribution')

        term_freq_df = pd.DataFrame(table_stats['term_frequency'], columns=['term', 'count'])
        term_freq_df.columns = ['Term', 'Count']
        term_freq_df['Frequency'] = (term_freq_df['Count'] / term_freq_df['Count'].sum())
        term_freq_df['Frequency'] = term_freq_df['Frequency'].map('{:.5%}'.format)
        st.dataframe(term_freq_df, use_container_width=True, hide_index=True, key=i)
        i = i + 1

    with texts:
        variable_stats = config_dict['variables']

        variable_stats = __transform_list_of_dicts(variable_stats, 'name')

        chosen_variables = list(variable_stats.keys())

        variables = st.multiselect(
            "View texts",
            chosen_variables,
            chosen_variables[:5],
            key='variables'
        )

        for variable_name in st.session_state.variables:
            variable = variable_stats[variable_name]
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
                        st.metric(__change_font("Ratio Uppercase (\\%)"),
                                  __print_metric(variable['ratio_uppercase'], percentage=True))
                        st.metric(__change_font("Ratio Digits (\\%)"),
                                  __print_metric(variable['ratio_digits'], percentage=True))
                        st.metric(__change_font("Ratio Special Characters (\\%)"),
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
                    st.plotly_chart(fig, use_container_width=True, key=i)
                    i = i + 1

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
                    st.dataframe(term_freq_df, use_container_width=True, hide_index=True, key=i)
                    i = i + 1

                with tab4:
                    term_freq_df = pd.DataFrame(variable['special_characters_distribution'],
                                                columns=['type', 'count'])
                    term_freq_df.columns = ['Special Character', 'Count']
                    term_freq_df['Special Character'].replace(' ', 'Whitespace', inplace=True)
                    term_freq_df.sort_values(by=['Count'], ascending=False, inplace=True)
                    term_freq_df['Frequency'] = (term_freq_df['Count'] / variable['num_characters'])
                    term_freq_df['Frequency'] = term_freq_df['Frequency'].map('{:.5%}'.format)
                    st.dataframe(term_freq_df, use_container_width=True, hide_index=True, key=i)
                    i = i + 1

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
                        st.dataframe(words_df, use_container_width=True, hide_index=True, key=i)
                        i = i + 1

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

                        st.plotly_chart(fig, use_container_width=True, key=i)
                        i = i + 1

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
                        st.dataframe(sentences_df, use_container_width=True, hide_index=True, key=i)
                        i = i + 1

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

                        st.plotly_chart(fig, use_container_width=True, key=i)
                        i = i + 1


def profiler_rdfGraph(config_dict: dict) -> None:
    table_stats = config_dict['table']
    i = 0
    dataset_stats, conn_comp = st.columns(2)
    with dataset_stats:
        st.subheader("Dataset Statistics")
        dataset_stats1, dataset_stats2 = st.columns(2)
        with dataset_stats1:
            st.metric(__change_font("Number of Nodes"), __print_metric(table_stats['num_nodes']))
            st.metric(__change_font("Number of Edges"), __print_metric(table_stats['num_edges']))
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
        st.plotly_chart(fig, use_container_width=True, key=i)
        i = i + 1
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
            cols[0].dataframe(degree_df, use_container_width=True, hide_index=True, key=i)
            i = i + 1

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

            cols[1].plotly_chart(fig, use_container_width=True, key=i)
            i = i + 1
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
            cols[0].dataframe(degree_df, use_container_width=True, hide_index=True, key=i)
            i = i + 1

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

            cols[1].plotly_chart(fig, use_container_width=True, key=i)
            i = i + 1
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
            cols[0].dataframe(degree_df, use_container_width=True, hide_index=True, key=i)
            i = i + 1

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

            cols[1].plotly_chart(fig, use_container_width=True, key=i)
            i = i + 1
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
            cols[0].dataframe(degree_df, use_container_width=True, hide_index=True, key=i)
            i = i + 1

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

            cols[1].plotly_chart(fig, use_container_width=True, key=i)
            i = i + 1
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


def profiler_hierarchical(config_dict: dict) -> None:
    uniqueness_counts_dict, types_count_dict, types_names_dict, uniqueness_names_dict, nested_levels_dict = \
        __calc_hierarchical(config_dict['variables'])

    table_stats = config_dict['table']
    dataset_stats, rec_types = st.columns(2)
    with dataset_stats:
        st.subheader("Dataset Statistics")
        st.metric(__change_font("Number of records"), __print_metric(table_stats['num_records']))
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
            st.dataframe(depth_df, use_container_width=True, hide_index=True, key=0)

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

            st.plotly_chart(fig, use_container_width=True, key=1)
        with plot:
            nested_levels_dict = dict(
                sorted(nested_levels_dict.items(), key=lambda item: item[1], reverse=True))
            nested_levels_df = pd.DataFrame(nested_levels_dict.items(), columns=['Nested Level', 'Count'])
            nested_levels_df = nested_levels_df[nested_levels_df.Count != 0]

            fig = px.pie(nested_levels_df, values='Count', names='Nested Level')
            fig.update_traces(textposition='inside', textfont_size=14)
            fig.update_layout(title_text='Nested Level Distribution', title_x=0.5,
                              title_y=1, title_xanchor='center',
                              title_yanchor='top', legend_font_size=14,
                              title_font_size=18)
            st.plotly_chart(fig, use_container_width=True, key=2)
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


def profiler_tabular_timeseries(config_dict: dict) -> None:
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
                st.metric(__change_font("Missing values (\\%)"),
                          __print_metric(table_stats['p_cells_missing'], percentage=True))
                st.metric(__change_font("Total size in memory"),
                          __print_metric(table_stats['memory_size'], memory=True))
                st.metric(__change_font("Average record size in memory"),
                          __print_metric(table_stats['record_size'], record=True))
            if 'ts_gaps_frequency_distribution' in table_stats and not table_stats['light_mode']:
                if len(table_stats['ts_gaps_frequency_distribution']) != 0:
                    with gap_stats:
                        st.subheader('Total Gap Length')
                        cols = st.columns(2)
                        gaps_dict = {'Minimum': __print_metric(table_stats['ts_min_gap']),
                                     'Mean': __print_metric(table_stats['ts_avg_gap']),
                                     'Maximum': __print_metric(table_stats['ts_max_gap'])
                                     }

                        gaps_df = pd.DataFrame(gaps_dict.items(), columns=['Statistics', 'Values'])
                        cols[0].dataframe(gaps_df, use_container_width=True, hide_index=True)

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
                st.metric(__change_font("Missing cells (\\%)"),
                          __print_metric(table_stats['p_cells_missing'], percentage=True))
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

        variable_stats = __transform_list_of_dicts(variable_stats, 'name')

        chosen_variables = list(variable_stats.keys())

        variables = st.multiselect(
            "View columns",
            chosen_variables,
            chosen_variables[:5],
            key='variables'
        )

        i = 0
        for variable_name in st.session_state.variables:
            variable = variable_stats[variable_name]
            with st.expander(variable['name'] + ' ( ' + variable['type'] + ' ) '):
                if table_stats['light_mode'] and variable['type'] != 'Unsupported':
                    stats, plot = st.columns(2)
                    with stats:
                        st.metric(__change_font("Distinct"),
                                  __print_metric(variable['n_distinct']))
                        st.metric(__change_font("Distinct (\\%)"),
                                  __print_metric(variable['p_distinct'], percentage=True))
                        st.metric(__change_font("Missing values"),
                                  __print_metric(variable['num_missing']))
                        st.metric(__change_font("Missing values (\\%)"),
                                  __print_metric(variable['p_missing'],
                                                 percentage=True))
                        st.metric(__change_font("Unique"), __print_metric(variable['n_unique']))
                        st.metric(__change_font("Unique (\\%)"),
                                  __print_metric(variable['uniqueness'], percentage=True))
                        st.metric(__change_font("Memory size"),
                                  __print_metric(variable['memory_size'],
                                                 memory=True))

                elif variable['type'] == 'Textual':
                    stats, plot = st.columns(2)
                    with stats:
                        st.metric(__change_font("Distinct"),
                                  __print_metric(variable['n_distinct']))
                        st.metric(__change_font("Distinct (\\%)"),
                                  __print_metric(variable['p_distinct'], percentage=True))
                        st.metric(__change_font("Missing values"),
                                  __print_metric(variable['num_missing']))
                        st.metric(__change_font("Missing values (\\%)"),
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
                        st.plotly_chart(fig, use_container_width=True, key=i)
                        i = i + 1

                    tab1, tab2 = st.tabs(["Overview", "Distributions"])

                    with tab1:
                        unique, ratios = st.columns(2)

                        with unique:
                            st.subheader("Unique")
                            st.metric(__change_font("Unique"), __print_metric(variable['n_unique']))
                            st.metric(__change_font("Unique (\\%)"),
                                      __print_metric(variable['uniqueness'], percentage=True))

                        with ratios:
                            st.subheader("Ratios")
                            st.metric(__change_font("Ratio Uppercase (\\%)"),
                                      __print_metric(variable['ratio_uppercase'], percentage=True))
                            st.metric(__change_font("Ratio Digits (\\%)"),
                                      __print_metric(variable['ratio_digits'], percentage=True))
                            st.metric(__change_font("Ratio Special Characters (\\%)"),
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
                            st.dataframe(characters_df, use_container_width=True, hide_index=True, key=i)
                            i = i + 1

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

                            st.plotly_chart(fig, use_container_width=True, key=i)
                            i = i + 1

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
                            st.dataframe(words_df, use_container_width=True, hide_index=True, key=i)
                            i = i + 1

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
                            st.plotly_chart(fig, use_container_width=True, key=i)
                            i = i + 1

                elif variable['type'] == 'Categorical':
                    stats, plot = st.columns(2)
                    with stats:
                        st.metric(__change_font("Distinct"), __print_metric(variable['n_distinct']))
                        st.metric(__change_font("Distinct (\\%)"),
                                  __print_metric(variable['p_distinct'], percentage=True))
                        st.metric(__change_font("Missing values"), __print_metric(variable['num_missing']))
                        st.metric(__change_font("Missing values (\\%)"),
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
                        st.plotly_chart(fig, use_container_width=True, key=i)
                        i = i + 1

                    tab1, tab2 = st.tabs(["Overview", "Categories"])
                    with tab1:
                        unique, sample = st.columns(2)

                        with unique:
                            st.subheader("Unique")
                            st.metric(__change_font("Unique"), __print_metric(variable['n_unique']))
                            st.metric(__change_font("Unique (\\%)"),
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
                        st.dataframe(categories_df, use_container_width=True, hide_index=True, key=i)
                        i = i + 1

                elif variable['type'] == 'Unsupported':
                    stats, plot = st.columns(2)
                    with stats:
                        st.metric(__change_font("Distinct"),
                                  __print_metric(variable['n_distinct']))
                        st.metric(__change_font("Distinct (\\%)"),
                                  __print_metric(variable['p_distinct'], percentage=True))
                        st.metric(__change_font("Missing values"),
                                  __print_metric(variable['num_missing']))
                        st.metric(__change_font("Missing values (\\%)"),
                                  __print_metric(variable['p_missing'],
                                                 percentage=True))
                        st.metric(__change_font("Unique"), __print_metric(variable['n_unique']))
                        st.metric(__change_font("Unique (\\%)"),
                                  __print_metric(variable['uniqueness'], percentage=True))
                        st.metric(__change_font("Memory size"),
                                  __print_metric(variable['memory_size'],
                                                 memory=True))

                elif variable['type'] == 'Numeric':
                    with st.container():
                        cols = st.columns(3)
                        cols[0].metric(__change_font("Distinct"), __print_metric(variable['n_distinct']))
                        cols[0].metric(__change_font("Distinct (\\%)"),
                                       __print_metric(variable['p_distinct'], percentage=True))
                        cols[0].metric(__change_font("Missing values"),
                                       __print_metric(variable['num_missing']))
                        cols[0].metric(__change_font("Missing values (\\%)"),
                                       __print_metric(variable['p_missing'], percentage=True))
                        cols[0].metric(__change_font("Infinite"), __print_metric(variable['n_infinite']))
                        cols[0].metric(__change_font("Infinite (\\%)"),
                                       __print_metric(variable['p_infinite'], percentage=True))
                        cols[0].metric(__change_font("Mean"), __print_metric(variable['average']))

                        cols[1].metric(__change_font("Minimum"), __print_metric(variable['min']))
                        cols[1].metric(__change_font("Maximum"), __print_metric(variable['max']))
                        cols[1].metric(__change_font("Zeros"), __print_metric(variable['n_zeros']))
                        cols[1].metric(__change_font("Zeros (\\%)"),
                                       __print_metric(variable['p_zeros'], percentage=True))
                        cols[1].metric(__change_font("Negative"), __print_metric(variable['n_negative']))
                        cols[1].metric(__change_font("Negative (\\%)"),
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

                        cols[2].altair_chart(bar_freq, use_container_width=True, key=i)
                        i = i + 1

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

                            stat_cols[0].dataframe(quan_stats_df, use_container_width=True, hide_index=True, key=i)
                            i = i + 1

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

                            stat_cols[0].plotly_chart(fig, use_container_width=True, key=i)
                            i = i + 1

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
                                desc_stats_dict['Monotonicity'] = 'Not monotonic'
                            else:
                                desc_stats_dict['Monotonicity'] = 'Monotonic'

                            quan_stats_df = pd.DataFrame(desc_stats_dict.items(),
                                                         columns=['Statistics', 'Values'])

                            stat_cols[1].dataframe(quan_stats_df, use_container_width=True,
                                                   hide_index=True, key=i)
                            i = i + 1

                    with tab2:

                        value_counts = {value_counts['value']: value_counts['count']
                                        for value_counts in variable['freq_value_counts']}
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
                            five_min_value_counts = {value_counts['value']: value_counts['count']
                                                     for value_counts in variable['freq_five_min_values']}
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
                            five_max_value_counts = {value_counts['value']: value_counts['count']
                                                     for value_counts in variable['freq_five_max_values']}
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
                        st.metric(__change_font("Distinct (\\%)"),
                                  __print_metric(variable['p_distinct'], percentage=True))
                        st.metric(__change_font("Missing values"), __print_metric(variable['num_missing']))
                        st.metric(__change_font("Missing values (\\%)"),
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
                        st.plotly_chart(fig, use_container_width=True, key=i)
                        i = i + 1

                    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Overview", "Sample", "Mbr",
                                                                        "Convex Hull", "Heat Map",
                                                                        "Frequencies", "Distributions"])
                    with tab1:
                        unique, geometry = st.columns(2)

                        with unique:
                            st.subheader("Unique")
                            st.metric(__change_font("Unique"), __print_metric(variable['n_unique']))
                            st.metric(__change_font("Unique (\\%)"),
                                      __print_metric(variable['uniqueness'], percentage=True))
                        with geometry:
                            st.subheader("Geometry")
                            st.metric(__change_font("Crs"), __print_metric(variable['crs']))
                            st.metric(__change_font("Centroid"), __print_metric(variable['centroid']))

                    with tab2:
                        samples = {sample['row']: sample['cat']
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
                                        for value_counts in variable['freq_value_counts']}
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
                            st.dataframe(length_df, use_container_width=True, hide_index=True, key=i)
                            i = i + 1

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
                            st.plotly_chart(fig, use_container_width=True, key=i)
                            i = i + 1

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
                            st.dataframe(area_df, use_container_width=True, hide_index=True, key=i)
                            i = i + 1

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
                            st.plotly_chart(fig, use_container_width=True, key=i)
                            i = i + 1

                elif variable['type'] == 'DateTime':
                    stats1, stats2, plot = st.columns(3)
                    with stats1:
                        st.metric(__change_font("Distinct"), __print_metric(variable['n_distinct']))
                        st.metric(__change_font("Distinct (\\%)"),
                                  __print_metric(variable['p_distinct'], percentage=True))
                        st.metric(__change_font("Missing values"), __print_metric(variable['num_missing']))
                        st.metric(__change_font("Missing values (\\%)"),
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

                        plot.altair_chart(bar_freq, use_container_width=True, key=i)
                        i = i + 1

                elif variable['type'] == 'TimeSeries':
                    with st.container():
                        cols = st.columns(3)
                        cols[0].metric(__change_font("Distinct"), __print_metric(variable['n_distinct']))
                        cols[0].metric(__change_font("Distinct (\\%)"),
                                       __print_metric(variable['p_distinct'], percentage=True))
                        cols[0].metric(__change_font("Missing values"),
                                       __print_metric(variable['num_missing']))
                        cols[0].metric(__change_font("Missing values (\\%)"),
                                       __print_metric(variable['p_missing'], percentage=True))
                        cols[0].metric(__change_font("Infinite"), __print_metric(variable['n_infinite']))
                        cols[0].metric(__change_font("Infinite (\\%)"),
                                       __print_metric(variable['p_infinite'], percentage=True))
                        cols[0].metric(__change_font("Mean"), __print_metric(variable['average']))

                        cols[0].metric(__change_font("Minimum"), __print_metric(variable['min']))
                        cols[1].metric(__change_font("Maximum"), __print_metric(variable['max']))
                        cols[1].metric(__change_font("Zeros"), __print_metric(variable['n_zeros']))
                        cols[1].metric(__change_font("Zeros (\\%)"),
                                       __print_metric(variable['p_zeros'], percentage=True))
                        cols[1].metric(__change_font("Negative"), __print_metric(variable['n_negative']))
                        cols[1].metric(__change_font("Negative (\\%)"),
                                       __print_metric(variable['p_negative']))
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

                        cols[2].altair_chart(bar_freq, use_container_width=True, key=i)
                        i = i + 1

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

                            gaps_col[1].plotly_chart(fig, use_container_width=True, key=i)
                            i = i + 1
                        else:
                            st.subheader('There are no gaps in the timeseries')

                    tab1, tab2, tab3, tab4 = st.tabs(["Statistics", "Common Values",
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

                            stat_cols[0].plotly_chart(fig, use_container_width=True, key=i)
                            i = i + 1

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
                                desc_stats_dict['Monotonicity'] = 'Not monotonic'
                            else:
                                desc_stats_dict['Monotonicity'] = 'Monotonic'

                            desc_stats_dict['Augmented Dickey-Fuller test p-value'] = __print_metric(
                                variable['add_fuller'])

                            quan_stats_df = pd.DataFrame(desc_stats_dict.items(),
                                                         columns=['Statistics', 'Values'])

                            stat_cols[1].dataframe(quan_stats_df, use_container_width=True, hide_index=True, key=i)
                            i = i + 1

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

                            tsfresh_stats_df = pd.DataFrame(tsfresh_stats_dict.items(),
                                                            columns=['Statistics', 'Values'])

                            stat_cols[2].dataframe(tsfresh_stats_df, use_container_width=True,
                                                   hide_index=True, key=i)
                            i = i + 1
                    with tab2:

                        value_counts = {value_counts['value']: value_counts['count']
                                        for value_counts in variable['freq_value_counts']}

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
                        min_tab, max_tab = st.tabs(['Minimum 10 values', 'Maximum 10 values'])
                        with min_tab:
                            five_min_value_counts = {value_counts['value']: value_counts['count']
                                                     for value_counts in variable['freq_five_min_values']}
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
                            five_max_value_counts = {value_counts['value']: value_counts['count']
                                                     for value_counts in variable['freq_five_max_values']}
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
                    with tab4:
                        fig = __recreate_plots(variable['acf_pacf'])
                        st.pyplot(fig)
                        i = i + 1
                elif variable['type'] == 'Boolean':
                    cols = st.columns(2)
                    cols[0].metric(__change_font("Distinct"), __print_metric(variable['n_distinct']))
                    cols[0].metric(__change_font("Distinct (\\%)"),
                                   __print_metric(variable['p_distinct'], percentage=True))
                    cols[0].metric(__change_font("Missing values"),
                                   __print_metric(variable['num_missing']))
                    cols[0].metric(__change_font("Missing values (\\%)"),
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
                    cols[1].plotly_chart(fig, use_container_width=True, key=i)
                    i = i + 1

                else:
                    st.metric(__change_font("Type"), f"{variable['type']}")
