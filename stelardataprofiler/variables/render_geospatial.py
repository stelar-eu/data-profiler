from typing import List

import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import HeatMap
from shapely.geometry import box


from pandas_profiling.config import Settings
from pandas_profiling.report.formatters import (
    fmt,
    fmt_bytesize,
    fmt_number,
    fmt_numeric,
    fmt_percent,
    help,
)
from pandas_profiling.report.presentation.core import (
    HTML,
    Container,
    FrequencyTable,
    FrequencyTableSmall,
    Image,
    Table,
    VariableInfo,
)
from pandas_profiling.report.presentation.core.renderable import Renderable
from pandas_profiling.report.presentation.frequency_table_utils import freq_table
from pandas_profiling.report.structure.variables.render_common import render_common
from pandas_profiling.visualisation.plot import cat_frequency_plot, histogram


def __render_geospatial_frequency(
        config: Settings, summary: dict, varid: str
) -> Renderable:
    frequency_table = Table(
        [
            {
                "name": "Unique",
                "value": fmt_number(summary["n_unique"]),
                "hint": help(
                    "The number of unique values (all values that occur exactly once in the dataset)."
                ),
                "alert": "n_unique" in summary["alert_fields"],
            },
            {
                "name": "Unique (%)",
                "value": fmt_percent(summary["p_unique"]),
                "alert": "p_unique" in summary["alert_fields"],
            },
        ],
        name="Unique",
        anchor_id=f"{varid}_unique_stats",
        style=config.html.style,
    )

    return frequency_table


def __render_geospatial_statistics(
        config: Settings, summary: dict, varid: str
) -> Renderable:
    geom_table = Table(
        [
            {
                "name": "Crs",
                "value": fmt(summary["crs"]),
                "alert": False,
            },
            {
                "name": "Length",
                "value": fmt_number(summary["length"]),
                "alert": False,
            },
            {
                "name": "Centroid",
                "value": fmt(summary["centroid"]),
                "alert": False,
            },
            '''
            {
                "name": "Convex hull",
                "value": fmt(summary["union_convex_hull"]),
                "alert": False,
            },
            '''
        ],
        name="Geometry",
        anchor_id=f"{varid}_geom_stats",
        style=config.html.style,
    )

    return geom_table


def __render_geospatial(config: Settings, summary: dict) -> dict:
    varid = summary["varid"]
    n_obs_cat = config.vars.cat.n_obs
    image_format = config.plot.image_format

    template_variables = render_common(config, summary)

    info = VariableInfo(
        summary["varid"],
        summary["varname"],
        "Geometry",
        summary["alerts"],
        summary["description"],
        style=config.html.style,
    )

    table = Table(
        [
            {
                "name": "Distinct",
                "value": fmt(summary["n_distinct"]),
                "alert": "n_distinct" in summary["alert_fields"],
            },
            {
                "name": "Distinct (%)",
                "value": fmt_percent(summary["p_distinct"]),
                "alert": "p_distinct" in summary["alert_fields"],
            },
            {
                "name": "Missing",
                "value": fmt(summary["n_missing"]),
                "alert": "n_missing" in summary["alert_fields"],
            },
            {
                "name": "Missing (%)",
                "value": fmt_percent(summary["p_missing"]),
                "alert": "p_missing" in summary["alert_fields"],
            },
            {
                "name": "Memory size",
                "value": fmt_bytesize(summary["memory_size"]),
                "alert": False,
            },
        ],
        style=config.html.style,
    )

    fqm = FrequencyTableSmall(
        freq_table(
            freqtable=summary["geom_types"],
            n=summary["count"],
            max_number_to_print=n_obs_cat,
        ),
        redact=config.vars.cat.redact,
    )

    template_variables["top"] = Container([info, table, fqm], sequence_type="grid")

    # ============================================================================================

    frequency_table = FrequencyTable(
        template_variables["freq_table_rows"],
        name="",
        anchor_id=f"{varid}common_values",
        redact=config.vars.cat.redact,
    )

    unique_stats = __render_geospatial_frequency(config, summary, varid)

    # Add Sample Map
    sample_df = pd.DataFrame.from_dict(summary["first_rows"].to_dict(), orient="index", columns=["geometry"])
    sample_gdf = gpd.GeoDataFrame(
        sample_df, geometry=gpd.GeoSeries.from_wkt(sample_df.geometry))
    centroid = sample_gdf.geometry.unary_union.centroid
    lon = centroid.x
    lat = centroid.y
    m = folium.Map(location=[lat, lon], zoom_start=10, tiles="CartoDB positron")
    for _, r in sample_gdf.iterrows():
        gjson = folium.GeoJson(data=r[0])
        # gjson.add_child(folium.Popup(r[0].wkt))
        gjson.add_to(m)
    iframe = m._repr_html_()
    sample_map = HTML(iframe,
                      anchor_id=f"{varid}_sample",
                      name='Sample')

    # Add Mbr Map
    geom_stats = __render_geospatial_statistics(config, summary, varid)
    overview_items = [unique_stats, geom_stats]
    mbr_box = gpd.GeoSeries.from_wkt([summary["mbr"]])
    centroid = mbr_box.centroid
    lon = centroid.x
    lat = centroid.y
    m = folium.Map(location=[lat, lon], zoom_start=10, tiles="CartoDB positron")
    folium.GeoJson(data=mbr_box[0]).add_to(m)
    iframe = m._repr_html_()
    mbr_map = HTML(iframe,
                   anchor_id=f"{varid}_mbr",
                   name="Mbr")

    '''
    mbr = summary["mbr"]
    mbr_box = box(mbr[0], mbr[1], mbr[2], mbr[3])
    centroid = mbr_box.centroid
    lon = centroid.x
    lat = centroid.y
    m = folium.Map(location=[lat, lon], zoom_start=10, tiles="CartoDB positron")
    folium.GeoJson(data=mbr_box).add_to(m)
    iframe = m._repr_html_()
    mbr_map = HTML(iframe,
                   anchor_id=f"{varid}_mbr",
                   name="Mbr")
    '''
    # Add Convex Hull Map
    convex_hull = gpd.GeoSeries.from_wkt([summary["union_convex_hull"]])
    centroid = convex_hull.centroid
    lon = centroid.x
    lat = centroid.y
    m = folium.Map(location=[lat, lon], zoom_start=10, tiles="CartoDB positron")
    folium.GeoJson(data=convex_hull[0]).add_to(m)
    iframe = m._repr_html_()
    convex_hull_map = HTML(iframe,
                           anchor_id=f"{varid}_convex_hull",
                           name="Convex Hull")

    # Add HeatMap
    heatmap_df = pd.DataFrame(summary["heatmap"]).values
    centroid = gpd.GeoSeries.from_wkt([summary["centroid"]])
    lon = centroid.x
    lat = centroid.y
    m = folium.Map(location=[lat, lon], zoom_start=10, tiles="CartoDB positron")
    HeatMap(heatmap_df).add_to(folium.FeatureGroup(name="Heat Map").add_to(m))
    folium.LayerControl().add_to(m)
    iframe = m._repr_html_()
    heatmap = HTML(iframe,
                   anchor_id=f"{varid}_heatmap",
                   name="Heat Map")

    string_items: List[Renderable] = [frequency_table]

    show = config.plot.cat_freq.show
    max_unique = config.plot.cat_freq.max_unique

    if show and (max_unique > 0):
        if isinstance(summary["value_counts_without_nan"], list):
            string_items.append(
                Container(
                    [
                        Image(
                            cat_frequency_plot(
                                config,
                                s,
                            ),
                            image_format=image_format,
                            alt=config.html.style._labels[idx],
                            name=config.html.style._labels[idx],
                            anchor_id=f"{varid}cat_frequency_plot_{idx}",
                        )
                        if summary["n_distinct"][idx] <= max_unique
                        else HTML(
                            f"<h4 class='indent'>{config.html.style._labels[idx]}</h4><br />"
                            f"<em>Number of variable categories passes threshold (<code>config.plot.cat_freq.max_unique</code>)</em>"
                        )
                        for idx, s in enumerate(summary["value_counts_without_nan"])
                    ],
                    anchor_id=f"{varid}cat_frequency_plot",
                    name="Plot",
                    sequence_type="batch_grid",
                    batch_size=len(config.html.style._labels),
                )
            )
        elif (
                len(config.html.style._labels) == 1 and summary["n_distinct"] <= max_unique
        ):
            string_items.append(
                Image(
                    cat_frequency_plot(
                        config,
                        summary["value_counts_without_nan"],
                    ),
                    image_format=image_format,
                    alt="Plot",
                    name="Plot",
                    anchor_id=f"{varid}cat_frequency_plot",
                )
            )

    bottom_items = [
        Container(
            overview_items,
            name="Overview",
            anchor_id=f"{varid}overview",
            sequence_type="batch_grid",
            batch_size=len(overview_items),
            titles=False,
        ),
        sample_map,
        mbr_map,
        convex_hull_map,
        heatmap,
        Container(
            string_items,
            name="Frequencies",
            anchor_id=f"{varid}string",
            sequence_type="named_list"
            if len(config.html.style._labels) > 1
            else "batch_grid",
            batch_size=len(config.html.style._labels),
        ),
    ]

    # Bottom
    template_variables["bottom"] = Container(
        bottom_items, sequence_type="tabs", anchor_id=f"{varid}bottom"
    )

    return template_variables
