"""Generate the report."""
from pandas_profiling.report.presentation.flavours import HTMLReport
import copy
import json
from tqdm.auto import tqdm
from stelardataprofiler.overview import __get_dataset_items
from pandas_profiling.config import Settings
from pandas_profiling.report.presentation.core import (
    HTML,
    Container,
    Dropdown,
    Collapse,
)

from pandas_profiling.report.presentation.core.renderable import Renderable
from pandas_profiling.report.presentation.core.root import Root
from pandas_profiling.report.structure.correlations import get_correlation_items
from pandas_profiling.report.structure.report import (
    get_interactions,
    get_missing_items,
    get_sample_items,
    get_duplicates_items
)
import pandas as pd
import numpy as np
from pandas_profiling.model.sample import Sample
from pandas_profiling.model.summarizer import format_summary

from pandas_profiling.model.alerts import AlertType
from pandas_profiling.report.presentation.core import Sample, ToggleButton, Variable
from typing import Any, Callable, Dict, List, Union
from stelardataprofiler.variables.render_timeseries import __render_timeseries
from stelardataprofiler.variables.render_geospatial import __render_geospatial
from stelardataprofiler.variables.render_textual import __render_textual

from pathlib import Path
from pandas_profiling.report.presentation.flavours.html.templates import (
    create_html_assets,
)
import warnings


def __get_report_structure(config: Settings, summary: dict, variables: Container = None) -> Root:
    """Generate an HTML report from summary statistics and a given sample.
    Args:
      config: report Settings object
      summary: Statistics to use for the overview, variables, correlations and missing values.
      variables: In case of 'default' (tsfresh_mode) we add a sample of the timeseries
    Returns:
      The profile report in HTML format
    """
    disable_progress_bar = not config.progress_bar
    with tqdm(
            total=1, desc="Generate report structure", disable=disable_progress_bar
    ) as pbar:
        alerts = summary["alerts"]

        section_items: List[Renderable] = [
            Container(
                __get_dataset_items(config, summary, alerts),
                sequence_type="tabs",
                name="Overview",
                anchor_id="overview",
            ),
        ]

        if len(summary["variables"]) > 0:
            section_items.append(
                Dropdown(
                    name="Variables",
                    anchor_id="variables-dropdown",
                    id="variables-dropdown",
                    items=list(summary["variables"]),
                    item=Container(
                        __render_variables_section(config, summary),
                        sequence_type="accordion",
                        name="Variables",
                        anchor_id="variables",
                    ),
                )
            )

        scatter_items = get_interactions(config, summary["scatter"])
        if len(scatter_items) > 0:
            section_items.append(
                Container(
                    scatter_items,
                    sequence_type="tabs" if len(scatter_items) <= 10 else "select",
                    name="Interactions",
                    anchor_id="interactions",
                ),
            )

        corr = get_correlation_items(config, summary)
        if corr is not None:
            section_items.append(corr)

        missing_items = get_missing_items(config, summary)
        if len(missing_items) > 0:
            section_items.append(
                Container(
                    missing_items,
                    sequence_type="tabs",
                    name="Missing values",
                    anchor_id="missing",
                )
            )

        sample_items = get_sample_items(config, summary["sample"])
        if len(sample_items) > 0:
            section_items.append(
                Container(
                    items=sample_items,
                    sequence_type="tabs",
                    name="Sample",
                    anchor_id="sample",
                )
            )

        duplicate_items = get_duplicates_items(config, summary["duplicates"])
        if len(duplicate_items) > 0:
            section_items.append(
                Container(
                    items=duplicate_items,
                    sequence_type="batch_grid",
                    batch_size=len(duplicate_items),
                    name="Duplicate rows",
                    anchor_id="duplicate",
                )
            )

        if variables is not None:
            section_items.append(variables)

        sections = Container(
            section_items,
            name="Root",
            sequence_type="sections",
            full_width=config.html.full_width,
        )

        pbar.update()

    footer = HTML(
        content='Report generated by <a href="https://ydata.ai/?utm_source=opensource&utm_medium=pandasprofiling'
                '&utm_campaign=report">YData</a>. '
    )

    return Root("Root", sections, footer, style=config.html.style)


def __get_html_report(config: Settings, description: dict, variables: Container = None) -> str:
    """Transforms the html dict of the profiler to a profile report object.

    Args:
        config: Custom settings
        description: Dict containing a description for each variable in the DataFrame.
        variables: In case of 'default' (tsfresh_mode) we add a sample of the timeseries

    Returns:
        A profile report object
    """
    report = __get_report_structure(config, description, variables)
    with tqdm(
            total=1, desc="Render HTML", disable=not config.progress_bar
    ) as pbar:
        html = HTMLReport(copy.deepcopy(report)).render(
            nav=config.html.navbar_show,
            offline=config.html.use_local_assets,
            inline=config.html.inline,
            assets_prefix=config.html.assets_prefix,
            primary_color=config.html.style.primary_colors[0],
            logo=config.html.style.logo,
            theme=config.html.style.theme,
            title='Profiler',
            date=description["analysis"]["date_start"],
            version='0.1',
        )

        if config.html.minify_html:
            from htmlmin.main import minify

            html = minify(html, remove_all_empty_space=True, remove_comments=True)
        pbar.update()
    return html


def __to_file(config: Settings, output: str, output_file: Union[str, Path], silent: bool = True) -> None:
    """Write the report to a file.
    Args:
        output: The html or json of the profiler
        config: Custom settings
        output_file: The name or the path of the file to generate including the extension (.html, .json).
        silent: if False, opens the file in the default browser or download it in a Google Colab environment
    """
    if not isinstance(output_file, Path):
        output_file = Path(str(output_file))

    if output_file.suffix == ".json":
        data = output
    else:
        if not config.html.inline:
            config.html.assets_path = str(output_file.parent)
            if config.html.assets_prefix is None:
                config.html.assets_prefix = str(output_file.stem) + "_assets"
            create_html_assets(config, output_file)

        data = output

        if output_file.suffix != ".html":
            suffix = output_file.suffix
            output_file = output_file.with_suffix(".html")
            warnings.warn(
                f"Extension {suffix} not supported. For now we assume .html was intended. "
                f"To remove this warning, please use .html or .json."
            )

    disable_progress_bar = not config.progress_bar
    with tqdm(
            total=1, desc="Export report to file", disable=disable_progress_bar
    ) as pbar:
        output_file.write_text(data, encoding="utf-8")
        pbar.update()

    if not silent:
        try:
            from google.colab import files  # noqa: F401

            files.download(output_file.absolute().as_uri())
        except ModuleNotFoundError:
            import webbrowser

            webbrowser.open_new_tab(output_file.absolute().as_uri())


def __to_json(config: Settings, description_set: dict) -> str:
    """Transforms the html dict of the profiler to a json formatted string.
    Args:
        config: Custom settings
        description_set: Dict containing a description for each variable in the DataFrame.

    Returns:
        a json formatted string
    """

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
                return encode_it(o.to_dict(orient="records"))
            elif isinstance(o, np.ndarray):
                return encode_it(o.tolist())
            elif isinstance(o, Sample):
                return encode_it(o.dict())
            elif isinstance(o, np.generic):
                return o.item()
            else:
                return str(o)

    description = description_set

    with tqdm(
            total=1, desc="Render JSON", disable=not config.progress_bar
    ) as pbar:
        description = format_summary(description)
        description = encode_it(description)
        data = json.dumps(description, indent=4)
        pbar.update()
    return data


def __render_variables_section(config: Settings, dataframe_summary: dict) -> list:
    """Render the HTML for each of the variables in the DataFrame.
    Args:
        config: report Settings object
        dataframe_summary: The statistics for each variable.
    Returns:
        The rendered HTML, where each row represents a variable.
    """

    templs = []

    descriptions = config.variables.descriptions
    show_description = config.show_variable_description
    reject_variables = config.reject_variables

    render_map = __get_render_map()

    for idx, summary in dataframe_summary["variables"].items():

        # Common template variables
        if not isinstance(dataframe_summary["alerts"], tuple):
            alerts = [
                alert.fmt()
                for alert in dataframe_summary["alerts"]
                if alert.column_name == idx
            ]

            alert_fields = {
                field
                for alert in dataframe_summary["alerts"]
                if alert.column_name == idx
                for field in alert.fields
            }

            alert_types = {
                alert.alert_type
                for alert in dataframe_summary["alerts"]
                if alert.column_name == idx
            }
        else:
            alerts = tuple(
                [alert.fmt() for alert in summary_alerts if alert.column_name == idx]
                for summary_alerts in dataframe_summary["alerts"]
            )  # type: ignore

            alert_fields = {
                field
                for summary_alerts in dataframe_summary["alerts"]
                for alert in summary_alerts
                if alert.column_name == idx
                for field in alert.fields
            }

            alert_types = {
                alert.alert_type
                for summary_alerts in dataframe_summary["alerts"]
                for alert in summary_alerts
                if alert.column_name == idx
            }

        template_variables = {
            "varname": idx,
            "varid": hash(idx),
            "alerts": alerts,
            "description": descriptions.get(idx, "") if show_description else "",
            "alert_fields": alert_fields,
        }

        template_variables.update(summary)

        # Per type template variables
        if isinstance(summary["type"], list):
            types = set(summary["type"])
            if len(types) == 1:
                variable_type = list(types)[0]
            else:
                # This logic may be treated by the typeset
                if (types == {"Numeric", "Categorical"}) or types == {
                    "Categorical",
                    "Unsupported",
                }:
                    # Treating numeric as categorical, if one is unsupported, still render as categorical
                    variable_type = "Categorical"
                else:
                    raise ValueError(f"Types for {idx} are not compatible: {types}")
        else:
            variable_type = summary["type"]
        render_map_type = render_map.get(variable_type, render_map["Unsupported"])
        template_variables.update(render_map_type(config, template_variables))

        # Ignore these
        if reject_variables:
            ignore = AlertType.REJECTED in alert_types
        else:
            ignore = False

        bottom = None
        if "bottom" in template_variables and template_variables["bottom"] is not None:
            btn = ToggleButton("More details", anchor_id=template_variables["varid"])
            bottom = Collapse(btn, template_variables["bottom"])

        var = Variable(
            template_variables["top"],
            bottom=bottom,
            anchor_id=template_variables["varid"],
            name=idx,
            ignore=ignore,
        )

        templs.append(var)

    return templs


# TODO: Add Textual in HTML
def __get_render_map() -> Dict[str, Callable]:
    import pandas_profiling.report.structure.variables as render_algorithms

    render_map = {
        "Boolean": render_algorithms.render_boolean,
        "Numeric": render_algorithms.render_real,
        "Complex": render_algorithms.render_complex,
        "DateTime": render_algorithms.render_date,
        "Categorical": render_algorithms.render_categorical,
        "URL": render_algorithms.render_url,
        "Path": render_algorithms.render_path,
        "File": render_algorithms.render_file,
        "Image": render_algorithms.render_image,
        "Unsupported": render_algorithms.render_generic,
        "TimeSeries": __render_timeseries,
        "Geometry": __render_geospatial,
        "Textual": __render_textual
    }

    return render_map

