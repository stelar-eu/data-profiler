from typing import List, Tuple, Union

import pandas as pd

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


def __render_textual_frequency(
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


def __render_textual_ratios(
    config: Settings, summary: dict, varid: str
) -> Renderable:
    ratio_table = Table(
        [
            {
                "name": "Ratio Uppercase",
                "value": fmt_percent(summary["ratio_uppercase"]),
            },
            {
                "name": "Ratio Digits",
                "value": fmt_percent(summary["ratio_digits"]),
            },
            {
                "name": "Ratio Special Characters",
                "value": fmt_percent(summary["ratio_special_characters"]),
            },
        ],
        name="Ratios",
        anchor_id=f"{varid}_ratio_stats",
        style=config.html.style,
    )

    return ratio_table

def __render_textual_distributions(
    config: Settings, summary: dict, varid: str
) -> Renderable:
    num_chars_distribution = summary['num_chars_distribution']

    num_chars_table = Table(
        [
            {
                "name": "Min",
                "value": fmt_number(int(num_chars_distribution['min'])),
            },
            {
                "name": "Max",
                "value": fmt_number(int(num_chars_distribution['max'])),
            },
            {
                "name": "Median",
                "value": fmt_number(int(num_chars_distribution['median'])),
            },
            {
                "name": "Average",
                "value": fmt_numeric(num_chars_distribution['average']),
            },
            {
                "name": "Standard Deviation",
                "value": fmt_numeric(num_chars_distribution['stddev']),
            },
            {
                "name": "Kurtosis",
                "value": fmt_numeric(num_chars_distribution['kurtosis']),
            },
            {
                "name": "Skewness",
                "value": fmt_numeric(num_chars_distribution['skewness']),
            },
            {
                "name": "Variance",
                "value": fmt_numeric(num_chars_distribution['variance']),
            },
            {
                "name": "Percentile - 10",
                "value": fmt_number(int(num_chars_distribution['percentile10'])),
            },
            {
                "name": "Percentile - 25",
                "value": fmt_number(int(num_chars_distribution['percentile25'])),
            },
            {
                "name": "Percentile - 75",
                "value": fmt_number(int(num_chars_distribution['percentile75'])),
            },
            {
                "name": "Percentile - 90",
                "value": fmt_number(int(num_chars_distribution['percentile90'])),
            },
        ],
        name="Number of Characters",
        anchor_id=f"{varid}_num_chars_distribution_stats",
        style=config.html.style,
    )

    num_words_distribution = summary['num_words_distribution']

    num_words_table = Table(
        [
            {
                "name": "Min",
                "value": fmt_number(int(num_words_distribution['min'])),
            },
            {
                "name": "Max",
                "value": fmt_number(int(num_words_distribution['max'])),
            },
            {
                "name": "Median",
                "value": fmt_number(int(num_words_distribution['median'])),
            },
            {
                "name": "Average",
                "value": fmt_numeric(num_words_distribution['average']),
            },
            {
                "name": "Standard Deviation",
                "value": fmt_numeric(num_words_distribution['stddev']),
            },
            {
                "name": "Kurtosis",
                "value": fmt_numeric(num_words_distribution['kurtosis']),
            },
            {
                "name": "Skewness",
                "value": fmt_numeric(num_words_distribution['skewness']),
            },
            {
                "name": "Variance",
                "value": fmt_numeric(num_words_distribution['variance']),
            },
            {
                "name": "Percentile - 10",
                "value": fmt_number(int(num_words_distribution['percentile10'])),
            },
            {
                "name": "Percentile - 25",
                "value": fmt_number(int(num_words_distribution['percentile25'])),
            },
            {
                "name": "Percentile - 75",
                "value": fmt_number(int(num_words_distribution['percentile75'])),
            },
            {
                "name": "Percentile - 90",
                "value": fmt_number(int(num_words_distribution['percentile90'])),
            },
        ],
        name="Number of Words",
        anchor_id=f"{varid}_num_words_distribution_stats",
        style=config.html.style,
    )

    return [num_chars_table, num_words_table]


def ___get_n(value: Union[list, pd.DataFrame]) -> Union[int, List[int]]:
    """Helper function to deal with multiple values"""
    if isinstance(value, list):
        n = [v.sum() for v in value]
    else:
        n = value.sum()
    return n


def __render_textual(config: Settings, summary: dict) -> dict:
    varid = summary["varid"]
    n_obs_cat = config.vars.cat.n_obs
    image_format = config.plot.image_format

    template_variables = render_common(config, summary)

    info = VariableInfo(
        summary["varid"],
        summary["varname"],
        "Textual",
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
    if 'language_distribution' in summary and summary['language_distribution']:
        temp_table = []

        for lang, perc in summary['language_distribution'].items():
            temp_table.append({
                "label": lang,
                "width": perc / 100.0,
                "count": perc,
                "percentage": perc,
                "n": len(summary['language_distribution']),
                "extra_class": ""
            })
        fqm = FrequencyTableSmall([temp_table], redact=config.vars.cat.redact, )

        template_variables["top"] = Container([info, table, fqm], sequence_type="grid")
    else:
        template_variables["top"] = Container([info, table], sequence_type="grid")

    # ============================================================================================

    unique_stats = __render_textual_frequency(config, summary, varid)

    ratio_stats = __render_textual_ratios(config, summary, varid)

    overview_items = [unique_stats, ratio_stats]

    distribution_items = __render_textual_distributions(config, summary, varid)

    bottom_items = [
        Container(
            overview_items,
            name="Overview",
            anchor_id=f"{varid}overview",
            sequence_type="batch_grid",
            batch_size=len(overview_items),
            titles=False,
        ),
        Container(
            distribution_items,
            name="Distributions",
            anchor_id=f"{varid}distributions",
            sequence_type="batch_grid",
            batch_size=len(config.html.style._labels),
            titles=False,
        ),
    ]

    # Bottom
    template_variables["bottom"] = Container(
        bottom_items, sequence_type="tabs", anchor_id=f"{varid}bottom"
    )

    return template_variables
