from typing import List

from pandas_profiling.config import Settings
from pandas_profiling.report.formatters import (
    fmt,
    fmt_bytesize,
    fmt_number,
    fmt_numeric,
    fmt_percent
)
from pandas_profiling.report.presentation.core import Container, Table
from pandas_profiling.report.presentation.core.renderable import Renderable
from pandas_profiling.report.structure.overview import (
    get_dataset_schema,
    get_dataset_column_definitions,
    get_dataset_alerts,
    get_dataset_reproduction
)

__all__ = []

def __get_dataset_items(config: Settings, summary: dict, alerts: list) -> list:
    """Returns the dataset overview (at the top of the report)
    Args:
        config: settings object
        summary: the calculated summary
        alerts: the alerts
    Returns:
        A list with components for the dataset overview (overview, reproduction, alerts)
    """

    items: List[Renderable] = [__get_dataset_overview(config, summary)]

    metadata = {key: config.dataset.dict()[key] for key in config.dataset.dict().keys()}

    if len(metadata) > 0 and any(len(value) > 0 for value in metadata.values()):
        items.append(get_dataset_schema(config, metadata))

    column_details = {
        key: config.variables.descriptions[key]
        for key in config.variables.descriptions.keys()
    }

    if len(column_details) > 0:
        items.append(get_dataset_column_definitions(config, column_details))

    if alerts:
        items.append(get_dataset_alerts(config, alerts))

    items.append(get_dataset_reproduction(config, summary))

    return items


def __get_dataset_overview(config: Settings, summary: dict) -> Renderable:
    table_metrics = [
        {
            "name": "Number of variables",
            "value": fmt_number(summary["table"]["n_var"]),
        },
        {
            "name": "Number of observations",
            "value": fmt_number(summary["table"]["n"]),
        },
        {
            "name": "Missing cells",
            "value": fmt_number(summary["table"]["n_cells_missing"]),
        },
        {
            "name": "Missing cells (%)",
            "value": fmt_percent(summary["table"]["p_cells_missing"]),
        },
        {
            "name": "Profiler Type",
            "value": fmt(summary["table"]["profiler_type"]),
        },
    ]
    if "n_duplicates" in summary["table"]:
        table_metrics.extend(
            [
                {
                    "name": "Duplicate rows",
                    "value": fmt_number(summary["table"]["n_duplicates"]),
                },
                {
                    "name": "Duplicate rows (%)",
                    "value": fmt_percent(summary["table"]["p_duplicates"]),
                },
            ]
        )

    table_metrics.extend(
        [
            {
                "name": "Total size in memory",
                "value": fmt_bytesize(summary["table"]["memory_size"]),
            },
            {
                "name": "Average record size in memory",
                "value": fmt_bytesize(summary["table"]["record_size"]),
            },
        ]
    )

    dataset_info = Table(
        table_metrics, name="Dataset statistics", style=config.html.style
    )

    dataset_types = Table(
        [
            {
                "name": str(type_name),
                "value": fmt_numeric(count, precision=config.report.precision),
            }
            for type_name, count in summary["table"]["types"].items()
        ],
        name="Variable types",
        style=config.html.style,
    )

    return Container(
        [dataset_info, dataset_types],
        anchor_id="dataset_overview",
        name="Overview",
        sequence_type="grid",
    )
