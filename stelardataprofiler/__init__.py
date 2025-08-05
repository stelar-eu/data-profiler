import os
os.environ["NUMBA_DISABLE_CUDA"] = "1"

from .raster.profiler import (profile_raster,
                              profile_vista_rasters,
                              profile_raster_with_config,
                              profile_vista_rasters_with_config)
from .hierarchical.profiler import profile_hierarchical, profile_hierarchical_with_config
from .rdfGraph.profiler import profile_rdfGraph, profile_rdfGraph_with_config
from .tabular_timeseries.profiler import (profile_timeseries,
                                          profile_tabular,
                                          profile_timeseries_with_config,
                                          profile_tabular_with_config,
                                          type_detection,
                                          type_detection_with_config)
from .text.profiler import profile_text, profile_text_with_config
from .profiler import prepare_mapping, run_profile
from .utils import read_config, write_to_json

__all__ = [
    # Raster profilers
    'profile_raster',
    'profile_vista_rasters',
    'profile_raster_with_config',
    'profile_vista_rasters_with_config',

    # Hierarchical profilers
    'profile_hierarchical',
    'profile_hierarchical_with_config',

    # RDF Graph profilers
    'profile_rdfGraph',
    'profile_rdfGraph_with_config',

    # Tabular/TimeSeries profilers
    'profile_timeseries',
    'profile_tabular',
    'profile_timeseries_with_config',
    'profile_tabular_with_config',
    'type_detection',
    'type_detection_with_config',

    # Text profilers
    'profile_text',
    'profile_text_with_config',

    # General profilers
    'run_profile',
    'prepare_mapping',

    # Read + Write
    'read_config',
    'write_to_json'
]
