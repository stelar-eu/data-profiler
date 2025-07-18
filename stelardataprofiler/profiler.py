from .raster.profiler import (profile_raster_with_config,
                              profile_vista_rasters_with_config)
from .hierarchical.profiler import profile_hierarchical_with_config
from .rdfGraph.profiler import profile_rdfGraph_with_config
from .tabular_timeseries.profiler import (profile_timeseries_with_config,
                                          profile_tabular_with_config)
from .text.profiler import profile_text, profile_text_with_config


def run_profile(config: dict) -> None:
    """
    This method executes the specified profiler and writes the resulting profile dictionary based on a configuration dictionary.

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
    import os
    import shutil

    # Get parameters required for conversion to RDF
    in_file = config['output']['json']
    rdf_file = config['output']['rdf']
    rdf_serialization = config['output']['serialization']
    profile_type = config['profile']['type'].lower()

    # Handle special cases (timeseries, vector) of tabular profile
    if profile_type in ('vector', 'timeseries'):
        profile_type = 'tabular'

    # Handle special cases (raster, vista) of raster profile
    if profile_type in ('raster', 'vista'):
        profile_type = 'raster'

    # Determine the directory where the JSON profile is stored
    output_dir = os.path.dirname(os.path.abspath(in_file))

    # Set path for mapping file to be saved in the same folder as the JSON profile
    map_file = os.path.join(output_dir, 'mapping.ttl')

    # Find the appropriate mapping template
    map_template = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'mappings', f'{profile_type}_mapping.ttl'
    )

    # Copy mapping template to destination as 'mapping.ttl'
    if not os.path.isfile(map_template):
        print(f'ERROR: Mapping {map_template} not found! Check whether such mapping exists at',
              os.path.abspath(map_template))
        sys.exit(1)

    shutil.copyfile(map_template, map_file)
    print(f'Mapping {map_template} copied to {map_file}')

    # Check again for file existence after copy
    if not os.path.isfile(map_file):
        print(f'ERROR: Mapping file {map_file} not found after copy!')
        sys.exit(1)

    # Edit the mapping file: replace placeholder with actual JSON profile path
    with open(map_file, 'r') as file:
        filedata = file.read()

    filedata = filedata.replace('./out/profile.json', in_file)

    with open(map_file, 'w') as file:
        file.write(filedata)
