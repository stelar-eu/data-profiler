from datetime import datetime
import os
from dataprofiler import Data, Profiler
import pandas as pd
import dateutil.parser
from ..utils import write_to_json


def profile_hierarchical_with_config(config: dict) -> None:
    """
    This method performs profiling on hierarchical data and writes the resulting profile dictionary based on a configuration dictionary.

    :param config: a dictionary with all configuration settings.
    :type config: dict
    :return: None.
    :rtype: None

    """
    # input file path(s)
    input_file_path = config['input']['files']

    if isinstance(input_file_path, list):
        if len(input_file_path) == 1:
            my_file_path = os.path.abspath(input_file_path[0])
        else:
            raise ValueError(f"Invalid input: {input_file_path} must be a valid file path or list with one file path")
    elif isinstance(input_file_path, str) and os.path.isfile(os.path.abspath(input_file_path)):
        my_file_path = os.path.abspath(input_file_path)
    else:
        raise ValueError(f"Invalid input: {input_file_path} must be a valid file path or list of file paths")

    # output file path
    output_json_path = os.path.abspath(config['output']['json'])

    # Run raster profile
    profile_dict = profile_hierarchical(my_file_path=my_file_path)

    # Write resulting profile dictionary
    write_to_json(profile_dict, output_json_path)


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
        'variables': []

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
