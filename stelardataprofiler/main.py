import sys
import os
import json
import shutil
from stelardataprofiler import write_to_json


def main():
    # Helper function
    def readConfig(jsonFile: str):
        """Read configuration settings from JSON.

        Args:
             jsonFile: Path to JSON file that contains the configuration parameters.

        Returns:
              A dictionary with all configuration settings.
        """

        with open(jsonFile) as f:
            dictConfig = json.load(f)
            return dictConfig

    # Read configuration
    config = readConfig(sys.argv[1])

    print(str(os.path.dirname(os.path.abspath(__file__))))
    profile_type: str = config['profile']['type']
    input_dir_path = config['input']['path']
    input_file_name = config['input']['file']
    output_dir_path = config['output']['path']
    output_json_name = config['output']['json']
    output_html_name = ''
    if 'html' in config['output']:
        output_html_name = config['output']['html']
    only_directory_path = False

    # Create input paths
    if input_file_name == '':
        input_path = os.path.abspath(input_dir_path)
        only_directory_path = True
    else:
        input_path = os.path.abspath(os.path.join(input_dir_path, input_file_name))

    # Create output file paths
    output_dir_path = os.path.abspath(output_dir_path)
    output_json_path = os.path.abspath(os.path.join(output_dir_path, output_json_name))
    output_html_path = ''
    if output_html_name != '':
        output_html_path = os.path.abspath(os.path.join(output_dir_path, output_html_name))

    # Run profile based on json parameters
    if profile_type.lower() == 'timeseries':
        if only_directory_path:
            print('No input file was found for timeseries profile!')
        else:
            my_file_path = input_path
            if 'time' in config['input']['columns']:
                from stelardataprofiler import profile_timeseries
                time_column = config['input']['columns']['time']
                header = config['input']['header']
                sep = config['input']['separator']
                profile_dict = profile_timeseries(my_file_path=my_file_path, time_column=time_column,
                                                  header=header, sep=sep, html_path=output_html_path)
                write_to_json(profile_dict, output_json_path)
            else:
                print("Please add 'time' as key and the time column name of the input .csv "
                      'as value in the json under input.columns')

    elif profile_type.lower() in ['tabular', 'vector']:
        if only_directory_path:
            print('No input file was found for tabular or vector profiles!')
        else:
            my_file_path = input_path
            header = config['input']['header']
            sep = config['input']['separator']
            columns_dict: dict = config['input']['columns']
            longitude_column: str = None
            latitude_column: str = None
            wkt_column: str = None
            from stelardataprofiler import profile_tabular
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

            write_to_json(profile_dict, output_json_path)
    elif profile_type.lower() == 'raster':
        my_path = input_path
        from stelardataprofiler import profile_raster
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

        write_to_json(profile_dict, output_json_path)
    elif profile_type.lower() == 'textual':
        my_path = input_path
        from stelardataprofiler import profile_text
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

        write_to_json(profile_dict, output_json_path)
    elif profile_type.lower() == 'hierarchical':
        my_file_path = input_path
        if only_directory_path:
            print('No input file was found for hierarchical profile!')
        else:
            from stelardataprofiler import profile_hierarchical
            profile_dict = profile_hierarchical(my_file_path=my_file_path)
            write_to_json(profile_dict, output_json_path)
    elif profile_type.lower() == 'rdfgraph':
        my_file_path = input_path
        if only_directory_path:
            print('No input file was found for rdfGraph profile!')
        else:
            from stelardataprofiler import profile_rdfGraph
            if 'serialization' not in config['input']:
                print("No rdflib format is specified so the default 'application/rdf+xml' is used.")
                parse_format: str = 'application/rdf+xml'
            else:
                parse_format: str = str(config['input']['serialization']).lower()
            profile_dict = profile_rdfGraph(my_file_path=my_file_path, parse_format=parse_format)
            write_to_json(profile_dict, output_json_path)
    else:
        print('The profile type is not available!\n'
              'Please use one of the following types:\n'
              "'timeseries', 'tabular', 'vector', 'raster', 'text', 'hierarchical', 'rdfGraph'")

    # Get parameters required for conversion to RDF
    output_path = config['output']['path']
    json_file = config['output']['json']
    rdf_file = config['output']['rdf']
    profile_type = config['profile']['type'].lower()
    rdf_serialization = config['output']['serialization']

    # Handle special cases (timeseries, vector) of tabular profile
    if profile_type == 'vector' or profile_type == 'timeseries':
        profile_type = 'tabular'

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


if __name__ == "__main__":
    exit(main())
