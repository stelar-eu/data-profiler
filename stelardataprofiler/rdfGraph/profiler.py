import os
from datetime import datetime
import dateutil.parser
import pandas as pd
from rdflib import Graph, RDF, URIRef
from rdflib.extras.external_graph_libs import rdflib_to_networkx_multidigraph
import networkx as nx
from ..utils import write_to_json

def profile_rdfGraph_with_config(config: dict) -> None:
    """
    This method performs profiling on rdfGraph data and writes the resulting profile dictionary based on a configuration dictionary.

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
    if 'serialization' not in config['input']:
        print("No rdflib format is specified so the default 'application/rdf+xml' is used.")
        parse_format: str = 'application/rdf+xml'
    else:
        parse_format: str = str(config['input']['serialization']).lower()

    profile_dict = profile_rdfGraph(my_file_path=my_file_path, parse_format=parse_format)

    # Write resulting profile dictionary
    write_to_json(profile_dict, output_json_path)


def profile_rdfGraph(my_file_path: str, parse_format: str = 'application/rdf+xml'):
    """
    This method performs profiling and generates a profiling dictionary for a given rdf file that exists in the given path.

    :param my_file_path: the path to a rdf file.
    :type my_file_path: str
    :param parse_format: the format of the rdf file. (see rdflib package to find the available formats e.g. 'turtle', 'application/rdf+xml', 'n3', 'nt', etc.)
    :type parse_format: str, optional
    :return: A dict which contains the results of the profiler for the rdf.
    :rtype: dict

    """

    # Calculate the number of nodes
    def __calc_num_nodes(g: Graph):
        return len(g.all_nodes())

    # Calculate the number of edges
    def __calc_num_edges(g: Graph):
        return len(g)

    # Calculate the number of namespaces
    def __calc_num_namespaces(g: Graph):
        v = g.serialize(format="ttl")

        return v.count('@prefix')

    # Calculate the number of classes and a class frequency list
    def __calc_class_features(g: Graph):

        num_classes = set()
        classes_distribution = dict()

        for cl in g.objects(predicate=RDF.type):
            if str(cl) not in classes_distribution:
                classes_distribution[str(cl)] = 0

            classes_distribution[str(cl)] += 1

            num_classes.add(str(cl))

        # List of classes and their frequencies in the graph
        class_distribution_list = []

        for c, v in sorted(classes_distribution.items(), key=lambda x: x[1], reverse=True):
            class_dict = dict({
                'class_name': c,
                'count': v
            })
            class_distribution_list.append(class_dict)

        return len(num_classes), class_distribution_list

    # Calculate the number of object type properties
    def __calc_num_object_properties(g: Graph):
        # Extract set from objects of triples
        object_list = {x for x in g.objects() if isinstance(x, URIRef)}
        # Append set extracted from subjects of triples
        object_list.update({x for x in g.subjects() if isinstance(x, URIRef)})

        return len(object_list)

    # Calculate the number of data type properties
    def __calc_num_datatype_properties(g: Graph):
        data_property_list = {x for x in g.objects() if not isinstance(x, URIRef)}

        return len(data_property_list)

    # Calculate the number of connected components and a list with each connected component and its number of nodes
    def __calc_cc_features(nx_g: nx.MultiDiGraph):
        nx_g_undirected = nx_g.to_undirected()
        cc = list(nx.connected_components(nx_g_undirected))

        cc_list = []

        for i, c in enumerate(cc):
            cc_dict = dict({
                'component_name': i,
                'num_nodes': len(c)
            })
            cc_list.append(cc_dict)

        return len(cc), cc_list

    # Calculate the density of the graph
    def __calc_density(nx_g: nx.MultiDiGraph):
        nx_g_density = nx.density(nx_g)

        return nx_g_density

    # Calculate the degree_centrality_distribution
    def __calc_degree_centrality(nx_g: nx.MultiDiGraph):

        dc = nx.degree_centrality(nx_g)
        degrees_centrality = []
        for _, v in dc.items():
            degrees_centrality.append(v)

        s = pd.Series(degrees_centrality)
        stats = s.describe(percentiles=[.10, .25, .75, .90])

        degree_centrality_distribution = {
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

        return degree_centrality_distribution

    # Calculate the degree_distribution
    def __calc_degree(nx_g: nx.MultiDiGraph):
        degrees = []
        for _, v in nx_g.degree:
            degrees.append(v)

        s = pd.Series(degrees)
        stats = s.describe(percentiles=[.10, .25, .75, .90])

        degree_distribution = {
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

        return degree_distribution

    # Calculate the in_degree_distribution
    def __calc_in_degree(nx_g: nx.MultiDiGraph):
        in_degrees = []
        for _, v in nx_g.in_degree:
            in_degrees.append(v)

        s = pd.Series(in_degrees)
        stats = s.describe(percentiles=[.10, .25, .75, .90])

        in_degrees_distribution = {
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

        return in_degrees_distribution

    # Calculate the out_degree_distribution
    def __calc_out_degree(nx_g: nx.MultiDiGraph):
        out_degrees = []
        for _, v in nx_g.out_degree:
            out_degrees.append(v)

        s = pd.Series(out_degrees)
        stats = s.describe(percentiles=[.10, .25, .75, .90])

        out_degrees_distribution = {
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

        return out_degrees_distribution

    profile_dict = {
        'analysis': {
            'title': 'Profiling Report',
            'date_start': '',
            'date_end': '',
            'duration': '',
            'filenames': [my_file_path]
        },
        'table': {
            'profiler_type': 'RDFGraph',
            'byte_size': 0,
            'num_nodes': 0,
            'num_edges': 0,
            'num_namespaces': 0,
            'num_classes': 0,
            'num_object_properties': 0,
            'num_datatype_properties': 0,
            'density': 0,
            'num_connected_components': 0,
            'connected_components': [],
            'degree_centrality_distribution': dict(),
            'degree_distribution': dict(),
            'in_degree_distribution': dict(),
            'out_degree_distribution': dict(),
            'class_distribution': []

        },
        'variables': []
    }

    # Start time
    now = datetime.now()
    start_string = now.strftime("%Y-%m-%d %H:%M:%S.%f")
    profile_dict['analysis']['date_start'] = start_string

    # File size
    profile_dict['table']['byte_size'] = os.path.getsize(my_file_path)

    g = Graph()
    g.parse(my_file_path, format=parse_format)

    # Number of nodes
    profile_dict['table']['num_nodes'] = __calc_num_nodes(g)

    # Number of edges
    profile_dict['table']['num_edges'] = __calc_num_edges(g)

    # Number of namespaces
    profile_dict['table']['num_namespaces'] = __calc_num_namespaces(g)

    # Number of Classes + class_distribution
    profile_dict['table']['num_classes'], profile_dict['table']['class_distribution'] = __calc_class_features(g)

    # Number of Object type properties
    profile_dict['table']['num_object_properties'] = __calc_num_object_properties(g)

    # Number of Data type properties
    profile_dict['table']['num_datatype_properties'] = __calc_num_datatype_properties(g)

    # Create networkx graph
    nx_g = rdflib_to_networkx_multidigraph(g)

    # Number of connected components + List of connected components
    profile_dict['table']['num_connected_components'], profile_dict['table'][
        'connected_components'] = __calc_cc_features(
        nx_g)

    # Density
    profile_dict['table']['density'] = __calc_density(nx_g)

    # Calculate degree_centrality_distribution
    profile_dict['table']['degree_centrality_distribution'] = __calc_degree_centrality(nx_g)

    # Calculate degree_distribution
    profile_dict['table']['degree_distribution'] = __calc_degree(nx_g)

    # Calculate in_degree_distribution
    profile_dict['table']['in_degree_distribution'] = __calc_in_degree(nx_g)

    # Calculate out_degree_distribution
    profile_dict['table']['out_degree_distribution'] = __calc_out_degree(nx_g)

    # End time
    now = datetime.now()
    end_string = now.strftime("%Y-%m-%d %H:%M:%S.%f")
    profile_dict['analysis']['date_end'] = end_string

    # Time Difference
    profile_dict['analysis']['duration'] = str(
        dateutil.parser.parse(profile_dict['analysis']['date_end']) - dateutil.parser.parse(
            profile_dict['analysis']['date_start']))

    return profile_dict