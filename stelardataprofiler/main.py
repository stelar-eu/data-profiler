import sys
import os
import json
import shutil
from .profiler import run_profile, prepare_mapping
from .utils import read_config

def main():
    # Read configuration to a dictionary
    # Configuration includes all parameters for input/output
    config = read_config(sys.argv[1])

    # Run the profiler according to user's configuration
    # This method executes the specified profiler and issues the JSON result (and the HTML, is specified)
    run_profile(config)

    # If "rdf" and "serialization" options are specified in config, it should also prepare the suitable mapping for subsequent generation of the RDF graph
    prepare_mapping(config)


if __name__ == "__main__":
    exit(main())
