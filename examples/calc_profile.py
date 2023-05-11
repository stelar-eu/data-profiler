import sys
import stelardataprofiler as dpl

# Execution command: python <path-to-python-script>.py <path-to-config>.json

# Read user configuration from JSON file given as argument
# Configuration includes all parameters for input/output of the profiling process

config = dpl.read_config(sys.argv[1])


# Run the profiler according to configuration
# This method executes the specified profiler and issues results in JSON (and in HTML, if specified in config)

dpl.run_profile(config)


# If "rdf" and "serialization" options are specified in config, ...
# ...also prepare the suitable mapping for subsequent generation of the RDF graph
# The resulting mapping must be used with the RML Mapper in Java in order to generate the RDF triples

dpl.prepare_mapping(config)


