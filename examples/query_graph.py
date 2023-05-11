import rdflib
from rdflib import Graph, URIRef
#from rdflib.namespace import Namespace, NamespaceManager, split_uri
from rdflib.namespace import OWL, RDF, RDFS  # type: ignore
#from rdflib.extras.external_graph_libs import rdflib_to_networkx_graph
#from rdflib.term import _is_valid_uri


# Load RDF graph with profiling information and submit SPARQL queries

# Create graph
g = Graph()

# Load the RDF graph from the output file specifying its serialization
g.parse('./output/tabular_vector_results.rdf', format='nt')

# Get statistics, e.g., graph size => number of statements (triples)
num_triples = len(g)
print('triples:', num_triples)

# ... or number of nodes:
num_nodes = len(g.all_nodes())
print('nodes:', num_nodes)

# Submit a SPARQL query against the RDF graph:
qres = g.query(
    """
PREFIX dcat: <http://www.w3.org/ns/dcat#>
PREFIX dct: <http://purl.org/dc/terms/>
PREFIX geo: <http://www.opengis.net/ont/geosparql#>
PREFIX geof: <http://www.opengis.net/def/function/geosparql/>
PREFIX klms: <http://stelar-project.eu/klms#>
SELECT ?prof ?attr_name ?geom ?geom_type ?pct WHERE {
  ?prof klms:contains ?attr .
  ?attr rdf:type klms:SpatialAttribute .
  ?attr dct:title ?attr_name .
  ?attr dcat:bbox ?geom .
  ?attr klms:geomTypeDistribution ?geom_type_distr .
  ?geom_type_distr klms:contains ?geom_type .
  ?geom_type klms:name "MultiPolygon" .
  ?geom_type klms:count ?pct .
  FILTER (?pct > 100)

} 
    """
)


# List the query results:

for row in qres:
    print(row)
