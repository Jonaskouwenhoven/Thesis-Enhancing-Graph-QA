from rdflib import Graph, SKOS, URIRef

from odata_graph.ttl_to_py.defined_namespace_generator import generate_defined_namespace

ttl_file = './odata_graph/graph/frequenties.ttl'

g = Graph()
g.parse(ttl_file, format='n3')

# Conversion dictionary for OData4 frequencies to FREQ types
FREQ_DICT = {str(label): uri for uri, label
             in set(g.subject_objects(SKOS.prefLabel)) | set(g.subject_objects(SKOS.altLabel))
             if label.language == 'nl'}

if __name__ == "__main__":
    generate_defined_namespace(ttl_file=ttl_file,
                               py_file='./odata_graph/namespaces/_FREQ.py',
                               ns_uri='https://vocabs.cbs.nl/def/Frequency')
