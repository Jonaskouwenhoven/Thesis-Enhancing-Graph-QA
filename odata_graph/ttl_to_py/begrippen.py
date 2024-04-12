from fuzzywuzzy import fuzz, process
from rdflib import Graph, SKOS

ttl_file = './odata_graph/graph/begrippen.trig'
g = Graph()
g.parse(ttl_file, format='trig')
g = list(g.store.contexts())[0]

begrippen = {str(v): k for k, v in g.subject_objects(SKOS.definition) if v.language == 'nl'}


def match_definition(description):
    def_match = process.extract(description, begrippen.keys(), scorer=fuzz.QRatio, limit=1)
    if len(def_match) > 0 and def_match[0][1] >= 90:
        stop = 1
        # Definition found matches pretty good with one from the taxonomy
        return begrippen[def_match[0][0]]
    else:
        return None
