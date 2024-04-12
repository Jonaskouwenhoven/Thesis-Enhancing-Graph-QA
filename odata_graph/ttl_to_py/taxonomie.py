from rdflib import Graph, SKOS

ttl_file = './odata_graph/graph/taxonomie.trig'
g = Graph()
g.parse(ttl_file, format='trig')
g = list(g.store.contexts())[0]

taxonomie = {str(v): k for k, v in set(g.subject_objects(SKOS.prefLabel)) | set(g.subject_objects(SKOS.altLabel))
             if v.language == 'nl'}


def match_taxon(label):
    taxonomie.get(label)
