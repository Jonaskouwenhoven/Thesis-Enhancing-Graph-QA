from rdflib import Graph, SKOS

ttl_file = './odata_graph/graph/geo_gebieden.ttl'
g = Graph()
g.parse(ttl_file, format='n3')

geo_gebieden = {str(v): k for k, v in set(g.subject_objects(SKOS.notation)) if v.language == 'nl'}


def match_geo(identifier):
    geo_gebieden.get(identifier)
