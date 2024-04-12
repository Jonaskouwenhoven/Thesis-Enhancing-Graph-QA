import datetime
import inspect
import os
from rdflib import Graph, URIRef, RDF, RDFS, OWL, DCTERMS as DCT


def generate_defined_namespace(ttl_file: str, py_file: str, ns_uri: str):
    """

        :param ttl_file: str path to the ttl file
        :param py_file: str path to the python file to be created
        :param ns_uri: namespace/ontology URI path
    """
    g = Graph()
    g.parse(ttl_file, format='n3')

    class_name = str(os.path.splitext(os.path.basename(os.path.normpath(py_file)))[0]).upper()

    with open(py_file, 'w', encoding='UTF-8') as f:
        f.write(inspect.cleandoc(f"""
            from rdflib.namespace import DefinedNamespace, Namespace
            from rdflib.term import URIRef


            class {class_name}(DefinedNamespace):
                \"""
                {g.value(URIRef(ns_uri), RDFS.label)}

                Generated from: {ns_uri}
                Date: {datetime.datetime.now()}
                \"""

                _fail = True
        """).strip())

        f.write('\n\n')

        for subject, predicate, object in g.triples((None, RDF.type, URIRef(ns_uri))):
            assert isinstance(predicate, URIRef)
            f.write(
                f"    {subject.split('/')[-1].replace('-', '_').upper()} = URIRef(\"{subject}\")  "
                f"# {' '.join((g.value(subject, DCT.description) or '').splitlines())}\n")

        f.write(f"\n    _NS = Namespace(\"{ns_uri}\")\n")
