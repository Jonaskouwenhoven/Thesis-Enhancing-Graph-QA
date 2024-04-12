from __future__ import annotations

import logging
from rdflib import Graph, Namespace, Literal, RDF, DCTERMS as DCT, URIRef, ConjunctiveGraph
from rdflib.term import Identifier, _is_valid_uri
from typing import Dict

logger = logging.getLogger(__name__)

conj = ConjunctiveGraph()
node_store: Dict[str, GraphNode] = {}
triples = []


class GraphNode:
    type_: URIRef
    ctx: Graph
    ns: Namespace

    def __new__(cls, identifier, *args, **kwargs):
        identifier = identifier.strip()
        if identifier in node_store:
            return node_store[identifier]

        node = super().__new__(cls)
        node_store[identifier] = node
        return node

    def __init__(self, identifier: str):
        identifier = identifier.strip()
        self.uri = self.ns.term(self.uid(identifier))
        self.identifier = identifier.strip()

        self.triple(self.uri, DCT.identifier, identifier)
        self.triple(self.uri, RDF.type, self.type_)

    @staticmethod
    def uid(code: str):
        """
            Normalize and validate OData4 codes to ensure they can be passed as valid URI's

            :param code: OData4 code to validate
            :returns: false if code would yield an invalid URI, the normalized code otherwise
        """
        code = code.replace(' ', '').strip()
        invalid_uri = not code or not _is_valid_uri(code)
        if invalid_uri:
            raise ValueError(f"Invalid identifier for URI given: {code}")
        return code

    def triple(self, sub, pred, obj):
        obj = obj if isinstance(obj, Identifier) else Literal(obj)

        triples.append({'sub': sub, 'pred': pred, 'obj': obj})
        self.ctx.add((sub, pred, obj))
