from __future__ import annotations

import logging
from rdflib import Graph, Namespace, Literal, RDF, DCTERMS as DCT, URIRef, QB, SKOS
from typing import List, Optional

from odata_graph.graph_node import conj, GraphNode
from odata_graph.ttl_to_py.begrippen import match_definition
from odata_graph.ttl_to_py.taxonomie import match_taxon
from odata_graph.ttl_to_py.geo_gebieden import match_geo
from utils.global_functions import secure_request
from utils.logical_forms import DIM

logger = logging.getLogger(__name__)

DIM_CTX = Graph(conj.store, URIRef("http://dimensies"))
SDMX_DIM = Namespace("http://purl.org/linked-data/sdmx/2009/dimension#")

BASE_URL = 'https://odata4.cbs.nl/CBS'  # TODO: put in config file


class GraphDimension(GraphNode):
    """
       dct:identifier
       skos:prefLabel
       skos:altLabel
    """
    type_: URIRef = QB.DimensionProperty
    ctx: Graph = DIM_CTX
    ns: Namespace = DIM.rdf_ns

    def __init__(self,
                 identifier: str,
                 pref_label: Literal,
                 alt_labels: List[Literal],
                 description: Optional[str] = None,  # to be converted to cbs:Begrip
                 kind: Optional[str] = None):
        super().__init__(identifier)

        self.triple(self.uri, SKOS.prefLabel, pref_label)
        for lbl in alt_labels:
            self.triple(self.uri, SKOS.altLabel, lbl)
            if tax_match := match_taxon(lbl):
                self.triple(self.uri, SKOS.closeMatch, tax_match)

        if tax_match := match_taxon(pref_label):
            self.triple(self.uri, SKOS.closeMatch, tax_match)

        if description:
            self.triple(self.uri, SKOS.definition, Literal(description, lang='nl'))
            if tax_def := match_definition(description):
                self.triple(self.uri, SKOS.definition, tax_def)

        if kind == 'TimeDimension':
            # TODO: do something official like dct:spatial or dct:temporal instead of using Literal
            self.triple(self.uri, RDF.type, 'TimeDimension')
        if kind in ['GeoDimension', 'GeoDetailDimension'] or identifier in ['RegioS2007']:
            if geo_uri := match_geo(identifier.strip()):
                self.triple(self.uri, SDMX_DIM.refArea, geo_uri)

    def connect_with_table(self, table: URIRef):
        self.triple(table, QB.dimension, self.uri)
        self.triple(self.uri, DCT.isPartOf, table)

    def add_child(self, child: URIRef):
        self.triple(child, SKOS.broader, self.uri)  # "broader" should read here as "has broader concept"
        self.triple(self.uri, SKOS.narrower, child)

    @staticmethod
    def crawl_dimensions(table_node: GraphNode):
        dims = secure_request(f"{BASE_URL}/{table_node.identifier}/Dimensions", max_retries=3, timeout=3)['value']

        # Get list of dimensions
        table_dimensions = []
        for dim in dims:
            dim['SubDimensions'] = []
            if dim['ContainsGroups'] and dim['ContainsCodes']:
                dim['Groups'] = []
                dimension_groups = secure_request(dim['GroupsUrl'], max_retries=3, timeout=3)['value']
                dimension_codes = secure_request(dim['CodesUrl'], max_retries=3, timeout=3)['value']
                for group in dimension_groups:
                    group['SubDimensions'] = []
                    group_id = group['Id']
                    for code in dimension_codes:
                        if not code['DimensionGroupId']:
                            dim['SubDimensions'].append(code)
                        if group_id == code['DimensionGroupId']:
                            group['SubDimensions'].append(code)
                    dim['Groups'].append(group)
            elif dim['ContainsCodes']:
                dimension_codes = secure_request(dim['CodesUrl'], max_retries=3, timeout=3)['value']
                dim['SubDimensions'] = []
                for code in dimension_codes:
                    dim['SubDimensions'].append(code)
            table_dimensions.append(dim)

        for dim in table_dimensions:
            description = dim['Description'].strip() if dim.get('Description') else None

            dim_node = GraphDimension(identifier=dim['Identifier'],
                                      pref_label=Literal(dim['Title'], lang='nl'),
                                      alt_labels=[], description=description, kind=dim.get('Kind'))

            # Connect measure to corresponding table
            dim_node.connect_with_table(table_node.uri)

            # Connect parents with child
            codes = dim['SubDimensions']  # Codes hanging directly under dimension (without group)
            for group in dim.get('Groups', []):
                for subdim in group['SubDimensions']:
                    codes.append(subdim)

            for code in codes:
                try:
                    child_node = GraphDimension(identifier=code['Identifier'],
                                                pref_label=Literal(code['Title'], lang='nl'),
                                                alt_labels=[], description=code['Description'],
                                                kind=dim.get('Kind'))
                    dim_node.add_child(child_node.uri)
                except (AssertionError, ValueError, KeyError):
                    logger.warning(f"Can't add {code['Identifier']} as child to dimension {dim_node.identifier}!")
