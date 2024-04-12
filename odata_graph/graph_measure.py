from __future__ import annotations

import logging
import re

from rdflib import Graph, Namespace, Literal, DCTERMS as DCT, URIRef, QB, SKOS, RDF
from typing import List, Optional

from odata_graph.graph_node import conj, GraphNode
from odata_graph.sparql_controller import SCOT
from odata_graph.ttl_to_py.begrippen import match_definition
from odata_graph.ttl_to_py.taxonomie import match_taxon
from odata_graph.ttl_to_py.unit_convertor import UNIT_DICT, UNIT, MULTIPLIER, UNIT_OF_SYSTEM
from utils.global_functions import secure_request
from utils.logical_forms import MSR

logger = logging.getLogger(__name__)

MSR_CTX = Graph(conj.store, URIRef("http://onderwerpen"))

BASE_URL = 'https://odata4.cbs.nl/CBS'  # TODO: put in config file


class GraphMeasure(GraphNode):
    """
       dct:identifier
       qudt:unitOfSystem
       qudt:conversionMultiplier
       skos:prefLabel
       skos:altLabel
    """
    type_: URIRef = QB.MeasureProperty
    ctx: Graph = MSR_CTX
    ns: Namespace = MSR.rdf_ns

    def __init__(self,
                 identifier: str,
                 pref_label: Literal,
                 alt_labels: List[Literal],
                 total: bool = False,
                 unit_of_system: Optional[str] = None,
                 conversion_multiplier: Optional[str] = None,
                 unit: Optional[URIRef] = None,
                 description: Optional[str] = None  # to be converted to cbs:Begrip
                 ):
        super().__init__(identifier)

        self.triple(self.uri, SKOS.prefLabel, pref_label)
        for lbl in alt_labels:
            self.triple(self.uri, SKOS.altLabel, lbl)
            if tax_match := match_taxon(lbl):
                self.triple(self.uri, SKOS.closeMatch, tax_match)

        if tax_match := match_taxon(pref_label):
            self.triple(self.uri, SKOS.closeMatch, tax_match)

        if total:
            self.triple(self.uri, RDF.type, SCOT.Total)

        if unit_of_system is not None:
            self.triple(self.uri, UNIT_OF_SYSTEM, unit_of_system)

        if conversion_multiplier is not None:
            self.triple(self.uri, MULTIPLIER, conversion_multiplier)

        if description:
            self.triple(self.uri, SKOS.definition, Literal(description, lang='nl'))
            if tax_def := match_definition(description):
                self.triple(self.uri, SKOS.definition, tax_def)

        if unit is not None:
            self.triple(self.uri, UNIT, unit_of_system)

    def connect_with_table(self, table: URIRef):
        self.triple(table, QB.measure, self.uri)
        self.triple(self.uri, DCT.isPartOf, table)

    def add_parent(self, parent: URIRef):
        self.triple(self.uri, SKOS.broader, parent)  # "broader" should read here as "has broader concept"
        self.triple(parent, SKOS.narrower, self.uri)

    @staticmethod
    def crawl_Measure(table_node: GraphNode):
        table_url = f"{BASE_URL}/{table_node.identifier}"
        table_parts = [m['name'] for m in secure_request(f"{table_url}", max_retries=3, timeout=3)['value']]

        if 'MeasureCodes' in table_parts:
            measure_codes: List[dict] = secure_request(f"{table_url}/MeasureCodes", max_retries=3, timeout=3)['value']
            for msr in measure_codes:
                try:
                    description = msr['Description'].strip() if msr.get('Description') else None

                    unit, multiplier = None, None
                    if msr.get('Unit') and (u_dict := UNIT_DICT.get(msr['Unit'], False)):
                        unit = u_dict.get('unit')
                        multiplier = u_dict.get('multiplier')

                    # TODO: define totality
                    total = False
                    if re.match(r".*\b(totaal|waarde|totale)\b.*", msr['Title'], flags=re.IGNORECASE):
                        total = True

                    msr_node = GraphMeasure(identifier=msr.get('Identifier'),
                                            pref_label=Literal(msr['Title'], lang='nl'),
                                            alt_labels=[], description=description, unit_of_system=msr.get('Unit'),
                                            unit=unit, conversion_multiplier=multiplier, total=total)

                    # Connect measure to corresponding table
                    msr_node.connect_with_table(table_node.uri)
                except ValueError:
                    logger.warning(f"Measure {msr.get('Identifier')} for table_id {table_node.identifier} is not a valid id!")
                    continue

                # Add relations between measure code and parent measure group
                if parent_id := msr.get('MeasureGroupId'):
                    try:
                        parent_uri = GraphMeasure.ns.term(GraphNode.uid(parent_id))
                        msr_node.add_parent(parent_uri)
                    except (AssertionError, ValueError, KeyError):
                        logger.warning(f"Can't add {msr['MeasureGroupId']} as parent to measure {msr_node.identifier}!")

        if 'MeasureGroups' in table_parts:
            measure_groups = secure_request(f"{table_url}/MeasureGroups", max_retries=3, timeout=3)['value']
            for group in measure_groups:
                try:
                    description = group['Description'].strip() if group.get('Description') else None
                    group_node = GraphMeasure(identifier=group['Id'], pref_label=Literal(group['Title'], lang='nl'),
                                              alt_labels=[], description=description)
                except ValueError:
                    logger.warning(f"MeasureGroup {group['Id']} for table_id {table_node.identifier} is not a valid id!")
                    continue

                # Add relations between group node and parent measure groups if applicable
                if parent_id := group.get('ParentId'):
                    try:
                        parent_uri = GraphMeasure.ns.term(GraphNode.uid(parent_id))
                        group_node.add_parent(parent_uri)
                    except (AssertionError, ValueError, KeyError):
                        logger.warning(f"Can't add {group['ParentId']} as parent to measure group {group_node.identifier}!")
