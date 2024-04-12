from __future__ import annotations

import logging
import os
import pickle
import regex as re
from rdflib import Graph, Namespace, Literal, DCTERMS as DCT, URIRef, DCAT, RDF
from tqdm import tqdm
from typing import Union, List, Optional

import config
from odata_graph.graph_node import conj, GraphNode
from odata_graph.graph_dimension import GraphDimension, DIM_CTX
from odata_graph.graph_measure import GraphMeasure, MSR_CTX
from odata_graph.ttl_to_py.freq_convertor import FREQ_DICT
from odata_graph.ttl_to_py.taxonomie import match_taxon
from utils.global_functions import secure_request
from utils.logical_forms import TABLE, uri_to_code

logger = logging.getLogger(__name__)

TABLE_CTX = Graph(conj.store, URIRef("http://datasets"))


class GraphTable(GraphNode):
    """
        dct:identifier
        dct:modified
        dct:source
        dct:title
        dcat:catalog
        dct:description
        dct:temporal
        dcat:startDate
        dcat:endDate
        dct.accrualPeriodicity
    """
    type_: URIRef = DCAT.Dataset
    ctx: Graph = TABLE_CTX
    ns: Namespace = TABLE.rdf_ns

    def __init__(self,
                 identifier: str,
                 modified: str,
                 source: str,
                 title: Union[Literal],
                 catalog: str,
                 keywords: List[str],
                 table_period: List[str, Optional[str]],
                 accrual_periodicity: URIRef,
                 description: Optional[Literal] = None):
        super().__init__(identifier)

        self.triple(self.uri, DCT.modified, modified)
        self.triple(self.uri, DCT.source, source)
        self.triple(self.uri, DCT.title, title)
        self.triple(self.uri, DCAT.catalog, catalog)

        for keyword in keywords:
            if tax_match := match_taxon(keyword):
                self.triple(self.uri, DCAT.keyword, tax_match)
            else:
                logger.warning(f"Couldn't match keyword {keyword} of table {identifier} to a taxonomy tag.")

        # TODO: dct:spatial

        if len(table_period) > 0:
            self.triple(self.uri, DCT.temporal, f"{table_period[0]}/{table_period[-1]}")
            self.triple(self.uri, DCAT.startDate, table_period[0])  # dates should be in ISO8601 format
            self.triple(self.uri, DCAT.endDate, table_period[-1])

        # TODO: dct:temporalResolution/sdmx:freqDiss not yet provided by OData4
        #  (=frequency of the data in the dataset, i.e. number of bankruptcies on a monthly bases)

        # The frequency at which dataset is published => TODO: surround by try-catch
        self.triple(self.uri, DCT.accrualPeriodicity, accrual_periodicity)

        if description:
            self.triple(self.uri, DCT.description, description)

    @classmethod
    def crawl_table(cls, identifier: str):
        props = secure_request(f"{BASE_URL}/{identifier}/Properties", max_retries=3, timeout=3)

        description = Literal(props['Description'], lang='nl') if props.get('Description') else None
        table_period = re.findall(r'\b[1,2]\d{3}\b', props['TemporalCoverage'])
        if len(table_period) > 1:
            table_period = table_period[::len(table_period) - 1]

        # The frequency at which dataset is published => TODO: surround by try-catch
        accrual_periodicity = FREQ_DICT[props['Frequency']]

        # TODO: use Amunet or smth
        keywords = []

        node = cls(identifier=props['Identifier'], modified=props['Modified'], source=props['Catalog'],
                   title=Literal(props['Title'], lang='nl'), catalog=props['Catalog'], description=description,
                   keywords=keywords, table_period=table_period, accrual_periodicity=accrual_periodicity)

        GraphMeasure.crawl_Measure(node)
        GraphDimension.crawl_dimensions(node)

BASE_URL = 'https://odata4.cbs.nl/CBS'
table_list = secure_request(f"{BASE_URL}/datasets", max_retries=3, timeout=3)

if not table_list:
    raise ValueError("No tables could be fetched from OData4.")

OVERWRITE = True
REPOSITORY = config.GRAPH_DB_REPO

# Load in existing graphs to append to
graphs = [
    ('./odata_graph/500_graph/datasets.trig', TABLE_CTX),
    ('./odata_graph/500_graph/onderwerpen.trig', MSR_CTX),
    ('./odata_graph/500_graph/dimensions.trig', DIM_CTX),
]

for file_name, ctx in graphs:
    if os.path.exists(file_name):
        print(f"Local file found. Loading graph from {file_name}. "
              f"Depending on size this can take a while.")
        ctx.parse(file_name, format='trig')

with open('./data/annotated_31_jan_with_expanded_query.pkl', 'rb') as file:
    annotated_data = pickle.load(file)

failed_tables_path = './odata_graph/500_graph/failed_tables.txt'
failed_tables = []
if os.path.exists(failed_tables_path):
    with open(failed_tables_path, 'r') as file:
        failed_tables = list(set(file.read().splitlines()))

fetch_tables = ([t['Identifier'] for t in table_list['value'] if t['Language'] == 'nl' and
                t['Identifier'] in annotated_data['table_id'].unique() and
                t['Identifier'] not in {uri_to_code(t) for t in TABLE_CTX.subjects(RDF.type, DCAT.Dataset)}] +
                failed_tables)
for table_id in tqdm(set(fetch_tables), desc="*Prrt prrrt* Generating RDF triples for tables...",
                     bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}'):
    try:
        GraphTable.crawl_table(table_id)

        # Remove table from failed_tables once succeeded
        if os.path.exists(failed_tables_path):
            with open(failed_tables_path, "r+") as f:
                new_f = f.readlines()
                f.seek(0)
                for line in new_f:
                    if table_id not in line:
                        f.write(line)
                f.truncate()
    except Exception as e:
        logger.error(f"Failed to generate RDF triples for table {table_id}: {e}")
        with open(failed_tables_path, 'a+') as file:
            file.write(f"{table_id}\n")

    # Store intermediate result
    for file_name, ctx in graphs:
        with open(file_name, 'wb') as file:
            file.write(ctx.serialize(format='trig').encode())
