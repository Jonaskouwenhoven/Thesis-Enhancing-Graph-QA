import logging
import os.path
import textwrap
from typing import Union

from SPARQLWrapper import SPARQLWrapper, JSON, RDFXML
from rdflib import Graph, ConjunctiveGraph, Namespace

import config
from utils.logical_forms import TABLE, MSR, DIM

logger = logging.getLogger(__name__)

SCOT = Namespace("http://statistics.gov.scot/def/dimension/")  # Great scot!
QUDT = Namespace("http://qudt.org/2.1/schema/qudt/")


class SparqlEngine(object):
    def __init__(self, local=True, oxigraph=True):
        """
            Object for handling SPARQL request to the graph. The graph can run either locally
            or through a GraphDB connection.

            :param local: use local graph file if available
            :param oxigraph: use Rust bindings to handle local graph. Does not work in
                             multiprocessing mode as Rust/PyO3 objects cannot be pickled.
        """
        if local and os.path.exists(config.GRAPH_FILE):
            logger.info(f"Local file found. Loading graph from {config.GRAPH_FILE}. "
                        f"Depending on size this can take a while.")
            self.graph = Graph(store="Oxigraph") if oxigraph else Graph()
            self.graph.parse(config.GRAPH_FILE, format='ttl')

            self.local = True
            logger.info("SPARQL SparqlEngine is running from local file!")
        else:
            REPOSITORY = config.GRAPH_DB_REPO
            self.sparql = SPARQLWrapper(f"http://4.231.28.187:7200/repositories/{REPOSITORY}")
            self.sparql.setCredentials("cerborix", "2Jt@d&M5g")  # TODO: make proper connection with secrets/env vars

            self.local = False
            logger.info("SPARQL SparqlEngine is running from GraphDB!")

    def select(self, query: str) -> list:
        if self.local:
            result = self.graph.query(query)
            return list(result)
        else:
            self.sparql.setQuery(query)
            self.sparql.setReturnFormat(JSON)
            self.sparql.method = 'GET'

            try:
                res = self.sparql.queryAndConvert()
                return res['results']['bindings']
            except Exception as e:
                logger.error(f"Failed to perform SELECT query: {e}")

    def construct(self, query: str) -> Union[ConjunctiveGraph|Graph]:
        if self.local:
            result = self.graph.query(query)
            return result.graph
        else:
            self.sparql.setQuery(query)
            self.sparql.setReturnFormat(RDFXML)
            self.sparql.method = 'GET'
            try:
                g = self.sparql.queryAndConvert()  # subgraph following from candidate nodes
                return g
            except Exception as e:
                logger.error(f"Failed to perform CONSTRUCT query: {e}")

    def explode_subgraph(self, nodes: list, table_cutoff: int = 5, verbose=False) -> Graph:
        """
            Construct a subgraph from the GraphDB based on a list of given table, measure and
            dimension nodes. The graph returned will contain all the measure/dimension-table
            relations and the corresponding metadata of and hierarchies between all nodes.

            :param nodes: list of URI's (table, measure or dimension) of nodes to explode
            :param table_cutoff: TODO
            :param verbose: TODO
            :returns: subgraph
        """
        explode_tables = [n for n in nodes if n in TABLE.rdf_ns][:table_cutoff]  # TODO: with 5+ tables exploding the subgraph takes a long time
        table_filter = ("?s IN (<" + '>, <'.join(explode_tables) + ">)") if explode_tables else ""
        explode_obs = [n for n in nodes if n in MSR.rdf_ns or n in DIM.rdf_ns][:10]  # TODO: request header becomes too large with too many observation nodes. 'Use scroll api'
        obs_filter = ("?o IN (<" + '>, <'.join(explode_obs) + ">)") if explode_obs else ""

        query = (f"""
            PREFIX qb: <http://purl.org/linked-data/cube#>
            PREFIX dct: <http://purl.org/dc/terms/>
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
            PREFIX scot: <http://statistics.gov.scot/def/dimension/>
            PREFIX qudt: <http://qudt.org/2.1/schema/qudt/>
    
            CONSTRUCT {{ 
                ?s ?p ?obs ;                                                            # return table-msr/dim triples
                   dct:title ?t_title .
                ?obs ?hierarchy ?group ;                                                # return dim hierarchy triples
                     ?has_type ?type ;                                                  # return optional TC/GC indication
                     skos:prefLabel ?obs_label ;
                     ?has_unit ?unit .                                                  # return optional units for Measure
            }} WHERE {{
                {{                                                                      # get all tables and their MSRs/DIMs
                    SELECT DISTINCT ?s WHERE {{
                        ?s qb:measure|qb:dimension ?o .
                        FILTER NOT EXISTS {{                                            # don't explode time and geo dims
                            VALUES ?dim {{'TimeDimension' 'GeoDimension'}}
                            ?o a ?dim .
                        }}
                        FILTER ({table_filter} {'||' if explode_tables and explode_obs else ''}     # select specific table relevant triples
                                {obs_filter}) .                                                     # select specific MSR/DIM triples
                    }}
                }}
                ?s ?p ?obs                                                              # explode!
                FILTER (?p = qb:measure || ?p = qb:dimension) .
                ?s dct:title ?t_title .                                                 # fetch titles and labels for all subjects
                ?obs skos:prefLabel ?obs_label .
                FILTER (?p = qb:measure || ?p = qb:dimension) .
                FILTER NOT EXISTS {{
                    VALUES ?type {{'TimeDimension' 'GeoDimension'}}
                    ?obs a ?type ;
                         skos:broader ?d .                                              # only get dim groups for time and geo dims
                }}
                OPTIONAL {{
                    ?obs ?hierarchy ?group                                              # get hierarchy between dimensions (broader)
                    FILTER (?hierarchy = skos:broader || ?hierarchy = skos:narrower)
                    FILTER NOT EXISTS {{                                                # don't get hierarchy for time and geo dims
                        VALUES ?type {{'TimeDimension' 'GeoDimension'}}
                        ?obs a ?type .
                    }}
                    FILTER EXISTS {{ 
                        ?obs a qb:DimensionProperty .
                        ?s ?p ?group .                                                  # only get hierarchy for dimension relevant to tables in subgraph
                    }}
                }}
                OPTIONAL {{                                                             # get notion if dim group is TC or GC
                    VALUES ?type {{'TimeDimension' 'GeoDimension' 
                                   scot:Total scot:confidenceInterval}}
                    ?obs a ?type .
                    ?obs ?has_type ?type .
                }}
                OPTIONAL {{                                                             # get OData4 units of measure (non-standardised)
                    ?obs qudt:unitOfSystem ?unit .
                    ?obs ?has_unit ?unit .
                }}
            }}
        """)

        if verbose:
            logger.debug(textwrap.dedent(query))

        return self.construct(query)

    def construct_node_graph(self, nodes: list, table_cutoff: int = 5, verbose=False) -> Graph:
        """
            Construct a graph containing only the given nodes.

            :param nodes: list of URIs of nodes to include in the graph
            :returns: subgraph
        """
        explode_tables = [n for n in nodes if n in TABLE.rdf_ns][:table_cutoff]
        table_filter = ("?s IN (<" + '>, <'.join(explode_tables) + ">)") if explode_tables else ""
        explode_obs = [n for n in nodes if n in MSR.rdf_ns or n in DIM.rdf_ns][:20]
        obs_filter = ("?o IN (<" + '>, <'.join(explode_obs) + ">)") if explode_obs else ""

        query = (f"""
            CONSTRUCT {{
                ?s ?p ?o .
            }} WHERE {{
                ?s ?p ?o .
                FILTER ({table_filter} {'||' if explode_tables and explode_obs else ''}
                        {obs_filter}) .
            }}
        """)

        if verbose:
            logger.debug(textwrap.dedent(query))

        return self.construct(query)

    def explode_subgraph_msr_dims_only(self, nodes: list, table_cutoff: int = 5, verbose=False) -> Graph:
        """
            Construct a subgraph from the GraphDB based on a list of given table, measure and
            dimension nodes. The graph returned will contain all the measure/dimension-table
            relations. This function omits all hierarchical relations between Measure and
            dimensions, labels and units! Use `explode_subgraph()` to obtain the full graph if
            needed.

            :param nodes: list of URI's (table, measure or dimension) of nodes to explode
            :returns: subgraph
        """
        explode_tables = [n for n in nodes if n in TABLE.rdf_ns][:table_cutoff]
        table_filter = ("?s IN (<" + '>, <'.join(explode_tables) + ">)") if explode_tables else ""
        explode_obs = [n for n in nodes if n in MSR.rdf_ns or n in DIM.rdf_ns][:20]
        obs_filter = ("?o IN (<" + '>, <'.join(explode_obs) + ">)") if explode_obs else ""

        query = (f"""
            PREFIX qb: <http://purl.org/linked-data/cube#>
    
            CONSTRUCT {{
                ?s ?p ?obs .                                                            # return table-msr/dim triples
            }} WHERE {{
                {{                                                                      # get all tables and their MSRs/DIMs
                    SELECT DISTINCT ?s WHERE {{
                        ?s qb:measure|qb:dimension ?o .
                        FILTER NOT EXISTS {{                                            # don't explode time and geo dims
                            VALUES ?dim {{'TimeDimension' 'GeoDimension'}}
                            ?o a ?dim .
                        }}
                        # select specific table relevant triples and select specific MSR/DIM triples
                        FILTER ({table_filter} {'||' if explode_tables and explode_obs else ''}
                                {obs_filter}) .
                    }}
                }}
                ?s ?p ?obs                                                              # explode!
                FILTER (?p = qb:measure || ?p = qb:dimension)
                FILTER NOT EXISTS {{                                                    # don't return time/geo dims
                    VALUES ?dim {{'TimeDimension' 'GeoDimension'}}
                    ?obs a ?dim .
                }}
            }}
        """)

        if verbose:
            logger.debug(textwrap.dedent(query))

        return self.construct(query)

    def get_table_geo_dims(self, table: TABLE) -> dict:
        query = (f"""
            PREFIX qb: <http://purl.org/linked-data/cube#>
            PREFIX dct: <http://purl.org/dc/terms/>
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
            
            SELECT DISTINCT ?id ?prefLabel WHERE {{ 
                <{table.uri}> qb:dimension ?dim .
                ?dim dct:identifier ?id ;
                     skos:prefLabel ?prefLabel .
                FILTER (REGEX(?id, "^(?!(BU|WK)).*$")) .  # TODO: for now skip BU en WK dimension to avoid overflow
                FILTER EXISTS {{
                    ?dim a 'GeoDimension' ;
                         skos:broader ?d .
                }}
            }}
        """)

        res = self.select(query)
        return {d['id']['value']: d['prefLabel']['value'] for d in res}

    def get_table_graph(self, table: TABLE) -> Graph:
        """
            Construct a subgraph from the GraphDB based a single given table. The graph
            returned will contain all the measure/dimension-table  relations and the
            corresponding metadata of and hierarchies between all nodes.

            :param table: table node to construct graph for
            :returns: subgraph
        """
        query = (f"""
            PREFIX qb: <http://purl.org/linked-data/cube#>
            PREFIX dct: <http://purl.org/dc/terms/>
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
            PREFIX scot: <http://statistics.gov.scot/def/dimension/>
            PREFIX qudt: <http://qudt.org/2.1/schema/qudt/>
        
            CONSTRUCT {{ 
                ?s ?p ?obs ;                                                            # return table-msr/dim triples
                   dct:title ?t_title .
                ?obs ?hierarchy ?group ;                                                # return dim hierarchy triples
                     ?has_type ?type ;                                                  # return optional TC/GC indication
                     skos:prefLabel ?obs_label ;
                     ?has_unit ?unit .                                                  # return optional units for Measure
            }} WHERE {{
                BIND (<{table.uri}> AS ?s) .
                ?s ?p ?obs
                FILTER (?p = qb:measure || ?p = qb:dimension) .
                FILTER NOT EXISTS {{
                    VALUES ?type {{'TimeDimension' 'GeoDimension'}}
                    ?obs a ?type ;
                         skos:broader ?d .                                              # only get dim groups for time and geo dims
                }}
                ?s dct:title ?t_title .
                ?obs skos:prefLabel ?obs_label .
                OPTIONAL {{
                    ?obs ?hierarchy ?group                                              # get hierarchy between dimensions (broader)
                    FILTER (?hierarchy = skos:broader || ?hierarchy = skos:narrower)
                    FILTER NOT EXISTS {{                                                # don't get hierarchy for time and geo dims
                        VALUES ?type {{'TimeDimension' 'GeoDimension'}}
                        ?obs a ?type .
                    }}
                    FILTER EXISTS {{ 
                        ?obs a qb:DimensionProperty .
                        ?s ?p ?group .                                                  # only get hierarchy for dimension relevant to tables in subgraph
                    }}
                }}
                OPTIONAL {{                                                             # get notion if dim group is TC or GC
                    VALUES ?type {{'TimeDimension' 'GeoDimension' 
                                   scot:Total scot:confidenceInterval}}
                    ?obs a ?type .
                    ?obs ?has_type ?type .
                }}
                OPTIONAL {{                                                             # get OData4 units of measure (non-standardised)
                    ?obs qudt:unitOfSystem ?unit .
                    ?obs ?has_unit ?unit .
                }}
            }}
        """)

        return self.construct(query)

    def get_table_time_dims(self, table: TABLE) -> dict:
        query = (f"""
            PREFIX qb: <http://purl.org/linked-data/cube#>
            PREFIX dct: <http://purl.org/dc/terms/>
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    
            SELECT DISTINCT ?id ?prefLabel WHERE {{ 
                <{table.uri}> qb:dimension ?dim .
                ?dim dct:identifier ?id ;
                     skos:prefLabel ?prefLabel .
                FILTER EXISTS {{
                    ?dim a 'TimeDimension' ;
                         skos:broader ?d .
                }}
            }}
        """)

        res = self.select(query)
        return {d['id']['value']: d['prefLabel']['value'] for d in res}
