import faiss
import json
import logging
import numpy as np
import re
import string
from collections import OrderedDict
from nltk.tokenize import word_tokenize
from rdflib import Graph, QB, URIRef
from sentence_transformers import SentenceTransformer
from typing import Tuple, Optional
from uuid import uuid4

import config
from odata_graph.sparql_controller import SparqlEngine
from pipeline.query_expander import expand_query
import pandas as pd
import pipeline.retrieveandscore as rs
from utils.logical_forms import uri_to_code, TABLE, MSR, DIM

logger = logging.getLogger(__name__)

import faiss
import json
import logging
import numpy as np
import re
import string
from collections import OrderedDict
from nltk.tokenize import word_tokenize
from rdflib import Graph, QB, URIRef
from sentence_transformers import SentenceTransformer
from typing import Tuple, Optional
from uuid import uuid4

import config
from odata_graph.sparql_controller import SparqlEngine
from pipeline.query_expander import expand_query
import pandas as pd
import pipeline.retrieveandscore as rs
from utils.logical_forms import uri_to_code, TABLE, MSR, DIM

evaltable_df = pd.read_pickle("data/tabledf.pkl")
evaltable_df = evaltable_df.drop_duplicates(subset=['table_id'])
def retrieve_Tables_BM25(query, k=25, full_graph: bool = False, verbose=False):
    """
        Find all candidate entities corresponding with the query in the graph using the Elastic index. The result
        will be the candidate Measure and dimensions with their corresponding matching scores, and the resulting
        subgraph of all the connected tables and their connected Measure and dimensions from these candidates.

        :param query: NL search query
        :param k: number of closest ES-nodes to consider
        :param full_graph: fetch full graph, including dimension hierarchy relations (slower)
        :param verbose: print the executed SPARQL query
        :returns: (Subgraph, {candidate Measure: scores}, {candidate dims: scores})
    """
    query = [w for w in word_tokenize(query) if w not in string.punctuation]
    query = query + expand_query(query)

    # TODO: get top k-nodes, explode and ask ES separately for all BM25 codes per node in the subgraph
    q = {
        "track_total_hits": False,
        "_source": ["_id"],
        "min_score": 0.01,
        "size": k,
        "query": {
            "match": {
                "body": {
                    "minimum_should_match": 1,
                    "query": ' '.join(query)
                }
            }
        }
    }



    resp = es.search(body=q, index='graph-nl')
    er_nodes = [n['_id'] for n in resp.body['hits']['hits']]
    er_scores =  [n['_score'] for n in resp.body['hits']['hits']]
    # er_nodes = [n.split("/")[-1] for n in er_nodes]

    ids, scores  = [], []
    for id, score in zip(er_nodes, er_scores):
        if "dataset" in id:
            id_split = id.split("/")[-1]
            if id_split in evaltable_df['table_id'].to_list():
                ids.append(id_split)
                scores.append(score)

    
            



    data_frame = pd.DataFrame({"id": ids, "score": scores})
    data_frame['score'] = data_frame['score']/data_frame['score'].max()
    if len(er_nodes) == 0:
        raise ValueError("No nodes found in Elastic that correspond with the given query.")

    # Get all (table, measure/dimension, entity) combinations following from the relations of the candidate ER nodes
    return data_frame[:10]




class EntityRetriever:
    def __init__(self, engine: SparqlEngine, colbert = False, bm25 = False):
        self.engine = engine
        self.colbert = colbert
        self.bm25 = bm25
        self.retriever = rs.RetrieverInitializer().initialize_Colbert() if colbert else None


    def get_candidate_nodes_better(self, query, target=None, full_graph: bool = False, verbose=False) -> \
            Tuple[OrderedDict, Optional[Graph]]:
        # query = [w for w in word_tokenize(query) if w not in string.punctuation]
        # query = ' '.join(query + expand_query(query))
        # query = query.lower()
        query_embedding = rs.sentence_transformer_encode(query.lower())

        if self.retriever:
            tables = rs.retrieve_top_tables_colbert(query, self.retriever, n_tables=10)

        elif self.bm25: 
            tables = retrieve_Tables_BM25(query, k=10, full_graph=False, verbose=False)
        else:

            tables = rs.retrieve_top_tables(query_embedding, n_tables=10)
        tablesret = tables
        # reranked_tables = rs.reranked(tables, query_embedding)
        # reranked_tables['combined_score'] = reranked_tables['measure_score'] * 0.1 + reranked_tables['score'] * 0.9
        # reranked_tables = reranked_tables.sort_values(by=['combined_score'], ascending=False)[:10]  # TODO: magic number
        # # reranked_tables = reranked_tables.sort_values(by=['measure_score', 'score'], ascending=False)[:10]  # TODO: magic number

        # tables = reranked_tables
            #     reranked_tables.loc[reranked_tables['table_id'] == golden_table, 'measure_score'] = 1.

        er_nodes = {t['id']: {
            'score': t['score'],  # will be used for the prior multiplications in the decoder
            'matched_words': [uuid4()]
        } for _, t in tables.iterrows()}
        
        tables['rel'] = tables['score']
        retrieved_dataframes = [tables[['id', 'score']]]
        total_dicts = []
        for index, row in tables[:].iterrows():
            
            
            table_uri = row['id'].split('/')[-1]
            Measure  = rs.retrieve_Measure(query, query_embedding, table_uri, with_threshold=False)
            retrieved_dataframes.append(Measure[['id', 'rel']].rename(columns={"rel": "score"}))
            dimensions = rs.retrieve_dimensions(query, query_embedding, table_uri, with_threshold=False)

            for dim in dimensions:
                retrieved_dataframes.append(dim[['id', 'score']])

            table_dict = rs.create_table_dict(table_uri,Measure, dimensions)
            total_dicts.append(table_dict)

            Measure_cleaned = Measure[['id', 'rel', 'desc']]
            Measure_cleaned['type'] = 'measure'

            msr_dims_list = [Measure_cleaned]
            for d in dimensions:
                d_cleaned = d[['id', 'score', 'desc']]
                d_cleaned.rename(columns={'score': 'rel'}, inplace=True)
                d_cleaned['type'] = 'dimension'

                msr_dims_list.append(d_cleaned)

            s = pd.concat(msr_dims_list, ignore_index=True)
            # TODO: make sure no duplicates are returned in the first place
            s = s.drop_duplicates(subset=['id'])
            msr_dims = s.drop_duplicates().set_index('id', drop=True).to_dict(orient='index')
            er_nodes |= {(DIM if v['type'] == 'dimension' else MSR).rdf_ns.term(k): {
                'score': v['rel'],
                'matched_words': [uuid4()]
            } for k, v in msr_dims.items()}
        
        er_nodes = dict(sorted(er_nodes.items(), key=lambda x: x[1]['score'], reverse=True))
        # EXPLODE!
        try:
            expl_fun = self.engine.explode_subgraph
            g = expl_fun(list(er_nodes.keys()), verbose=verbose)
        except Exception as e:
            g = None
            print(f"Its not working!! \n{e}")
            logger.error(f"Failed to perform subgraph exploding: {e}")


        ranked_nodes: OrderedDict[str, Tuple[np.array, OrderedDict[str, np.array]]] = OrderedDict()
        for _, row in tables.iterrows():
            table_uri = row['id']
            ranked_nodes[str(table_uri)] = (
                row['score'],  # will be used for the prior multiplications in the decoder
                OrderedDict(dict(sorted({
                    **{str(m): er_nodes[m]['score'] for m in set(g.objects(table_uri, QB.measure)) if
                       m in er_nodes},
                    **{str(d): er_nodes[d]['score'] for d in set(g.objects(table_uri, QB.dimension)) if
                       d in er_nodes}
                }.items(), key=lambda x: x[1], reverse=True)))
            )

        return ranked_nodes, g, total_dicts, pd.concat(retrieved_dataframes), tablesret

    def get_candidate_nodes_baselines(self, query, target=None, full_graph: bool = False, verbose=False) -> \
        Tuple[OrderedDict, Optional[Graph]]:
        # query = [w for w in word_tokenize(query) if w not in string.punctuation]
        # query = ' '.join(query + expand_query(query))
        query = query.lower()
        query_embedding = rs.sentence_transformer_encode(query)

        tables = rs.retrieve_top_tables(query_embedding, n_tables=10)
        tables['rel'] = tables['score']
        reranked_tables = rs.reranked(tables, query_embedding)
        reranked_tables = reranked_tables.sort_values(by=['measure_score', 'rel'], ascending=False)[:10]  # TODO: magic number

        reranked_tables['score'] = reranked_tables['measure_score'] * 0.2 + reranked_tables['rel'] * 0.8
# 
        reranked_tables = reranked_tables.sort_values(by=['score'], ascending=False)[:10]
            #     reranked_tables.loc[reranked_tables['table_id'] == golden_table, 'measure_score'] = 1.

        er_nodes = {t['id']: {
            'score': t['score'],  # will be used for the prior multiplications in the decoder
            'matched_words': [uuid4()]
        } for _, t in tables.iterrows()}
        
        s_total = {}
        total_dicts = []
        for index, row in tables[:1].iterrows():
            
            
            table_uri = row['id'].split('/')[-1]
            Measure  = rs.retrieve_Measure(query, query_embedding, table_uri)
            dimensions = rs.retrieve_dimensions(query, query_embedding, table_uri)
            table_dict = rs.create_table_dict(table_uri,Measure, dimensions)
            total_dicts.append(table_dict)

            Measure_cleaned = Measure[['id', 'rel', 'desc']]
            Measure_cleaned['type'] = 'measure'

            msr_dims_list = [Measure_cleaned]
            for d in dimensions:
                d_cleaned = d[['id', 'score', 'desc']]
                d_cleaned.rename(columns={'score': 'rel'}, inplace=True)
                d_cleaned['type'] = 'dimension'

                msr_dims_list.append(d_cleaned)

            s = pd.concat(msr_dims_list, ignore_index=True)
            s_total[table_uri] = s
            # TODO: make sure no duplicates are returned in the first place
            s = s.drop_duplicates(subset=['id'])
            msr_dims = s.drop_duplicates().set_index('id', drop=True).to_dict(orient='index')
            er_nodes |= {(DIM if v['type'] == 'dimension' else MSR).rdf_ns.term(k): {
                'score': v['rel'],
                'matched_words': [uuid4()]
            } for k, v in msr_dims.items()}
        
        # er_nodes = dict(sorted(er_nodes.items(), key=lambda x: x[1]['score'], reverse=True))
        # EXPLODE!
        # try:
        #     expl_fun = self.engine.explode_subgraph_msr_dims_only if not full_graph else self.engine.explode_subgraph
        #     g = expl_fun(list(er_nodes.keys()), verbose=verbose)
        # except Exception as e:
        #     g = None
        #     print(f"Its not working!! \n{e}")
        #     logger.error(f"Failed to perform subgraph exploding: {e}")

        expl_fun = self.engine.explode_subgraph_msr_dims_only if not full_graph else self.engine.explode_subgraph
        g = expl_fun(list(er_nodes.keys()), verbose=verbose)

        tables = set(g.subjects(QB.measure, None))
        Measure = set(g.objects(None, QB.measure))
        dims = set(g.objects(None, QB.dimension))

        return (g,
                {uri_to_code(t): er_nodes[t] for t in tables if t in er_nodes},
                {uri_to_code(m): er_nodes[m] for m in Measure if m in er_nodes},
                {uri_to_code(d): er_nodes[d] for d in dims if d in er_nodes}, total_dicts)

        # print(er_nodes)
        # ranked_nodes: OrderedDict[str, Tuple[np.array, OrderedDict[str, np.array]]] = OrderedDict()
        # for _, row in reranked_tables.iterrows():
        #     table_uri = row['id']
        #     ranked_nodes[str(table_uri)] = (
        #         row['score'],  # will be used for the prior multiplications in the decoder
        #         OrderedDict(dict(sorted({
        #             **{str(m): er_nodes[m]['score'] for m in set(g.objects(table_uri, QB.measure)) if
        #             m in er_nodes},
        #             **{str(d): er_nodes[d]['score'] for d in set(g.objects(table_uri, QB.dimension)) if
        #             d in er_nodes}
        #         }.items(), key=lambda x: x[1], reverse=True)))
            
            
        #     )

        # tables = set(g.subjects(QB.measure, None))
        # Measure = set(g.objects(None, QB.measure))
        # dims = set(g.objects(None, QB.dimension))
        # print(tables, "tables\n", Measure, "Measure\n", dims, "dims\n")

        # return (g,
        #         tables,
        #         {uri_to_code(m): er_nodes[str(m)] for m in Measure if str(m) in er_nodes},
        #         {uri_to_code(d): er_nodes[str(d)] for d in dims if str(d) in er_nodes})

    # @staticmethod
    # def get_candidate_nodes_better(query, k=25, full_graph: bool = False, verbose=False) -> \
    #         Tuple[Graph, Dict, Dict, Dict]:
    #     query = [w for w in word_tokenize(query) if w not in string.punctuation]
    #     query = query + expand_query(query)

    #     query_embedding = sentence_transformer_encode(' '.join(query))

    #     tables = retrieve_top_tables(query_embedding)
    #     er_nodes = {t['id']: {'score': t['rel'], 'matched_words': [uuid4()]} for _, t in tables.iterrows()}
    #     print(tables)
    #     for index, row in tables.iterrows():
    #         print(f"\nIn table {row['id']} we found the following:\n")
    #         s = get_similarity_scores_for_Measure(query_embedding, row['id'])
    #         # TODO: make sure no duplicates are returned in the first place
    #         msr_dims = s.drop_duplicates().set_index('id', drop=True).to_dict(orient='index')
    #         er_nodes |= {(DIM if v['type'] == 'dimensie' else MSR).rdf_ns.term(k): {
    #             'score': v['rel'],
    #             'matched_words': [uuid4()]
    #         } for k, v in msr_dims.items()}
    #         print(s)

    #     # EXPLODE!
    #     expl_fun = explode_subgraph_msr_dims_only if not full_graph else explode_subgraph
    #     g = expl_fun(list(er_nodes.keys()), verbose=verbose)

    #     tables = set(g.subjects(QB.measure, None))
    #     Measure = set(g.objects(None, QB.measure))
    #     dims = set(g.objects(None, QB.dimension))

    #     return (g,
    #             {uri_to_code(t): er_nodes[t] for t in tables if t in er_nodes},
    #             {uri_to_code(m): er_nodes[m] for m in Measure if m in er_nodes},
    #             {uri_to_code(d): er_nodes[d] for d in dims if d in er_nodes})


    def get_candidate_nodes_bm25(self, query, k=25, full_graph: bool = False, verbose=False) -> \
            (OrderedDict[str, Tuple[float, OrderedDict[str, float]]], Graph):
        """
            Find all candidate entities corresponding with the query in the graph using the Elastic index. The result
            will be the candidate Measure and dimensions with their corresponding matching scores, and the resulting
            subgraph of all the connected tables and their connected Measure and dimensions from these candidates.

            :param query: NL search query
            :param k: number of closest ES-nodes to consider
            :param full_graph: fetch full graph, including dimension hierarchy relations (slower)
            :param verbose: print the executed SPARQL query
            :returns: (Subgraph, {candidate Measure: scores}, {candidate dims: scores})
        """
        query = [w for w in word_tokenize(query) if w not in string.punctuation]
        query = query + expand_query(query)

        # TODO: get top k-nodes, explode and ask ES separately for all BM25 codes per node in the subgraph
        q = {
            "track_total_hits": False,
            "_source": ["_id"],
            "min_score": 0.01,
            "size": k,
            "query": {
                "match": {
                    "body": {
                        "minimum_should_match": 1,
                        "query": ' '.join(query)
                    }
                }
            }
        }

        if verbose:
            logger.debug("Querying Elastic...")

        resp = es.search(body=q, index='graph-nl')
        er_nodes = [n['_id'] for n in resp.body['hits']['hits']]
        if len(er_nodes) == 0:
            raise ValueError("No nodes found in Elastic that correspond with the given query.")

        # Get all (table, measure/dimension, entity) combinations following from the relations of the candidate ER nodes
        if verbose:
            logger.debug("Exploding nodes and creating subgraph...")

        # EXPLODE!
        expl_fun = self.engine.explode_subgraph_msr_dims_only if not full_graph else self.engine.explode_subgraph
        g = expl_fun(er_nodes, verbose=verbose)

        # Get BM25-scores of all nodes in subgraph
        subgraph_nodes = [str(s) for s in set(g.subjects()) | set(g.objects())]
        q = {
            "size": len(subgraph_nodes),
            "_source": ["_id"],
            "query": {
                "bool": {
                    "should": [
                        {"match": {"body": ' '.join(query)}}
                    ],
                    "filter": [
                        {"terms": {"_id": subgraph_nodes}}
                    ]
                }
            },
            "highlight": {  # TODO: phrase recognized entities and only return sorted phrase highlights
                "pre_tags": ['<match>'],
                "post_tags": ['</match>'],
                "order": "score",
                "number_of_fragments": 1,   # TODO: for phrase highlighting increase fragment size to something sensible
                "highlight_query": {
                    "bool": {
                        "minimum_should_match": 0,
                        "should": [{
                            "multi_match": {
                                "query": ' '.join(query),
                                "type": "best_fields",
                                "fields": ['body'],
                            }
                        }]
                    }
                },
                "fields": {
                    "body": {
                        "boundary_scanner_locale": "NL-nl",
                        "type": "fvh"
                    }
                }
            }
        }

        er_nodes = {}
        resp = es.search(body=q, index='graph-nl')
        for doc in resp.body['hits']['hits']:
            er_nodes[doc['_id']] = {
                "score": doc['_score'],
                "matched_words": (set([s.lower() for s in re.findall(r"<match>(.*?)</match>",
                                                                     doc['highlight']['body'][0])])
                                  if 'highlight' in doc else None)
            }

        tables = set(g.subjects(QB.measure, None))
        Measure = set(g.objects(None, QB.measure))
        dims = set(g.objects(None, QB.dimension))

        return (g,
                {uri_to_code(t): er_nodes[str(t)] for t in tables if str(t) in er_nodes},
                {uri_to_code(m): er_nodes[str(m)] for m in Measure if str(m) in er_nodes},
                {uri_to_code(d): er_nodes[str(d)] for d in dims if str(d) in er_nodes})