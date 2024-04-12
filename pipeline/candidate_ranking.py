import pandas as pd
from operator import itemgetter
from rdflib import Graph, RDF, QB, SKOS
from typing import Dict, Literal

from odata_graph.sparql_controller import SCOT
from utils.logical_forms import TABLE, MSR, DIM, uri_to_code


class CandidateRanker:
    def __init__(self, graph: Graph, tables, Measure, dims):
        self.graph = graph
        self.tables = tables
        self.Measure = Measure
        self.dims = dims

    def rank_entities_bm25(self, total_boost_factor=2, epsilon=0.1) -> \
            Dict[str, Dict[Literal['Measure', 'dims'], Dict[str, float]]]:
        """
            Group and rank retrieved entities from BM25 by grouping them on matched words and returning
            the best scoring Measure and dimensions per group per table. Boost 'SCOT.Total'-typed entities.

            :param total_boost_factor: factoor to boost Measure/dimensions of type scot:total with
            :param epsilon: minimum score per document
        """
        # print(f"These are the dioms!e! {self.dims}")
        totals = set(self.graph.subjects(RDF.type, SCOT.Total))
        Measure = {k: {'score': v['score'] * (total_boost_factor if MSR.rdf_ns.term(k) in totals else 1),
                        'matched_words': v['matched_words']} for k, v in self.Measure.items() if v['score'] > epsilon}
        dims = {k: {'score': v['score'] * (total_boost_factor if DIM.rdf_ns.term(k) in totals else 1),
                    'matched_words': v['matched_words']} for k, v in self.dims.items() if v['score'] > epsilon}

        ranked_entities: Dict[str, Dict[Literal['Measure', 'dims'], Dict[str, float]]] = {}
        for table in self.tables.keys():
            ranked_entities[table] = {'Measure': {}, 'dims': {}}

            table_uri = TABLE.rdf_ns.term(table)
            table_msrs = {uri_to_code(m) for m in self.graph.objects(table_uri, QB.measure)}
            table_dims = {uri_to_code(m) for m in self.graph.objects(table_uri, QB.dimension)}

            # TODO: speed up w/ numpy
            ranked_msrs = {}
            msrs_by_word = pd.DataFrame([(k, v['score'], word) for k, v in Measure.items() if k in table_msrs
                                         for word in v['matched_words'] or []],
                                        columns=['id', 'score', 'matched_word'])
            if not msrs_by_word.empty:
                # Save top-scoring measure for every individual word-match group
                for matched_word, group in msrs_by_word.groupby('matched_word'):
                    top_msr = group.sort_values('score', ascending=False).iloc[0]
                    ranked_msrs[top_msr['id']] = top_msr['score']

                ranked_entities[table]['Measure'] = dict(sorted(ranked_msrs.items(), reverse=True, key=itemgetter(1)))

            ranked_dims = {}
            dims_by_word = pd.DataFrame([(k, v['score'], word) for k, v in dims.items() if k in table_dims
                                         for word in v['matched_words'] or []],
                                        columns=['id', 'score', 'matched_word'])
            if not dims_by_word.empty:
                dim_groups = (set(self.graph.objects(table_uri, QB.dimension)) -
                              set(self.graph.subjects(SKOS.broader, None)))
                # Save top-scoring dim for every dim_group for every individual word-match group
                for group_uri in dim_groups:
                    group_nodes = {uri_to_code(d) for d in (set(self.graph.objects(group_uri, SKOS.narrower)) -
                                                            set(self.graph.subjects(SKOS.narrower, None)) &
                                                            set(self.graph.objects(table_uri, QB.dimension)))}

                    word_groups = dims_by_word[dims_by_word['id'].isin(group_nodes)].groupby('matched_word')
                    for matched_word, group in word_groups:
                        top_dim = group.sort_values('score', ascending=False).iloc[0]
                        ranked_dims[top_dim['id']] = top_dim['score']

                ranked_entities[table]['dims'] |= dict(sorted(ranked_dims.items(), reverse=True, key=itemgetter(1)))

        # TODO: experiment with removing all dims below a relative threshold score based on the top scoring msrs/dims
        return ranked_entities


    def rank_entities_above_threshold(self, threshold=0.1) -> dict:
        """
        Return Measure and dimensions with scores above the specified threshold.
        
        :param threshold: Score threshold for filtering entities.
        :return: Dictionary of tables with their high-scoring Measure and dimensions.
        """
        # Filter Measure and dimensions based on the threshold
        filtered_Measure = {k: v for k, v in self.Measure.items() if v['score'] > threshold}
        filtered_dims = {k: v for k, v in self.dims.items() if v['score'] > threshold}

        # Initialize the dictionary to hold the results
        ranked_entities = {table: {'Measure': {}, 'dims': {}} for table in self.tables.keys()}

        for table in self.tables.keys():
            # Identify Measure and dimensions relevant to the table
            table_Measure = set(filtered_Measure.keys())
            table_dims = set(filtered_dims.keys())

            # Filter Measure and dimensions specific to the current table
            ranked_entities[table]['Measure'] = {k: v for k, v in filtered_Measure.items() if k in table_Measure}
            ranked_entities[table]['dims'] = {k: v for k, v in filtered_dims.items() if k in table_dims}

        return ranked_entities