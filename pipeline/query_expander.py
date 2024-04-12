"""
    This module contains code that is used for expanding incoming queries with extra keywords to enhance the retrieval
    of documents.
"""
import itertools
import pickle
import unidecode
from Levenshtein import ratio
from typing import List

from config import THESAURUS_PATH

with open(THESAURUS_PATH, 'rb') as data:
    thesaurus = pickle.load(data) if data else {'synonyms': {}}


def expand_query(query: list) -> List[str]:
    """
        Expand relevant query terms by getting synonyms from the CBS Taxonomy thesaurus

        :param query: given query terms to expand
        :return: expanded query, with the expanded words added after the original
    """
    query = [unidecode.unidecode(q) for q in query]
    expandable_terms = set()
    for s in thesaurus['synonyms']:
        if s and len(s.split()) <= len(query):
            if all(w in query for w in unidecode.unidecode(s).split()):
                expandable_terms.update(thesaurus['synonyms'][s][:4])

    if expandable_terms:
        compare_terms = list(expandable_terms) + query
        if len(query) > 1:
            compare_terms += [" ".join(query)]
        delete = set(query)

        for x, y in itertools.combinations(compare_terms, 2):
            if ratio(x, y) >= 0.8:
                if x in query or y in query:
                    delete.add(x)
                    delete.add(y)
                else:
                    if len(x) > len(y):
                        delete.add(x)
                    else:
                        delete.add(y)

        for d in delete:
            if d in compare_terms:
                compare_terms.remove(d)

        return compare_terms[::-1]
    else:
        return query
