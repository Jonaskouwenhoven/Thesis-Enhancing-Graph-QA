import itertools
import operator
from elasticsearch.helpers import bulk
from tqdm import tqdm

from elastic import es
from odata_graph.sparql_controller import SparqlEngine
from utils.global_functions import secure_request


engine = SparqlEngine()


def index_nodes():
    tables = secure_request('https://odata4.cbs.nl/CBS/datasets', json=True, max_retries=9, timeout=20)['value']
    fetch_tables = [t['Identifier'] for t in tables]

    tables = {}
    nodes = {}
    for table in tqdm(fetch_tables, desc="Getting nodes", bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}'):
        props = secure_request(f"https://odata4.cbs.nl/CBS/{table}/Properties", json=True, max_retries=9, timeout=20)
        while not props:  # Can't continue when props isn't filled
            props = secure_request(f"https://odata4.cbs.nl/CBS/{table}/Properties",
                                   json=True, max_retries=9, timeout=20)

        tables[f"https://opendata.cbs.nl/#/CBS/nl/dataset/{table}"] = {
            'body': ' '.join([props['Title'], props['Description'], props['Summary'], props['LongDescription']]),
            'type': 'table'
        }

        # Get all Measure and dimensions for a table. Skip the time and geo dimensions
        query = (f"""
            PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
            PREFIX dct: <http://purl.org/dc/terms/>
            PREFIX qb: <http://purl.org/linked-data/cube#>
            
            SELECT ?o ?label WHERE {{ 
                ?s qb:measure|qb:dimension ?o .
                FILTER NOT EXISTS {{
                    VALUES ?dim {{'TimeDimension' 'GeoDimension'}}
                    ?o ?has_type ?dim .
                }}
                ?s dct:identifier ?id .
                OPTIONAL {{ ?o skos:prefLabel|skos:altLabel|skos:definition|dct:description|dct:subject ?label }}
                FILTER (?id = "{table}")
                FILTER (!BOUND(?label) || lang(?label) = "nl")
            }}
        """)

        # TODO: not all tables have to be in the graph. Handle this properly
        try:
            result = engine.select(query)
            props = [(r['o']['value'], (r.get('label', False) or {'value': ''})['value']) for r in result]
            it = itertools.groupby(props, operator.itemgetter(0))
            for key, subiter in it:
                val = ' '.join(prop[1] for prop in subiter)
                nodes[key] = {'body': val, 'type': 'node'}
                tables[f"https://opendata.cbs.nl/#/CBS/nl/dataset/{table}"]['body'] += ' ' + val
        except Exception as e:
            print(f"Failed to fetch table nodes: {e}")

    params = {
        "k1": 1.2,  # positive tuning parameter that calibrates the document term frequency scaling
        "b": 0.3,   # 0.75,  # parameter that determines the scaling by document length
        "d": 1.0    # makes sure the component of term frequency normalization by doc. length is properly lower-bounded
    }

    settings = {
        "number_of_shards": 1,
        "number_of_replicas": 1,
        "index": {
            "similarity": {
                "bm25_plus": {
                    "type": "scripted",
                    "script": {
                        "source": f"double idf = Math.log((field.docCount+1.0)/(term.docFreq+1.0)) + 1.0;"
                                  f"return query.boost * idf * (("
                                  f"(doc.freq * ({params['k1']} + 1))/(doc.freq + ({params['k1']} * (1 - {params['b']} + "
                                  f"({params['b']} * doc.length/(field.sumTotalTermFreq/field.docCount)))))) + {params['d']});"
                    }
                }
            },
            "store.preload": ["nvd", "dvd"],
        },
        "analysis": {
            "filter": {
                "dutch_stop": {
                    "type": "stop",
                    "ignore_case": True,
                    "stopwords": ["_dutch_", "hoeveel", "waar", "waarom", "aantal", "welke", "wanneer", "waardoor", "gemiddeld"]
                },
                "dutch_stemmer": {
                    "type": "stemmer",
                    "language": "dutch"
                },
                "index_shingle": {
                    "type": "shingle",
                    "min_shingle_size": 2,
                    "max_shingle_size": 3,
                },
                "ascii_folding": {
                    "type": "asciifolding",
                    "preserve_original": False
                },
            },
            "analyzer": {
                "graph-nl": {
                    "enc_tokenizer": "standard",
                    "filter": [
                        "lowercase",
                        "dutch_stop",
                        "apostrophe",
                        "ascii_folding",
                        "dutch_stemmer",
                        "index_shingle",
                    ]
                }
            }
        }
    }

    mappings = {
        'properties': {
            'unique_id': {
                'type': 'keyword'
            },
            'body': {
                'type': 'text',
                "term_vector": "with_positions_offsets",
                'analyzer': "graph-nl",
                'search_analyzer': "graph-nl",
                "similarity": "bm25_plus",
            },
            'type': {
                'type': 'keyword'
            },
            "embedding_vector": {
                "type": "dense_vector",
                "dims": 768,
                "index": True,
                "similarity": "cosine"
            }
        }
    }

    if es.indices.exists(index='graph-nl'):
        es.indices.delete(index='graph-nl')
    es.indices.create(index='graph-nl', settings=settings, mappings=mappings)

    ops = []
    try:
        for i, (id_, d) in enumerate(tqdm((nodes | tables).items(), desc=f"Indexing nodes",
                                          bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}')):
            search_doc = {
                'unique_id': id_,
                'body': d['body'],
                'type': d['type']
            }

            ops.append({
                '_index': 'graph-nl',
                '_id': id_,
                '_source': search_doc
            })
            if i % 50 == 0 and i > 0:
                bulk(es, ops, chunk_size=50, request_timeout=30)
                ops = []
        if len(ops) > 0:
            bulk(es, ops, chunk_size=50, request_timeout=30)
            ops = []
    except StopIteration:
        print('StopIteration')
        bulk(es, ops, chunk_size=50, request_timeout=30)
    except Exception as e:
        print('Indexing stopped unexpectedly')
        print(e)


if __name__ == "__main__":
    index_nodes()
