from transformers import AutoTokenizer

import config
from model.encoderdecoder_trainer import SExpressionModelEngine
from odata_graph.sparql_controller import SparqlEngine
from pipeline.entity_retriever import EntityRetriever
from pipeline.odata_executor import ODataExecutor
from utils.logical_forms import uri_to_code, MSR, DIM

engine = SparqlEngine(local=False)
er = EntityRetriever(engine=engine)

enc_tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_ENC_PATH)
dec_tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_DEC_PATH)

model = SExpressionModelEngine(enc_tokenizer=enc_tokenizer,
                               dec_tokenizer=dec_tokenizer,
                               pretrained_or_checkpoint=f"{config.TRAINED_MODEL_PATH}/model",
                               train_mode=False,
                               beam_size=2)


def qa_model(query: str, return_sexp: bool = False, verbose: bool = False):
    sexp, ranked_nodes = model.conditional_generation(query, verbose)

    if return_sexp:
        return sexp

    if verbose:
        print(sexp)
        sexp.print_tree()

    try:
        ranked_msrs = [uri_to_code(o) for o in ranked_nodes.keys() if o in MSR.rdf_ns]
        ranked_dims = [uri_to_code(o) for o in ranked_nodes.keys() if o in DIM.rdf_ns]
        Measure = dict(zip(ranked_msrs, range(len(ranked_msrs))))
        dims = dict(zip(ranked_msrs, range(len(ranked_dims))))

        odata = ODataExecutor(query, sexp, Measure, dims, model.entity_retriever.engine)
        answer = odata.query_odata()
        answer.query = query

        return answer.sexp, answer
    except:
        return sexp, None


if __name__ == "__main__":
    # query = 'Uitgaven zorg overheid 2020'
    # query = 'Vervoer pijpleidingen in Nederland?'
    # query = 'Hoeveel mensen gingen er met de trein op vakantie naar het buitenland?'
    # query = 'Internetgebruik van bedrijven'
    query = 'Gemiddelde energieprijzen van consumenten in 2018'

    _, answer = qa_model(query, verbose=False)
    print(answer)
