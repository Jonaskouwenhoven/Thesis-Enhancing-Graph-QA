import pandas as pd
import numpy as np
import utils.utils as utils
import torch
import torch
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from neural_cherche import models, utils, train, losses


TABLEDF = pd.read_pickle("data/tabledf.pkl")
TABLEDF = TABLEDF[TABLEDF['table_id'] != '85210NED']
# TABLEDF = utils.add_tensor_embedding_column(TABLEDF, "embeddings")


QUERYDF = pd.read_pickle("data/querydf.pkl")[:5]

RANDOMSTATE = 42
TESTSIZE = 0.2
# df = utils.add_t

TRAINDF, TESTDF = train_test_split(QUERYDF, test_size=TESTSIZE, random_state=RANDOMSTATE) 


def parse_arguments():
    parser = argparse.ArgumentParser(description="Argument Parser Example")
    parser.add_argument('--model_name', type=str, default='textgain/allnli-GroNLP-bert-base-dutch-cased', help='Name of the model (default: gronlp)')
    parser.add_argument('--negative_n', type=int, default=5, help='Negative N value (default: 5)')
    parser.add_argument('--output_name', type=str, default="ColBERT", help='Output name of the trained model')

    return parser.parse_args()



def create_train_set(negative_n):
    TRAIN = []


    anchor_eval, positive_eval, negative_eval = [], [], []
    for index, row in TRAINDF[:].iterrows():
        query = row['Query']
        table_id = row['table_id']
        query_embedding = torch.tensor(row['Embeddings'])  # Make sure this is a numpy array or in a format that cosine_similarity can handle

        positive_emb = torch.tensor(TABLEDF[TABLEDF['table_id'] == table_id]['embeddings'].values[0])
        postive_text = TABLEDF[TABLEDF['table_id'] == table_id]['Summary'].values[0]
        # Prepare embeddings for cosine similarity comparison, excluding the positive one
        negative_candidates = TABLEDF[TABLEDF['table_id'] != table_id]
        
        # Reshape query_embedding for compatibility with cosine_similarity function
        query_embedding_reshaped = query_embedding.reshape(1, -1)
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_embedding_reshaped, np.vstack(negative_candidates['embeddings'].values))
        
        negative_candidates['similarity'] = similarities[0]

        negative_candidates = negative_candidates.sort_values(by='similarity', ascending=False)

        negative_texts = negative_candidates['Summary']

        for i in range(negative_n):
            negative_text = negative_texts.values[i]
            TRAIN.append((query, postive_text, negative_text))

        negative_candidates = negative_candidates.sort_values(by='similarity', ascending=True)

        for i in range(negative_n):
            negative_text = negative_texts.values[i]
            TRAIN.append((query, postive_text, negative_text))


    return TRAIN


def maintrain(TRAIN, model, output_name):

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-6)

    for step, (anchor, positive, negative) in enumerate(utils.iter(
            TRAIN,
            epochs=1,
            batch_size=8,
            shuffle=True
        )):

        loss = train.train_colbert(
            model=model,
            optimizer=optimizer,
            anchor=anchor,
            positive=positive,
            negative=negative,
            step=step,
            gradient_accumulation_steps=35,
        )

        if (step + 1) % 500 == 0:
            print(f"At step {step + 1}, loss: {loss['loss']}")

        if (step + 1) % 1000 == 0:
            # Save the model every 1000 steps
            model.save_pretrained(output_name)
    
    return model



if __name__ == '__main__':
    
    args = parse_arguments()
    output_name = args.output_name
    TRAIN = create_train_set(args.negative_n)

    model = models.ColBERT(
    model_name_or_path=args.model_name,
    device="cuda" if torch.cuda.is_available() else "cpu"
    )

    model = maintrain(TRAIN, model, output_name)

    model.save_pretrained(output_name)


