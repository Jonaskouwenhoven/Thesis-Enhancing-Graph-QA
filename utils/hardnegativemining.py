import pandas as pd
from tqdm import tqdm
import random
import torch
import torch.nn.functional as F
import utils.utils as utils
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from tqdm import tqdm 
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def hardnegativemining_cbs_groups(train_df : pd.DataFrame, table_df : pd.DataFrame, n=5) -> pd.DataFrame:
    """
    Summary:
    Create a DataFrame containing triplets of (query, anchor_embedding, positive_embedding, negative_embedding)

    Args:
    - train_df (pd.DataFrame): DataFrame containing query and table data for training.
    - table_df (pd.DataFrame): DataFrame containing table data with embeddings.

    Returns:
    - pd.DataFrame: DataFrame containing triplets.
    """
    queries, anchor_embeddings, positive_embeddings, negative_embeddings = [], [], [], []

    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc = "Creating Triples using Groups"):
        query = row['query']
        query_emb_tensor = row['tensor_embedding']
        golden_table = row['table_id']
        positive_emb_tensor = table_df[table_df['table_id'] == golden_table]['tensor_embedding'].values[0]
        golden_group = table_df[table_df['table_id'] == golden_table]['Groepnaam'].values[0]
        # print(golden_group)
        negative_groups = table_df[table_df['Groepnaam'] != golden_group]['Groepnaam'].unique()
        table_df_groupnaams = table_df['Groepnaam'].unique()
        # print(table_df_groupnaams)
        # print(golden_group in table_df_groupnaams)
        
        for negative_group in negative_groups:
            try:
                negative_table = table_df[table_df['Groepnaam'] == negative_group].sample(n, replace=True)
                for _, row_neg in negative_table.iterrows():
                    negative_emb = row_neg['tensor_embedding']
                    queries.append(query)
                    anchor_embeddings.append(query_emb_tensor)
                    positive_embeddings.append(positive_emb_tensor)
                    negative_embeddings.append(negative_emb)
            except Exception as e:
                continue

                
    df_triplets = pd.DataFrame({
        'query': queries,
        'anchor_embedding': anchor_embeddings,
        'positive_embedding': positive_embeddings,
        'negative_embedding': negative_embeddings
    })

    return df_triplets


def hardnegativemining_bm25(train_df : pd.DataFrame, table_df : pd.DataFrame, n=5) -> pd.DataFrame:
    """
    Summary:
    Create a DataFrame containing triplets of (query, anchor_embedding, positive_embedding, negative_embedding)

    Args:
    - train_df (pd.DataFrame): DataFrame containing query and table data for training.
    - table_df (pd.DataFrame): DataFrame containing table data with embeddings.

    Returns:
    - pd.DataFrame: DataFrame containing triplets.
    """

    tokenized_corpus = [doc.split(" ") for doc in table_df['desc']]

    # Create BM25 object.
    bm25 = BM25Okapi(tokenized_corpus)

    queries, anchor_embeddings, positive_embeddings, negative_embeddings = [], [], [], []

    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Creating triplets using BM25"):
        query = row['query'].split(" ")  # Tokenize the query
        golden_table = row['table_id']

        # Get BM25 scores for the current query against all documents in the corpus.
        doc_scores = bm25.get_scores(query)

        # Create a DataFrame to store table IDs and their corresponding BM25 scores.
        df = pd.DataFrame({'table_id': table_df['table_id'], 'bm25_score': doc_scores})

        # Sort the DataFrame based on BM25 scores in descending order.
        df = df.sort_values(by='bm25_score', ascending=False)
        
        top_10 = df.head(n)

        table_ids = top_10['table_id'].tolist()
        
        # remove golden table from table_ids
        if golden_table in table_ids:
            table_ids.remove(golden_table)
            
        ## now we are going to mine the negatives
        
        for table in table_ids:
            temp_table_df = table_df[table_df['table_id'] == table]
            golden_table_df = table_df[table_df['table_id'] == golden_table]
            
            queries.append(row['query'])
            anchor_embeddings.append(row['tensor_embedding'])
            
            positive_embeddings.append(golden_table_df['tensor_embedding'].values[0])
            negative_embeddings.append(temp_table_df['tensor_embedding'].values[0])
            
    df_triplets = pd.DataFrame({
        'query': queries,
        'anchor_embedding': anchor_embeddings,
        'positive_embedding': positive_embeddings,
        'negative_embedding': negative_embeddings
    })

    return df_triplets



def hardnegativemining_topn_bottomn(query_df : pd.DataFrame, table_df : pd.DataFrame, n=5) -> pd.DataFrame:
    """
    Summary:
    Create a DataFrame containing triplets of (query, anchor_embedding, positive_embedding, negative_embedding)

    Args:
    - train_df (pd.DataFrame): DataFrame containing query and table data for training.
    - table_df (pd.DataFrame): DataFrame containing table data with embeddings.

    Returns:
    - pd.DataFrame: DataFrame containing triplets.
    """
    queries, anchor_embeddings, positive_embeddings, negative_embeddings = [], [], [], []

    for _, row in tqdm(query_df.iterrows(), total = len(query_df), desc = "Creating triplets using top_n-bottom_n"):
        try:
            # Anchor: Query
            query = row['query']
            query_emb = row['embedding']
            query_emb_tensor = row['tensor_embedding']
            table_id = row['table_id']

            # Positive Sample: Relevant Table
            positive_table_emb = table_df[table_df['table_id'] == table_id]['embeddings'].iloc[0]
            positive_table_emb_tensor = table_df[table_df['table_id'] == table_id]['tensor_embedding'].iloc[0]

            # Find Hard Negative Sample: Closest Non-Relevant Table
        
            negative_table = table_df[table_df['table_id'] != table_id]
            negative_table_emb = negative_table[['table_id', 'embeddings']]
            
            for hard_negative_id in utils.find_top_similar_tables(query_emb, negative_table_emb, k=n, ascending_param=True)['table_id'].to_list():

                hard_negative_emb_tensor = table_df[table_df['table_id'] == hard_negative_id]['tensor_embedding'].iloc[0]

                # Append to lists
                queries.append(query)
                anchor_embeddings.append(query_emb_tensor)
                positive_embeddings.append(positive_table_emb_tensor)
                negative_embeddings.append(hard_negative_emb_tensor)

            for hard_negative_id in utils.find_top_similar_tables(query_emb, negative_table_emb, k=n, ascending_param=False)['table_id'].to_list():

                hard_negative_emb_tensor = table_df[table_df['table_id'] == hard_negative_id]['tensor_embedding'].iloc[0]

                # Append to lists
                queries.append(query)
                anchor_embeddings.append(query_emb_tensor)
                positive_embeddings.append(positive_table_emb_tensor)
                negative_embeddings.append(hard_negative_emb_tensor)
        except Exception as e:
            print("Error", e)
            continue

    df_triplets = pd.DataFrame({
        'query': queries,
        'anchor_embedding': anchor_embeddings,
        'positive_embedding': positive_embeddings,
        'negative_embedding': negative_embeddings
    })

    return df_triplets

def hardnegativemining_random(query_df : pd.DataFrame, table_df : pd.DataFrame, n=5) -> pd.DataFrame:
    """
    Summary:
    Create a DataFrame containing triplets of (query, anchor_embedding, positive_embedding, negative_embedding)

    Args:
    - train_df (pd.DataFrame): DataFrame containing query and table data for training.
    - table_df (pd.DataFrame): DataFrame containing table data with embeddings.

    Returns:
    - pd.DataFrame: DataFrame containing triplets.
    """
    queries, anchor_embeddings, positive_embeddings, negative_embeddings = [], [], [], []

    for _, row in tqdm(query_df.iterrows(), total=len(query_df), desc="Creating triplets using random"):
        # Anchor: Query
        query = row['query']
        query_emb_tensor = row['tensor_embedding']

        table_id = row['table_id']

        # Positive Sample: Relevant Table
        positive_table_emb = table_df[table_df['table_id'] == table_id]['tensor_embedding'].iloc[0]

        # Find Random Negative Sample: Non-Relevant Table
        negative_table = table_df[table_df['table_id'] != table_id]
        negative_table_emb = negative_table[['table_id', 'tensor_embedding']]
        random_negative_id = random.choice(negative_table_emb['table_id'].to_list())
        random_negative_emb = negative_table_emb[negative_table_emb['table_id'] == random_negative_id]['tensor_embedding'].iloc[0]

        # Append to lists
        queries.append(query)
        anchor_embeddings.append(query_emb_tensor)
        positive_embeddings.append(positive_table_emb)
        negative_embeddings.append(random_negative_emb)

    df_triplets = pd.DataFrame({
        'query': queries,
        'anchor_embedding': anchor_embeddings,
        'positive_embedding': positive_embeddings,
        'negative_embedding': negative_embeddings
    })


    return df_triplets




def hardnegativemining_clusters(train_df : pd.DataFrame, table_df : pd.DataFrame, n_clusters = 15, n_negatives = 5) -> pd.DataFrame:
    """
    Create a DataFrame containing triplets of (query, anchor_embedding, positive_embedding, negative_embedding)
    based on K-means clustering.

    Args:
    - train_df (pd.DataFrame): DataFrame containing query and table data for training.
    - table_df (pd.DataFrame): DataFrame containing table data with clusters.
    - n_clusters (int): Number of clusters used for K-means clustering.
    - convert_embeddings (function): Function to convert embeddings to the required format.

    Returns:
    - pd.DataFrame: DataFrame containing triplets.
    """

    # Normalize the embeddings for better clustering performance
    normalized_embeddings = normalize(np.stack(table_df['embeddings'].values))

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(normalized_embeddings)
    table_df['cluster'] = kmeans.labels_

    # Initialize lists to store the triplets
    queries, anchor_embeddings, positive_embeddings, negative_embeddings = [], [], [], []

    # For each query, select hard negative examples based on cluster similarity
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Creating triplets using clusters"):
        try:
            query = row['query']
            table_id = row['table_id']
            query_emb_tensor = row['tensor_embedding']

            positive_table_emb = table_df[table_df['table_id'] == table_id]['embeddings'].iloc[0]
            positive_table_emb_tensor = table_df[table_df['table_id'] == table_id]['tensor_embedding'].iloc[0]

            # Calculate similarity between positive embedding and cluster centroids
            cluster_similarity = []
            for cluster_id in range(n_clusters):
                cluster_center = kmeans.cluster_centers_[cluster_id]
                similarity = np.dot(positive_table_emb, cluster_center)
                cluster_similarity.append((cluster_id, similarity))

            # Sort clusters by similarity in descending order
            cluster_similarity.sort(key=lambda x: x[1], reverse=True)

            # Select hard negatives from clusters in order of similarity
            n = n_negatives # Select 7 hard negatives from each cluster
            for cluster_id, _ in cluster_similarity:
                negative_tables = table_df[(table_df['cluster'] == cluster_id) & (table_df['table_id'] != table_id)]

                if not negative_tables.empty:
                    # Select "n" hard negatives from this cluster
                    selected_negatives = negative_tables.sample(n=n)

                    for _, neg_row in selected_negatives.iterrows():
                        hard_negative_emb_tensor = neg_row['tensor_embedding']

                        # Append to lists
                        queries.append(query)
                        anchor_embeddings.append(query_emb_tensor)
                        positive_embeddings.append(positive_table_emb_tensor)
                        negative_embeddings.append(hard_negative_emb_tensor)
        except Exception as e:
            # print("Error", e)
            continue

    # Create DataFrame for triplets
    df_triplets = pd.DataFrame({
        'query': queries,
        'anchor_embedding': anchor_embeddings,
        'positive_embedding': positive_embeddings,
        'negative_embedding': negative_embeddings
    })

    return df_triplets



def hardnegativemining_validation(model, validation_df, table_df, x=800):
    """
    Create a new triplets dataset using a model and a validation DataFrame.

    Args:
    - model (nn.Module): The neural network model.
    - validation_df (pd.DataFrame): Validation DataFrame.
    - table_df (pd.DataFrame): DataFrame containing table data with embeddings.
    - convert_triplet_embeddings (function): Function to convert embeddings to triplets.
    - device (str): The device (e.g., "cuda" or "cpu") to use for computations.
    - x (int): Number of rows to process from the validation DataFrame.

    Returns:
    - pd.DataFrame: New triplets DataFrame.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    queries, new_anchor, new_positive, new_negative = [], [], [], []
    val_df = validation_df[:x]
    table_identifiers = table_df['table_id'].to_list()
    table_embeddings = table_df['embeddings'].to_list()

    for _, row in tqdm(val_df.iterrows(), total=len(val_df)):
        try:
            query_text = row['query']
            query = row['embedding']
            tensor_query = row['tensor_embedding']
            golden_table = row['table_id']
            query_emb_tensor = torch.tensor(query).unsqueeze(0).to(device)
            cosine_similarities = []

            for table_emb in table_embeddings:
                table_emb_tensor = torch.tensor(table_emb).unsqueeze(0).to(device)

                output_query = model.query_encoder(query_emb_tensor.to(torch.float32))
                output_positive = model.table_encoder(table_emb_tensor.to(torch.float32))

                cosine_similarity = F.cosine_similarity(output_query, output_positive, dim=1)
                cosine_similarities.append(cosine_similarity.item())

            top = sorted(range(len(cosine_similarities)), key=lambda i: cosine_similarities[i], reverse=True)[:]

            top_table_ids = [table_identifiers[i] for i in top]

            if golden_table not in top_table_ids[:5]:
                for table in top_table_ids[:5]:
                    new_positive.append(table_df[table_df['table_id'] == golden_table]['tensor_embedding'].to_list()[0])
                    new_negative.append(table_df[table_df['table_id'] == table]['tensor_embedding'].to_list()[0])
                    queries.append(query_text)
                    new_anchor.append(tensor_query)

        except:
            continue

    # Create the new DataFrame with the hard negatives
    df_triplets2 = pd.DataFrame({
        'query': queries,
        'anchor_embedding': new_anchor,
        'positive_embedding': new_positive,
        'negative_embedding': new_negative
    })

    return df_triplets2