import numpy as np
import pandas as pd
import spacy
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# from utils.train import DualEncoder, ModifiedDualEncoder
from pipeline.model_arch import DualEncoder
from pipeline.query_expander import expand_query
from utils.logical_forms import TABLE
from rank_bm25 import BM25Okapi

# Load the Dutch model
# nlp = spacy.load("nl_core_news_sm")
model_embeddings = SentenceTransformer('textgain/allnli-GroNLP-bert-base-dutch-cased')
import warnings

warnings.filterwarnings("ignore")


def sentence_transformer_encode(sentences):
	embeddings = model_embeddings.encode(sentences)

	return embeddings



table_df = pd.read_pickle("data/tabledf.pkl")
# table_df.dropduplicates(subset=['table_id'], inplace=True)
table_df.drop_duplicates(subset=['table_id'], inplace=True)
Measure_df = pd.read_pickle("data/measure_dimensions_df.pkl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_tabel_retrieval = DualEncoder(768, 768).to(device)
model_tabel_retrieval.load_state_dict(torch.load("models/topnbottomn100epochs_8batch.pt", map_location=torch.device(device)))
model_tabel_retrieval.eval()

# model_measure = DualEncoder(768, 512).to(device)
# model_measure.load_state_dict(
# 	torch.load("model/measure_retrieval_model.pt", map_location=torch.device(device)))
# model_measure.eval()
# geo_time_df_per_table = pd.read_pickle('data/table_geo_time.pkl')


import pandas as pd
from neural_cherche import retrieve, models, utils, train
import torch

class RetrieverInitializer:
	def __init__(self, model_name_or_path="JoniJoniAl/ColBERTGecko", table_df_path="data/tabledf.pkl"):
		self.model_name_or_path = model_name_or_path
		self.table_df_path = table_df_path

	def initialize_Colbert(self):
		# Initialize the model
		model = models.ColBERT(
			model_name_or_path=self.model_name_or_path,
			device="cuda" if torch.cuda.is_available() else "cpu"
		)

		# Load and preprocess the table DataFrame
		table_df = pd.read_pickle(self.table_df_path)
		table_df.drop_duplicates(subset=['table_id'], inplace=True)

		documents = []
		for index, row in table_df.iterrows():
			document = {"id": row['table_id'], "text": row['Summary']}
			documents.append(document)

		# Initialize the retriever
		retriever = retrieve.ColBERT(
			key="id",
			on=["text"],
			model=model,
		)

		# Encode documents and add them to the retriever
		documents_embeddings = retriever.encode_documents(
			documents=documents,
			batch_size=8,
		)
		retriever = retriever.add(documents_embeddings=documents_embeddings)

		return retriever






def clean_Measure(Measure):
	new_Measure = []
	for m in Measure:
		if "JJ" in m:
			continue
		if "PC" in m:
			continue
		if "KW" in m:
			continue
		if "SJ" in m:
			continue
		if "TM" in m:
			continue
		if "CR" in m:
			continue
		if "LD" in m:
			continue
		else:
			new_Measure.append(m)

	return new_Measure


def get_new_table_embeddings():
	# Step 1: Embed the Query

	# Step 2: Prepare model and embeddings
	table_identifiers = table_df['table_id'].to_list()
	table_embeddings = table_df['extended_table_desc_embeddings'].to_list()

	# Step 3: Compute Similarity Scores
	transformed_table_embeddings, ids = [], []
	for i, table_emb in enumerate(table_embeddings):
		table_emb_tensor = torch.tensor(table_emb).unsqueeze(0).to(device)
		output_positive = model_tabel_retrieval.table_encoder(table_emb_tensor)
		transformed_table_embeddings.append(output_positive.detach().numpy()[0])
		ids.append(table_identifiers[i])

	return pd.DataFrame({"table_id": ids, "embedding": transformed_table_embeddings, "Original": table_embeddings})


def retrieve_top_tables_colbert(query,retriever, n_tables=30):
	# print(query, type(query))
	queries_embeddings = retriever.encode_queries(
		queries=[query],
		batch_size=8,
		tqdm_bar = False,
		)
	scores = retriever(
		queries_embeddings=queries_embeddings,
		batch_size=8,
		k=n_tables,
		tqdm_bar = False,
		)
 
	scores_df = pd.DataFrame(scores[0]).rename(columns={"id": "table_id", "similarity": "score"})

	scores_df['score'] = scores_df['score'] / scores_df['score'].max()
	scores_df['id'] = scores_df['table_id']
	return scores_df




def retrieve_top_tables(query_embedding, n_tables = 30):
	# Step 1: Embed the Query

	# Step 2: Prepare model and embeddings
	table_identifiers = table_df['table_id'].to_list()
	table_embeddings = table_df['embeddings'].to_list()
	table_titles = table_df['Table Title'].to_list()

	# Step 3: Compute Similarity Scores
	query_emb_tensor = torch.tensor(query_embedding).unsqueeze(0).to(device)
	similarities_cosine = []
	# output_query = model_tabel_retrieval.query_encoder(query_emb_tensor)

	for table_emb in table_embeddings:
		table_emb_tensor = torch.tensor(table_emb).unsqueeze(0).to(device)
		# output_positive = model_tabel_retrieval.table_encoder(table_emb_tensor)

		cosine_similarity_score = F.cosine_similarity(query_emb_tensor, table_emb_tensor, dim=1)
		similarities_cosine.append(cosine_similarity_score.item())

	# Step 4: Identify the Top 10 Tables
	top_10_indices_cosine = sorted(range(len(similarities_cosine)), key=lambda i: similarities_cosine[i], reverse=True)[:n_tables]
	top_table_urls= [TABLE.rdf_ns.term(table_identifiers[i]) for i in top_10_indices_cosine]
	top_similarity_scores = [similarities_cosine[i] for i in top_10_indices_cosine]
	top_table_ids = [table_identifiers[i] for i in top_10_indices_cosine]
	top_titles = [table_titles[i] for i in top_10_indices_cosine]
	# Step 5: Creating DataFrame
	result_df = pd.DataFrame({
		'id': top_table_urls,
		'score': top_similarity_scores,
		'table_id': top_table_ids,
		"table_title": top_titles
	})

	return result_df

def get_table_query_sim(query_embedding, table_id):
	table = table_df[table_df['table_id'] == table_id]
	if table.empty:
		raise KeyError(f"Table {table_id} does not have a precomputed embedding.")

	table_embedding = table.iloc[0]['embeddings']
	table_emb_tensor = torch.tensor(table_embedding).unsqueeze(0).to(device)
	query_emb_tensor = torch.tensor(query_embedding).unsqueeze(0).to(device)

	output_query = model_tabel_retrieval.query_encoder(query_emb_tensor)
	output_positive = model_tabel_retrieval.table_encoder(table_emb_tensor)

	cosine_similarity_score = F.cosine_similarity(output_query, output_positive, dim=1)
	return cosine_similarity_score.item()


def find_associations(df : pd.DataFrame, table_id : str) -> pd.DataFrame:
	"""
	Find all id and title pairs associated with a given table_id.

	Args:
	df (pd.DataFrame): DataFrame with 'id', 'title', and 'table_ids' columns, where 'table_ids' contains lists of table_ids.
	table_id (int or str): The table_id to find associations for.

	Returns:
	pd.DataFrame: A DataFrame containing all id and title pairs associated with the given table_id.
	"""
	# Filter the DataFrame for rows where the 'table_ids' list contains the specified table_id
	df_ids = df['id'].to_list()
	table_id = table_id.split("/")[-1]
	# print(f"Table id: {df.columns}", table_id)
	
	filtered_df = df[df['table_ids'].apply(lambda x: table_id in x)]
	# print(f"The length of the filtered df is {len(filtered_df)}\n")
	filtered_df['table_id'] = filtered_df['table_ids'].apply(lambda x: table_id)
	# print(f"The length of the filtered df is {len(filtered_df)}\n")
	# print(filtered_df.columns)
	# Return the filtered DataFrame, which now contains only the relevant id and title pairs
	return filtered_df[['id', 'title', 'embeddings', 'table_id']]




def reranked(tables_df, query_embedding, top_Measure=20, top_dimensions=15):
	query_emb_tensor = torch.tensor(query_embedding).unsqueeze(0).to(device)
	scores = []
	for table_id in tables_df['table_id'].unique():
		# Step 1: Embed the Query
		query_emb_tensor = torch.tensor(query_embedding).unsqueeze(0).to(device)

		# Step 2: Retrieve Nodes for the Table
		possible_nodes = find_associations(Measure_df, table_id)

		# Separate the Measure and dimensions
		possible_Measure = possible_nodes[possible_nodes['soort'] == 'onderwerp']['id'].unique()

		# If no Measure or dimensions found, return empty DataFrames
		if len(possible_Measure) == 0:
			scores.append(0)
			continue

		# Step 3: Compute Similarity Scores for Measure
		Measure_similarity = compute_similarity(query_emb_tensor, possible_Measure, Measure_df, device)

		# Step 3: Compute Similarity Scores for Dimensions
		# Step 4: Create DataFrames for the top similarity scores for Measure and Dimensions
		similarities = create_similarity_dataframe(Measure_similarity, Measure_df, top_n=top_Measure)[
			'rel'].to_list()

		if len(similarities) > 0:
			percentile_95 = np.max(similarities)
			# top_5_percentile_similarities = [sim for sim in similarities if sim >= percentile_95]
			scores.append(percentile_95)
		else:
			scores.append(0)

	tables_df['measure_score'] = scores
	return tables_df


def compute_similarity(query_emb_tensor, possible_nodes, Measure_df, device):
	similarities_cosine = []
	valid_ids = []

	for node_id in possible_nodes:
		try:
			node_embedding = Measure_df[Measure_df['id'] == node_id]['embeddings'].to_list()[0]
			node_tensor = torch.tensor(node_embedding).unsqueeze(0).to(device)
			similarity_cosine = F.cosine_similarity(query_emb_tensor, node_tensor, dim=1)
			similarities_cosine.append(similarity_cosine.item())
			valid_ids.append(node_id)
		except Exception as e:
			print(f"Error processing node ID {node_id}: {e}")

	return {
		'valid_ids': valid_ids,
		'similarities_cosine': similarities_cosine
	}


def create_similarity_dataframe(similarity_data, Measure_df, top_n=20):
	# Sort and select the top N similarities
	top_indices = sorted(range(len(similarity_data['similarities_cosine'])),
						 key=lambda i: similarity_data['similarities_cosine'][i], reverse=True)[:top_n]
	top_ids = [similarity_data['valid_ids'][i] for i in top_indices]
	top_similarity_scores = [similarity_data['similarities_cosine'][i] for i in top_indices]

	# Retrieve descriptions and types
	descriptions = []
	types = []
	for node_id in top_ids:
		description = Measure_df[Measure_df['id'] == node_id]['title'].to_list()
		node_type = Measure_df[Measure_df['id'] == node_id]['soort'].to_list()
		descriptions.append(description[0] if description else "NAV")
		types.append(node_type[0] if node_type else "Unknown")

	types = ['onderwerp' if t == 'onderwerp' else 'dimensie' for t in types]
	# Creating DataFrame
	result_df = pd.DataFrame({
		'id': top_ids,
		'rel': top_similarity_scores,
		'desc': descriptions,
		'type': types
	})

	return result_df

def retrieve_Measure(query, query_embedding, table_id, n_Measure=10, with_threshold = True):
	associated_df = find_associations(Measure_df, table_id)
	possible_Measure = table_df[table_df['table_id'] == table_id]['Measure'].values[0]

	if len(possible_Measure) == 0:
		return None

	measure_ids = possible_Measure
	dimension_df = associated_df[associated_df['id'].isin(measure_ids)]
	embeddings = dimension_df['embeddings'].to_list()
	scores_cosine = [cosine_similarity([query_embedding], [dimemb])[0][0] for dimemb in embeddings]	

	# scores_cosine = [sentence_transformer_encode(query_embedding, dimemb) for dimemb in embeddings]
	# print(scores_cosine)
	# query_embedding = torch.tensor(query_embedding, dtype=torch.float32, device=device)
	# query_embedding_model = model_measure.query_encoder(query_embedding.unsqueeze(0))
	# scores = []
	# for emb in embeddings:
	# 	emb = torch.tensor(emb, dtype=torch.float32, device=device)
	# 	emb = model_measure.table_encoder(emb.unsqueeze(0))
	# 	score = F.cosine_similarity(query_embedding_model, emb, dim=1)
	# 	scores.append(score.item())

	try:
		titles = [title.lower() for title in dimension_df['title'].to_list()]
		# print(titles)
		tokenized_titles = [title.split() for title in titles]
		tokenized_query = query.lower().split()

		bm25 = BM25Okapi(tokenized_titles)
		doc_scores = bm25.get_scores(tokenized_query)
		bm25_df = pd.DataFrame({'id': dimension_df['id'].values, 'bm_25': doc_scores})
	except:
		bm25_df = pd.DataFrame({'id': dimension_df['id'].values, 'bm_25': [0] * len(dimension_df)})

	# print(len(scores), len(dimension_df))
	scores_df = pd.DataFrame({'id': dimension_df['id'].values, 'emb_score': scores_cosine})
	scores_df = pd.merge(scores_df, bm25_df, on='id')

	scores_df['emb_score'] = scores_df['emb_score'] / (scores_df['emb_score'].max() + 1e-8)
	scores_df['bm_25'] = scores_df['bm_25'] / (scores_df['bm_25'].max() + 1e-8)

	scores_df['score'] = (0.8 * scores_df['emb_score'] + 0.2* scores_df['bm_25'])
	scores_df['rel'] = scores_df['score'] / scores_df['score'].max()
	scores_df = scores_df.sort_values(by=['rel'], ascending=False)
	if with_threshold:
	# Calculate the threshold based on the 95th percentile of 'rel' values
		threshold = np.percentile(scores_df['rel'], 80)
		# Count how many 'rel' values are above the threshold
		n_above_threshold = sum(scores_df['rel'] > threshold)


		
		# Use the smaller of n_Measure or n_above_threshold, but no more than 5
		n_selected_Measure = min(n_Measure, n_above_threshold, 5)
  
	else:
		n_selected_Measure = n_Measure

	descriptions = [associated_df[associated_df['id'] == id]['title'].values[0] for id in scores_df['id']]
	scores_df['desc'] = descriptions
	scores_df.drop(columns=['bm_25', 'emb_score'], inplace=True)

	# Return only the top n_selected_Measure rows
	return scores_df.head(n_selected_Measure)

# def retrieve_Measure(query : str, query_embedding : np.array, table_id : str, n_Measure=8) -> pd.DataFrame:
# 	associated_df = find_associations(Measure_df, table_id)
# 	possible_Measure = table_df[table_df['table_id'] == table_id]['Measure'].values[0]

# 	if len(possible_Measure) == 0:
# 		return None

# 	measure_ids = possible_Measure
# 	dimension_df = associated_df[associated_df['id'].isin(measure_ids)]
# 	embeddings = dimension_df['embeddings'].to_list()
# 	scores = [cosine_similarity(query_embedding.reshape(1, -1), emb.reshape(1, -1))[0][0] for emb in embeddings]

# 	try:
# 		titles = [title.lower() for title in dimension_df['title'].to_list()]
# 		tokenized_titles = [title.split() for title in titles]
# 		tokenized_query = query.lower().split()

# 		bm25 = BM25Okapi(tokenized_titles)
# 		doc_scores = bm25.get_scores(tokenized_query)
# 		bm25_df = pd.DataFrame({'id': dimension_df['id'].values, 'bm_25': doc_scores})
# 	except:
# 		bm25_df = pd.DataFrame({'id': dimension_df['id'].values, 'bm_25': [0] * len(dimension_df)})

# 	scores_df = pd.DataFrame({'id': dimension_df['id'].values, 'emb_score': scores, "embedding":embeddings})
# 	scores_df = pd.merge(scores_df, bm25_df, on='id')

# 	scores_df['emb_score'] = scores_df['emb_score'] / (scores_df['emb_score'].max() + 1e-8)
# 	scores_df['bm_25'] = scores_df['bm_25'] / (scores_df['bm_25'].max() + 1e-8)

# 	scores_df['score'] = (0.8*scores_df['emb_score'] + 0.2*scores_df['bm_25'])
# 	scores_df['rel'] = scores_df['score'] / scores_df['score'].max()
# 	scores_df = scores_df.sort_values(by=['rel'], ascending=False)

# 	# Calculate the threshold based on the 95th percentile of 'rel' values
# 	threshold = np.percentile(scores_df['rel'], 5)
# 	# Count how many 'rel' values are above the threshold
# 	n_above_threshold = sum(scores_df['rel'] > threshold)


	
# 	# Use the smaller of n_Measure or n_above_threshold, but no more than 5
# 	n_selected_Measure = min(n_Measure, n_above_threshold, 5)

# 	descriptions = [associated_df[associated_df['id'] == id]['title'].values[0] for id in scores_df['id']]
# 	scores_df['desc'] = descriptions
# 	scores_df.drop(columns=['bm_25', 'emb_score'], inplace=True)
# 	# print(scores_df.head(n_selected_Measure))
# 	# Return only the top n_selected_Measure rows
# 	return scores_df.head(n_selected_Measure)


def retrieve_dimensions(query, query_embedding, table_id, total_dim=30, max_per_dimension=5, with_threshold = True):
	associated_df = find_associations(Measure_df, table_id)
	possible_dimensions = table_df[table_df['table_id'] == table_id]['DimensionGroups'].values[0]
	dimension_names = list(possible_dimensions.keys())
	max_per_dimesion = int(total_dim / len(dimension_names))
	if len(dimension_names) == 0:
		return None

	all_scores = []

	for dimension_name in dimension_names:
		dimension_ids = possible_dimensions[dimension_name]
		dimension_df = associated_df[associated_df['id'].isin(dimension_ids)]
		embeddings = dimension_df['embeddings'].to_list()
		scores = [cosine_similarity(query_embedding.reshape(1, -1), emb.reshape(1, -1))[0][0] for emb in embeddings]

		try:
			titles = [title.lower() for title in dimension_df['title'].to_list()]
			tokenized_titles = [title.split() for title in titles]
			tokenized_query = query.lower().split()

			bm25 = BM25Okapi(tokenized_titles)
			doc_scores = bm25.get_scores(tokenized_query)
			bm25_df = pd.DataFrame({'id': dimension_df['id'].values, 'bm_25': doc_scores})
		except:
			bm25_df = pd.DataFrame({'id': dimension_df['id'].values, 'bm_25': [0] * len(dimension_df)})

		scores_df = pd.DataFrame({'id': dimension_df['id'].values, 'emb_score': scores, "desc":dimension_df['title'].values})
		scores_df = pd.merge(scores_df, bm25_df, on='id')

		scores_df['emb_score'] = scores_df['emb_score'] / (scores_df['emb_score'].max() + 1e-8)
		scores_df['bm_25'] = scores_df['bm_25'] / (scores_df['bm_25'].max() + 1e-8)

		scores_df['score'] = 0.7* scores_df['emb_score'] + 0.3*  scores_df['bm_25']
		scores_df['score'] = scores_df['score'] / scores_df['score'].max()
		scores_df = scores_df.sort_values(by=['score'], ascending=False)

		if len(scores_df) == 0:
			# print(f"No scores found for dimension {dimension_name}. Skipping this dimension., {table_id}")
			continue
		# Calculate the threshold based on the 95th percentile of 'score' values
  
		if with_threshold:
			threshold = np.percentile(scores_df['score'], 80)
			# Count how many 'score' values are above the threshold
			n_above_threshold = sum(scores_df['score'] > threshold)
			number_of_retrieved_dimensions = min(n_above_threshold, max_per_dimesion)

		else:
			number_of_retrieved_dimensions = max_per_dimesion


		descriptions = [associated_df[associated_df['id'] == id]['title'].values[0] for id in scores_df['id']]
		scores_df[dimension_name] = descriptions
		scores_df.drop(columns=['bm_25', 'emb_score'], inplace=True)

		all_scores.append(scores_df.head(number_of_retrieved_dimensions))

	return all_scores

def retrieve_dimensions_old(query :  str, query_embedding : np.array, table_id : str, total_dim=35, max_per_dimension=5) -> list:
	# print(f"Retrieving dimensions for table {table_id}, {query}")
	associated_df = find_associations(Measure_df, table_id)
	# print(f"The length of the associated df is {len(associated_df)}")
	## find out if table id is in table_df
	table_id = table_id.split("/")[-1]
	table_df_ids = table_df['table_id'].to_list()
	if table_id not in table_df_ids:
		print(f"Table {table_id} not found in table_df")
		return None
	

	possible_dimensions = table_df[table_df['table_id'] == table_id]['DimensionGroups'].values[0]
	dimension_names = list(possible_dimensions.keys())
	# print(f"Possible dimensions: {dimension_names}")
	max_per_dimesion = int(total_dim / len(dimension_names))
	if len(dimension_names) == 0:
		return None

	all_scores = []

	for dimension_name in dimension_names:
		dimension_ids = possible_dimensions[dimension_name]
		dimension_df = associated_df[associated_df['id'].isin(dimension_ids)]
		embeddings = dimension_df['embeddings'].to_list()
		scores = [cosine_similarity(query_embedding.reshape(1, -1), emb.reshape(1, -1))[0][0] for emb in embeddings]
		
# 		if dimension_name == "Geslacht":
# 			print(f"Scores for dimension {dimension_ids}: {dimension_df['title'].values}")


		try:
			titles = [title.lower() for title in dimension_df['title'].to_list()]
			tokenized_titles = [title.split() for title in titles]
			tokenized_query = query.lower().split()

			bm25 = BM25Okapi(tokenized_titles)
			doc_scores = bm25.get_scores(tokenized_query)
			bm25_df = pd.DataFrame({'id': dimension_df['id'].values, 'bm_25': doc_scores})
		except:
			bm25_df = pd.DataFrame({'id': dimension_df['id'].values, 'bm_25': [0] * len(dimension_df)})

		scores_df = pd.DataFrame({'id': dimension_df['id'].values, 'emb_score': scores, "desc":dimension_df['title'].values})
		scores_df = pd.merge(scores_df, bm25_df, on='id')

		scores_df['emb_score'] = scores_df['emb_score'] / (scores_df['emb_score'].max() + 1e-8)
		scores_df['bm_25'] = scores_df['bm_25'] / (scores_df['bm_25'] .max() + 1e-8)

		scores_df['score'] = (scores_df['emb_score']*10 + scores_df['bm_25'])/11
		scores_df['score'] = scores_df['score'] / scores_df['score'].max()
		scores_df = scores_df.sort_values(by=['score'], ascending=False)

		if len(scores_df) == 0:
			print(f"No scores found for dimension {dimension_name}. Skipping this dimension., {table_id}")
			continue
		
		# Filter rows that contain "Totaal" or "totaal" (case-insensitive)
		df_total = scores_df[scores_df['desc'].str.contains("Totaal", case=False)]

		# Exclude rows that contain "Totaal" or "totaal" (case-insensitive) from the original DataFrame
		scores_df = scores_df[~scores_df['desc'].str.contains("Totaal", case=False)]
  
		## Change the score of df_total index to 1
		df_total['score'] = 0.99
		scores_df = pd.concat([scores_df, df_total])
  		## need to find the index in the scores_df that has title "2022" in it, this one we always want to keep in the list
		df_2022 = scores_df[scores_df['desc'].str.contains("2022")]
		scores_df = scores_df[~scores_df['desc'].str.contains("2022")]
  
		## Change the score of df_total index to 1
		df_2022['score'] = 0.99
		scores_df = pd.concat([scores_df, df_2022])

  		# Sort the scores in descending order
		scores_df = scores_df.sort_values(by=['score'], ascending=False)

		# Display scores for a specific dimension if needed
# 		if dimension_name == "Geslacht":
# 			print(f"Scores for dimension {dimension_ids}: {scores_df['desc'].values}, {scores_df['score'].values}")

		# Calculate the threshold as 95% of the maximum score
		max_score = scores_df['score'].max()
		threshold = max_score * 0.4

		# Count how many 'score' values are above the threshold
		n_above_threshold = sum(scores_df['score'] >= threshold)

		# Determine the number of dimensions to retrieve, capped at max_per_dimension
		number_of_retrieved_dimensions = min(n_above_threshold, max_per_dimension)

		# Retrieve the descriptions for the top scores
		descriptions = [associated_df[associated_df['id'] == id]['title'].values[0] for id in scores_df['id']]

		# Add descriptions to the DataFrame and clean up
		scores_df[dimension_name] = descriptions
		scores_df.drop(columns=['bm_25', 'emb_score'], inplace=True)

		# Keep only the top-scoring dimensions based on the threshold
		correct = scores_df.head(number_of_retrieved_dimensions)

		# Append the filtered dimensions to the list
		all_scores.append(correct)
	return all_scores


def create_table_dict(table_id:str, Measure, dimensions):
	# Retrieve the table description based on the table_id from the row
	table_desc = table_df[table_df['table_id'] == table_id]['Table Title'].to_list()[0]

	# Initialize the dictionary template
	dict_template = {
		'table_id': table_id,
		'table_desc': table_desc,
		'table_info': {
			'Measure': {},
			'dimensionGroups': {}
		}
	}

	# Populate the Measure part of the dict
	for m_index, m_row in Measure.iterrows():
		dict_template['table_info']['Measure'][m_row['id']] = m_row['desc']

	# Populate the dimensionGroups part of the dict
	for dim_df in dimensions:
		dimension_group = dim_df.columns[-1]
		for index, row in dim_df.iterrows():
			if dimension_group not in dict_template['table_info']['dimensionGroups']:
				dict_template['table_info']['dimensionGroups'][dimension_group] = {}
			dict_template['table_info']['dimensionGroups'][dimension_group][row['id']] = row[dimension_group]

	return dict_template



if __name__ == "__main__":
	query = input("Question: ")
	expaned = (expand_query(query.split(' ')))
	# print(f'expanded: {expaned}')
	combined = ' '.join(expaned)
	query_embedding = sentence_transformer_encode(combined)

	tables = retrieve_top_tables(query_embedding, n_tables=30)
	reranked_tables = reranked(tables, query_embedding)
	reranked_tables = reranked_tables.sort_values(by=['measure_score', 'rel'], ascending=False)
	total_dicts = []
	for index, row in reranked_tables[:5].iterrows():
		print(f"\nIn table {row['table_id']} we found the following:\n")
		# s = get_similarity_scores(query, query_embedding, row['table_id'])