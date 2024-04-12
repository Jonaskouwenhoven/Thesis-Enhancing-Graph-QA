from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import torch
import pandas as pd

from tqdm import tqdm 
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from transformers import LongformerModel, LongformerTokenizer
import json
import random
import requests
import traceback
# from create_short_statline_url import create_short_url
import pandas as pd
import json
import requests
from collections import defaultdict
from transformers import GPT2Tokenizer
import json
import openai
import warnings
warnings.filterwarnings("ignore")
import os
from openai import OpenAI
openai.api_key = "ADD_YOUR_API_KEY"
os.environ["OPENAI_API_KEY"] = "ADD_YOUR_API"
client = OpenAI()


def get_info(table_id: str) -> str:
	
	
	url = f"https://opendata.cbs.nl/ODataFeed/odata/{table_id}/TableInfos?$format=json"
	response = requests.get(url)
	
	
	jsonfile = json.loads(response.text)['value'][0]
	text = jsonfile['ShortTitle'] + " " + jsonfile['Summary'] + " " + jsonfile['ShortDescription'] +" " + jsonfile['Description'] + " " 

	return jsonfile['ShortTitle']

def fetch_table_data(table_id: str) -> dict:
	title_codes = defaultdict(dict)
	errors = 0
	try:
		table_url = "https://odata4.cbs.nl/CBS/" + table_id
		r = requests.get(table_url).json()
		urls = [d['url'] for d in r['value']]

		for url in urls:
			if url in ['Observations', 'Properties', 'Dimensions']:
				continue
			req_url = table_url + "/" + url
			data = requests.get(req_url).json()['value']
			for entry in data:
				if 'Identifier' in entry:
					if 'MeasureGroupId' in entry and entry['MeasureGroupId']:
						title_codes[entry['Identifier']] = {
							'Title': title_codes[entry['MeasureGroupId']]['Title'] + "_" + entry['Title'] + "--(" + entry['Unit'].replace("1 000", "1000") + ")",
							'Description': entry['Description'].replace("\r\n", " ") if entry['Description'] else ""
						}
					else:
						title_codes[entry['Identifier']] = {
							'Title': entry['Title'] + "--(" + entry['Unit'].replace("1 000", "1000") + ")",
							'Description': entry['Description'].replace("\r\n", " ") if entry['Description'] else ""
						}
				elif 'Id' in entry:
					if 'ParentId' in entry and entry['ParentId']:
						title_codes[entry['Id']] = {
							'Title': title_codes[entry['ParentId']]['Title'] + "_" + entry['Title'],
							'Description': entry['Description'].replace("\r\n", " ") if entry['Description'] else ""
						}
					else:
						title_codes[entry['Id']] = {
							'Title': entry['Title'],
							'Description': entry['Description'].replace("\r\n", " ") if entry['Description'] else ""
						}
	except Exception as e:
		errors += 1
		# print(f"Error processing table {table_id}: {e}")

	# print("Errors encountered:", errors)
	return dict(title_codes)


from SPARQLWrapper import SPARQLWrapper, JSON
import itertools
import operator

REPOSITORY = 'odata4'
sparql = SPARQLWrapper(f"http://4.231.28.187:7200/repositories/{REPOSITORY}")
sparql.setCredentials("cerborix", "2Jt@d&M5g")
sparql.setReturnFormat(JSON)

def fetch_table_values(table_id: str) -> (dict, dict):
	"""Fetch the nodes and tables from the table."""
	query = (f"""
			PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
			PREFIX dct: <http://purl.org/dc/terms/>
			PREFIX qb: <http://purl.org/linked-data/cube#>
			
			SELECT ?o ?label WHERE {{ 
				?s qb:dimension ?o .
				?s dct:identifier ?id .
				OPTIONAL {{ ?o skos:prefLabel|skos:altLabel|skos:definition|dct:description|dct:subject ?label }}
				FILTER (?id = "{table_id}")
				FILTER (!BOUND(?label) || lang(?label) = "nl")
			}}
		""")
	sparql.setQuery(query)
	sparql.setReturnFormat(JSON)
	try:
		result = sparql.query().convert()['results']['bindings']
		props = [(r['o']['value'], (r.get('label', {}) or {'value': ''})['value']) for r in result]
		nodes = {}
		tables = {f"https://opendata.cbs.nl/#/CBS/nl/dataset/{table_id}": {'body': ''}}
		it = itertools.groupby(sorted(props, key=operator.itemgetter(0)), operator.itemgetter(0))
		for key, subiter in it:
			val = ' '.join(prop[1] for prop in subiter)
			nodes[key] = {'body': val, 'type': 'node'}
			tables[f"https://opendata.cbs.nl/#/CBS/nl/dataset/{table_id}"]['body'] += ' ' + val
		return nodes, tables
	except Exception as e:
		print(f"Failed to fetch table nodes: {e}")
		return {}, {}




def expressiontoprompt(expression:str, measure:str, dimensions: list) -> str:
	"""Create a prompt based on the table information and rows and columns.

	Args:
		expression (str): s-expression
		measure (str): measure id
		dimensions (list): list of dimensions

	Returns:
		str: prompt
	"""
	table_id = expression.split("(VALUE (")[1].split(" ")[0]
	table_title = get_info(table_id)
	# print(table_title)
	prompt = "In hoeverree behoort de vraag bij de tabel met omschrijving: " + table_title + ". "
	nodes, tables = fetch_table_values(table_id)
	ids, desc = [], []
	for node in nodes:
		id = node.split("/")[-1]
		ids.append(id)
		desc.append(nodes[node]['body'])

	temp_df = pd.DataFrame({'id': ids, 'desc': desc})
	thing = fetch_table_data(table_id)

	df = pd.DataFrame(thing)
	dft  = df.T
	ids = list(df.columns)
	dft['New_ids'] = ids

	prompt += f"Met de volgende eigenschappen: {dft[dft['New_ids'] == measure]['Title'].values[0]}, "
	
	prompt += "Met de volgende eigenschappen:"
	# print(dimensions)
	for dim in dimensions:
		# print(dim)
		prompt += dim[0] + " " + temp_df[temp_df['id'] == dim[1]]['desc'].values[0] + ", "
		
	return prompt

	
def create_instance(prompt:str, completion:str)-> dict:
	"""Create a training instance for the chatbot.

	Args:
		prompt (str): prompt
		completion (str): output

	Returns:
		dict: training instance
	"""
		
	SYSTEMMESSAGE = "In het nederlands genereer jij een query op basis van tabel informatie"

	return {"messages": [{"role": "system", "content":SYSTEMMESSAGE}, {"role": "user", "content": prompt}, {"role": "assistant", "content": completion}]}


def exact_openai_token_count(text: str) -> int:
	"""Count the number of tokens in the input text.

	Args:
		text (str): string to count tokens for

	Returns:
		int: number of tokens in the input text
	"""
	# Initialize the GPT-2 tokenizer
	tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

	# Tokenize the text and count the number of tokens
	tokens = tokenizer.encode(text)

	return len(tokens)



def chatgptSexpressions(query:str, model="gpt-3.5-turbo-0125") -> str:
	"""Generate a query based on the table information and rows and columns.
	
	Args:
		query (str): The query that best fits the table description
		model (str): The model to use for generating the query
		
	Returns:
		str: The generated query
	"""

	template = f"""
	You generate queries based on table information, and rows and columns, that answer that question here is the information : {query}
	- `query`: The query that best fits the table description

 	"""


	input_tokens = exact_openai_token_count(template)
	input_cost = input_tokens / 1000 * 0.0005    # Cost per 1K tokens for input

	# print(template)
	
	response = client.chat.completions.create(
		model=model,
		# model = 'gpt-4-0125-preview',
		# response_format={"type": "json_object"},
		messages=[
			{"role": "system", "content": "In het nederlands genereer jij een query op basis van tabel informatie"},
			{"role": "user", "content": query}
		],
		temperature=0.8,

	)
	print(response)
	output = response.choices[0].message.content
	output_tokens = exact_openai_token_count(f"{output}")
	output_cost = output_tokens / 1000 * 0.0015  # Cost per 1K tokens for output

	# Calculate the total cost by adding input and output costs
	total_cost = input_cost + output_cost

	# Print the costs
	print(f"Input tokens: {input_tokens}, Cost: ${input_cost:.4f}")
	print(f"Output tokens: {output_tokens}, Cost: ${output_cost:.4f}")
	print(f"Total cost: ${total_cost:.4f}")
	return output




# Function to calculate embeddings using Longformer
def calculate_longformer_embeddings(text, model_name='allenai/longformer-base-4096'):
    # Load the tokenizer and model
    tokenizer = LongformerTokenizer.from_pretrained(model_name)
    model = LongformerModel.from_pretrained(model_name)

    # Encode text input (add special tokens and convert to ids)
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=4096)

    # Longformer's attention mask (1s for all tokens, as we want to attend to all)
    inputs['attention_mask'] = torch.ones(inputs.input_ids.shape)

    # Forward pass, get hidden states
    outputs = model(**inputs)

    # The last hidden state is the sequence of hidden states at the output of the last layer
    last_hidden_states = outputs.last_hidden_state

    # You can pool the output into a single embedding (e.g., mean pooling)
    embeddings = torch.mean(last_hidden_states, dim=1)

    return embeddings.detach().numpy()[0]


def add_tensor_embedding_column(df, column_name):
	"""
	Add a new column 'tensor_embedding' to the DataFrame 'df' containing tensor embeddings of the specified column.

	Args:
		df (pd.DataFrame): The input DataFrame.
		column_name (str): The name of the column for which to create tensor embeddings.

	Returns:
		pd.DataFrame: The updated DataFrame with the 'tensor_embedding' column.
	"""
	df['tensor_embedding'] = df[column_name].apply(lambda embed: torch.tensor(embed))
	return df



class TripletDataset(Dataset):
	def __init__(self, anchor_embeddings, positive_embeddings, negative_embeddings):
		self.anchor_embeddings = anchor_embeddings
		self.positive_embeddings = positive_embeddings
		self.negative_embeddings = negative_embeddings

	def __len__(self):
		return len(self.anchor_embeddings)

	def __getitem__(self, idx):
		return {
			'anchor': self.anchor_embeddings[idx],
			'positive': self.positive_embeddings[idx],
			'negative': self.negative_embeddings[idx]
		}
	


def convert_triplet_embeddings(df, anchor_col, positive_col, negative_col):

	anchor_tensors = [torch.tensor(embed) for embed in df[anchor_col]]
	positive_tensors = [torch.tensor(embed) for embed in df[positive_col]]
	negative_tensors = [torch.tensor(embed) for embed in df[negative_col]]

	return anchor_tensors, positive_tensors, negative_tensors

# Example usage
# Assuming 'df_triplets' is your DataFrame with columns 'anchor_embedding', 'positive_embedding', 'negative_embedding'

def find_top_similar_Measure(embedding, df, k = 10, ascending_param = True, column = "Embeddings"):
	# Calculate cosine similarity between the input embedding and all rows in the DataFrame
	cosine_similarities = cosine_similarity([embedding], df[column].tolist())[0]

	# Create a new DataFrame with table_id and cosine_similarity columns
	similarity_df = pd.DataFrame({'table_id': df['table_id'], 'cosine_similarity': cosine_similarities})

	# Sort the DataFrame by cosine similarity in descending order
	similarity_df = similarity_df.sort_values(by='cosine_similarity', ascending=ascending_param)

	# Get the top two table IDs with the highest similarity scores
	top_similar_tables = similarity_df.head(k)


	return top_similar_tables

def kmeans_cluster(document_ids, floats, n_clusters=5):
	data = np.asarray(floats).reshape(-1, 1)
	# Create and fit the K-Means model
	kmeans = KMeans(n_clusters=n_clusters)
	kmeans.fit(data)

	# Get the cluster labels and cluster centers
	cluster_labels = kmeans.labels_
	cluster_centers = kmeans.cluster_centers_

	# Create a dictionary to store document_ids for each cluster label
	cluster_dict = {label: [] for label in range(n_clusters)}
	for i, label in enumerate(cluster_labels):
		cluster_dict[label].append(document_ids[i])

	# Find the cluster with the highest center value
	max_center_index = np.argmax(cluster_centers)

	# Get the document_ids for the cluster with the highest center value
	document_ids_for_max_cluster = cluster_dict[max_center_index]
	# print(document_ids_for_max_cluster)
	return document_ids_for_max_cluster,np.max(cluster_centers)

def find_top_similar_Measure2(embedding, df, k = 10, ascending_param = True, column = "Embeddings"):
	# Calculate cosine similarity between the input embedding and all rows in the DataFrame
	# print(embedding, type(embedding), df[column].tolist())
	cosine_similarities = cosine_similarity([embedding], df[column].tolist())[0]

	# Create a new DataFrame with table_id and cosine_similarity columns
	similarity_df = pd.DataFrame({'id': df['id'], 'cosine_similarity': cosine_similarities})

	# Sort the DataFrame by cosine similarity in descending order
	similarity_df = similarity_df.sort_values(by='cosine_similarity', ascending=ascending_param)

	# Get the top two table IDs with the highest similarity scores
	top_similar_tables = similarity_df.head(k)


	return top_similar_tables

def create_dataloader(df, batch_size = 64):
	train_anchor_embeddings = torch.stack(df['anchor_embedding'].tolist())
	train_positive_embeddings = torch.stack(df['positive_embedding'].tolist())
	train_negative_embeddings = torch.stack(df['negative_embedding'].tolist())



	# Create datasets
	dataset = TripletDataset(train_anchor_embeddings, train_positive_embeddings, train_negative_embeddings)
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

	return dataset, dataloader



def find_top_similar_tables(embedding, df, k = 10, ascending_param = True):
	# Calculate cosine similarity between the input embedding and all rows in the DataFrame
	cosine_similarities = cosine_similarity([embedding], df['embeddings'].tolist())[0]

	# Create a new DataFrame with table_id and cosine_similarity columns
	similarity_df = pd.DataFrame({'table_id': df['table_id'], 'cosine_similarity': cosine_similarities})

	# Sort the DataFrame by cosine similarity in descending order
	similarity_df = similarity_df.sort_values(by='cosine_similarity', ascending=ascending_param)

	# Get the top two table IDs with the highest similarity scores
	top_similar_tables = similarity_df.head(k)


	return top_similar_tables

def get_table_similarity_scores(model, table_embeddings, table_ids, query_embedding):
	"""
	This function calculates cosine similarity scores between a query embedding and multiple table embeddings,
	after processing them through the model.

	Args:
	- model: The PyTorch model for embedding comparison
	- table_embeddings: List of embeddings for each table
	- table_ids: List of IDs for each table
	- query_embedding: The embedding of the query

	Returns:
	- List of tuples (table_id, cosine_similarity_score)
	"""
	results_tablem, scores = [],[]

	query_emb_tensor = torch.tensor(query_embedding).unsqueeze(0).to(device)
	query_emb_output = model.forward_once(query_emb_tensor)

	for table_id, table_emb in zip(table_ids, table_embeddings):
		table_emb_tensor = torch.tensor(table_emb).unsqueeze(0).to(device)
		table_emb_output = model.forward_once(table_emb_tensor)

		cosine_similarity = F.cosine_similarity(query_emb_output, table_emb_output, dim=1).item()
		results_tablem.append(table_id)
		scores.append(cosine_similarity)

	return results_tablem, scores


class TripletLoss(nn.Module):
	def __init__(self, margin=0.3):
		super(TripletLoss, self).__init__()
		self.margin = margin

	def forward(self, anchor, positive, negative):
		pos_dist = F.pairwise_distance(anchor, positive, 2)
		neg_dist = F.pairwise_distance(anchor, negative, 2)
		loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
		return loss.mean()


	

class CosineTripletLoss(nn.Module):
	def __init__(self, margin=0.1, s=30.0):
		super(CosineTripletLoss, self).__init__()
		self.margin = margin
		self.s = s

	def forward(self, anchor, positive, negative):
		# Normalize the vectors to have unit length
		anchor = F.normalize(anchor, p=2, dim=1)
		positive = F.normalize(positive, p=2, dim=1)
		negative = F.normalize(negative, p=2, dim=1)

		# Calculate the cosine similarity for positive and negative pairs
		pos_similarity = torch.sum(anchor * positive, dim=1)
		neg_similarity = torch.sum(anchor * negative, dim=1)

		# Compute the loss according to the provided formula
		loss = -torch.log(
			torch.exp(self.s * (pos_similarity - self.margin)) /
			(torch.exp(self.s * (pos_similarity - self.margin)) + torch.exp(self.s * neg_similarity))
		)

		return loss.mean()



def calculate_accuracy(model, test_df, table_df, top_k=[1, 5, 10], x=1000, emb_to_train="embedding"):
	"""
	Calculate accuracy for the given model and test DataFrame.

	Args:
	- model (nn.Module): The neural network model.
	- test_df (pd.DataFrame): Test DataFrame.
	- table_df (pd.DataFrame): DataFrame containing le data with embeddings.
	- device (str): The device (e.g., "cuda" or "cpu") to use for computations.
	- top_k (list): List of top-k values to calculate accuracy for.
	- x (int): Number of rows to process from the test DataFrame.

	Returns:
	- dict: Dictionary containing accuracy values for each top-k value.
	"""
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model.eval()
	table_identifiers = table_df['table_id'].to_list()
	table_embeddings = table_df[emb_to_train].to_list()
	accuracy_dict = {k: 0 for k in top_k}

	# Pre-calculate all table encodings
	table_embeddings_tensors = [torch.tensor(emb).unsqueeze(0).to(device) for emb in table_embeddings]
	encoded_tables = [model.table_encoder(emb.to(torch.float32)) for emb in table_embeddings_tensors]

	for index, row in tqdm(test_df[:x].iterrows(), total=len(test_df[:x])):
		query_text = row['query']
		query = row['embedding']
		golden_table = row['table_id']
		query_emb_tensor = torch.tensor(query).unsqueeze(0).to(device)
		cosine_similarities = []

		output_query = model.query_encoder(query_emb_tensor.to(torch.float32))

		for encoded_table in encoded_tables:
			cosine_similarity = F.cosine_similarity(output_query, encoded_table, dim=1)
			cosine_similarities.append(cosine_similarity.item())

		top = sorted(range(len(cosine_similarities)), key=lambda i: cosine_similarities[i], reverse=True)[:]

		top_table_ids = [table_identifiers[i] for i in top]

		for k in top_k:
			if golden_table in top_table_ids[:k]:
				accuracy_dict[k] += 1

	total_rows = min(x, len(test_df))
	accuracy_dict = {k: v / total_rows for k, v in accuracy_dict.items()}

	return accuracy_dict

def retrieve_top_10(model, test_df, table_df, top_k=[1, 5, 10], x=10):
	"""
	Calculate accuracy for the given model and test DataFrame.

	Args:
	- model (nn.Module): The neural network model.
	- test_df (pd.DataFrame): Test DataFrame.
	- table_df (pd.DataFrame): DataFrame containing le data with embeddings.
	- device (str): The device (e.g., "cuda" or "cpu") to use for computations.
	- top_k (list): List of top-k values to calculate accuracy for.
	- x (int): Number of rows to process from the test DataFrame.

	Returns:
	- dict: Dictionary containing accuracy values for each top-k value.
	"""
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model.eval()
	table_identifiers = table_df['table_id'].to_list()
	table_embeddings = table_df['embedding'].to_list()
	accuracy_dict = {k: 0 for k in top_k}

	# Pre-calculate all table encodings
	table_embeddings_tensors = [torch.tensor(emb).unsqueeze(0).to(device) for emb in table_embeddings]
	encoded_tables = [model.table_encoder(emb.to(torch.float32)) for emb in table_embeddings_tensors]
	top_10s = []
	for index, row in tqdm(test_df[:x].iterrows(), total=len(test_df[:x])):
		query_text = row['query']
		query = row['embedding']
		golden_table = row['table_id']
		query_emb_tensor = torch.tensor(query).unsqueeze(0).to(device)
		cosine_similarities = []

		output_query = model.query_encoder(query_emb_tensor.to(torch.float32))

		for encoded_table in encoded_tables:
			cosine_similarity = F.cosine_similarity(output_query, encoded_table, dim=1)
			cosine_similarities.append(cosine_similarity.item())

		top = sorted(range(len(cosine_similarities)), key=lambda i: cosine_similarities[i], reverse=True)[:50]
		##
		sorted_cosine_similarities = sorted(cosine_similarities, reverse=True)
		top_table_ids = [table_identifiers[i] for i in top]
		temp_dict = {"query": query_text, "golden_table": golden_table, "top_10": top_table_ids[:50], "top_10_scores": sorted_cosine_similarities[:50]}
		top_10s.append(temp_dict)
		
	return top_10s
