
import config
from odata_graph.sparql_controller import SparqlEngine
from pipeline.entity_retriever import EntityRetriever
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import config
from odata_graph.sparql_controller import SparqlEngine
import numpy as np
from transformers import AutoTokenizer
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch

from copy import deepcopy
engine = SparqlEngine(local=False)
er = EntityRetriever(engine=engine)
import pandas as pd


enc_tokenizer = AutoTokenizer.from_pretrained("unsloth/mistral-7b-bnb-4bit")

base_model = AutoModelForCausalLM.from_pretrained("unsloth/mistral-7b-bnb-4bit", torch_dtype="auto")
base_model.resize_token_embeddings(len(enc_tokenizer))
# model = base_model
model = PeftModel.from_pretrained(base_model, "JoniJoniAl/diversetraining12maartlarger")
enc_tokenizer.sep_token = enc_tokenizer.unk_token
enc_tokenizer.sep_token_id = enc_tokenizer.unk_token_id


tabledf = pd.read_pickle("data/tabledf.pkl")
tabledf = tabledf.drop_duplicates(subset=['table_id'])

last_token_added = enc_tokenizer.bos_token

import utils.sexpressions as SExpression

import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Argument Parser for Entity Retrieval")
    parser.add_argument('--entityretrieval', type=str, choices=["colbert", "dualencoder", "bm25"], default='colbert', help='Entity retrieval engine (default: colbert)')

    return parser.parse_args()

def construct_graph_string(data):
    graph_string = ""
    table_id = data['table_id']
    table_desc = data['table_desc']
    Measure = data['table_info']['Measure']
    dimension_groups = data['table_info']['dimensionGroups']
    graph_string+= f"Table ID: {table_id}\nWith description: {table_desc}\nhas the following properties:\n"
    # Add nodes
    graph_string += "Nodes:\n"
    for measure_id, measure_desc in Measure.items():
        # print(measure_desc)
        graph_string += f"- {measure_id}: {(measure_desc)}\n"
    for group_name, group_data in dimension_groups.items():
        graph_string += f"- {group_name}:\n"
        for dimension_id, dimension_desc in group_data.items():
            graph_string += f"  - {dimension_id}: {dimension_desc}\n"
    return graph_string



def rename_dataframe_columns(nodes_dataframe):
    
    new_ids = []
    for index, row in nodes_dataframe.iterrows():
        id = row['id']
        if 'cbs' in id:
            new_ids.append(id.split("/")[-1])

        else:
            new_ids.append(id)

    nodes_dataframe['id'] = new_ids
    return nodes_dataframe



def normalize_and_merge_scores(model_scores_df, ranked_nodes_df):
    """
    Normalize the model scores and merge them with the retriever scores.
	"""

    # normalize model scores (Very important and Strange way to normalize)
    model_scores_df['scores'] = [1/abs(x + 1e-8) for x in model_scores_df['scores']]
    model_scores_df['normalized_scores'] = model_scores_df['scores'] / model_scores_df['scores'].max()
    # model_scores_df['normalized_scores'] = (model_scores_df['scores'] - min_score) / (max_score - min_score) 

    # Initialize an empty list for retriever scores
    retriever_scores = []

    # Iterate through model_scores_df to retrieve scores from ranked_nodes_df
    for index, row in model_scores_df.iterrows():
        id = row['tokens']
        
        # Check if the id is present in ranked_nodes_df and append the score, otherwise append 0
        if id in ranked_nodes_df['id'].values:
            retriever_scores.append(ranked_nodes_df[ranked_nodes_df['id'] == id]['score'].values[0])
        else:
            retriever_scores.append(0)
    
    # Add the retriever scores to the DataFrame
    model_scores_df['retriever_scores'] = retriever_scores
    
    # Normalize the retriever scores
    model_scores_df['retriever_scores'] = model_scores_df['retriever_scores'] / model_scores_df['retriever_scores'].max()
    # model_scores_df['retriever_scores'] = (model_scores_df['retriever_scores'] - model_scores_df['retriever_scores'].min()) / (model_scores_df['retriever_scores'].max()- meodel_scores_df['retriever_scores'].min())
    
    # Compute the final scores as the average of normalized_scores and retriever_scores
    model_scores_df['final_scores'] = (model_scores_df['normalized_scores'] *0.8 + model_scores_df['retriever_scores'] * 0.2) 
    
    # Sort the DataFrame by final_scores in descending order
    model_scores_df = model_scores_df.sort_values(by='final_scores', ascending=False)
    
    # Print the DataFrame after normalization and merging
    return model_scores_df


def find_top_output_sequences(model, input_text, ids):
    # Tokenize the input sequence
    ## make texts tensor
    input_ids = torch.tensor(input_text).unsqueeze(0)

    # Tokenize each output sequence and calculate the log probabilities
    log_probs = []
    for output_text in ids:
        # Tokenize the output sequence
        output_ids = torch.tensor(output_text).unsqueeze(0)
        input_output_ids = torch.cat((input_ids, output_ids), dim=-1)

        # Get the logits from the model
        with torch.no_grad():
            outputs = model(input_output_ids)
        logits = outputs.logits

        # Slice the logits to get the relevant output token positions
        output_logits = logits[0, input_ids.size(1) - 1:-1, :]

        # Apply softmax to get the probabilities
        probs = torch.softmax(output_logits, dim=-1)

        # Take the logarithm of the probabilities to get the log probabilities
        log_probs.append(torch.sum(torch.log(probs[range(len(output_ids[0])), output_ids[0]])).item())

    # Find the indices of the output sequences with the highest probabilities
    top_indices = sorted(range(len(log_probs)), key=lambda i: log_probs[i], reverse=True)

    # Create a list of tuples containing the output sequences and their log probabilities
    top_output_sequences = [(ids[i], log_probs[i]) for i in top_indices]
    
    return top_output_sequences



def conditional_generation(ranked_nodes, subgraph, table_dict, query, tokenizer, model, nodes_dataframe, not_allowed_table_ids = []):
		"""
			Inference function to generate an S-expression based on a given query
			using constrained beam search. Returns the best scoring generated valid
			S-expression and the ranked candidate nodes the expression is based on.

			:param query: query string to inference
			:param verbose: print status updates in console
			:returns: tuple containing a valid S-expression and the ranked MSR/DIM nodes it is based on
		"""
	
		if len(ranked_nodes) == 0:
			return None, {}  # TODO: handle properly
		tab = table_dict
		table_dict = str(table_dict).replace("(", " ")
		alpaca_prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. ### Instruction: {{}} ### Input: {{}} ### Response:{tokenizer.bos_token}"""
		
		string_temp = ""
		for element in tab:
			string_temp += f"{construct_graph_string(element)}\n"

		instruction = "You generate S-expression based on retrieved table information, reason over the information in the table and structure the S-expression with the correct IDs"
		
		table_dict_string =f"Use the information in {string_temp} to generate an S-expression that answers the question: {query}" 
		string = alpaca_prompt.format(
				instruction, # instruction
				table_dict_string, # input
			 # output - leave this blank for generation!
			)
		
		string = string.replace("'", " ")

		encodings = tokenizer(string)
		# Convert input_ids and attention_mask to numpy arrays
		input_ids = np.asarray(encodings['input_ids'])
		attention_mask = np.asarray(encodings['attention_mask'])

		input_ids = input_ids.reshape(1, -1)  # Reshape to [1, sequence_length]
		attention_mask = attention_mask.reshape(1, -1)  # Reshape to [1, sequence_length]

		# Convert them to PyTorch tensors
		input_ids_tensor = torch.from_numpy(input_ids)
		attention_mask_tensor = torch.from_numpy(attention_mask)

		# Assign the tensors back to encodings
		encodings['input_ids'] = input_ids_tensor
		encodings['attention_mask'] = attention_mask_tensor

		# sexp_beam_cache: Dict[int, SExpression] = {hash(''): SExpression(subgraph)}
		sexp = SExpression.SExpression(tab, query, subgraph)
		
		ranked_nodes_df = rename_dataframe_columns(nodes_dataframe)

		def _restrict_decode_vocab(batch_id, prefix_beam) -> list[int]:
			"""
			Helper function to get admissible tokens during beam search for generating the S-expression.

			:param batch_id: not used
			:param prefix_beam: generated beam so far
			:returns list containing token ids of admissible tokens
			"""
			global last_token_added
			beam_str = tokenizer.convert_ids_to_tokens(prefix_beam)
			try:
				res = len(beam_str) - 1 - beam_str[::-1].index(last_token_added)
			except ValueError:
				res = 0

			if 0 <= res < len(beam_str):
				beam_str = beam_str[res:] 
			else:
				beam_str = []

			## try and make this work
			output_admissibletoken = sexp.get_admissible_tokens()

			if len(output_admissibletoken) > 1:
				if len(output_admissibletoken) == 3:
					admissible_tokens, new_prompt, dict = output_admissibletoken

				if len(output_admissibletoken) == 4:
					admissible_tokens, new_prompt, dict, table_branch_node = output_admissibletoken
				else:
					admissible_tokens, new_prompt = output_admissibletoken
					dict = None
					table_branch_node = False
			else:
				admissible_tokens = output_admissibletoken
				new_prompt = None
				dict = None
				table_branch_node = False


			if table_branch_node:
				## we need to remove the table ids that are not allowed
				admissible_tokens = [t for t in admissible_tokens if t not in not_allowed_table_ids]




			if len(admissible_tokens) == 0:
				return [tokenizer.pad_token_id], None , None, False
			# admissible_tokens = sexp.get_admissible_tokens()
			if len(admissible_tokens) > 1 and "DIM" in admissible_tokens:
				admissible_tokens = ['DIM']

			if len(admissible_tokens) >= 1 and "OR" in admissible_tokens: ## remove OR
				## remove OR
				admissible_tokens = [t for t in admissible_tokens if t != "OR"]	

			if "WHERE" in admissible_tokens:
				admissible_tokens = ["WHERE"]


			return tokenizer(list(admissible_tokens), add_special_tokens=False)['input_ids'], new_prompt, dict, table_branch_node

		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		def prefix_allowed_tokens_func(batch_id, sent):
			prefix_beam = sent.tolist()
			admissible_tokens_list, new_prompt, dict, table_branche_node = _restrict_decode_vocab(batch_id, prefix_beam)


			if len(admissible_tokens_list) == 1:
				decoded_token = tokenizer.decode(admissible_tokens_list[0])

				if decoded_token == "<unk>":
					return decoded_token, sexp.__str__(), True, 0 
				sexp.add_token(decoded_token)

				return decoded_token, sexp.__str__(), False, 0


			if new_prompt is not None:

				check = tokenizer(new_prompt)['input_ids']
			else:
				check = prefix_beam
			top_output = find_top_output_sequences(model, check, admissible_tokens_list)

			scores = [score for tokens, score in top_output]
			admissible_tokens_list = [tokens for tokens, score in top_output]
			admissibleIDS = [tokenizer.decode(token) for token in admissible_tokens_list]
			model_scores_df = pd.DataFrame({"scores":scores, "tokens":admissibleIDS})
			model_scores_df = normalize_and_merge_scores(model_scores_df, ranked_nodes_df)


			max_score_index = torch.argmax(torch.tensor(scores))
			max_score_tokens = admissible_tokens_list[max_score_index]
			decoded_max_score_tokens = model_scores_df['tokens'].values[0]

			
			if "<s> " in decoded_max_score_tokens:
				decoded_max_score_tokens = decoded_max_score_tokens.replace("<s> ", "")

			sexp.add_token(decoded_max_score_tokens)
			return max_score_tokens, sexp.__str__(), False, max(scores)


		prompt = encodings['input_ids'][0]


		scores_generated = []
		while True:
			# Get the list of admissible tokens for the current prompt
			hightokens, sexp_String, finished, score = prefix_allowed_tokens_func(0, prompt)
			sexp_Tokens = tokenizer.encode(sexp_String)
			scores_generated.append(score)
			if len(hightokens) == 0:
				break

			if finished:
				break


			prompt = torch.cat([prompt, torch.tensor(sexp_Tokens)], dim=-1)
		return sexp.__str__(), prompt


if __name__ == "__main__":
    args = parse_arguments()

    if args.entityretrieval == "dualencoder":
        colbert_flag = False
        bm25_flag = False
    elif args.entityretrieval == "colbert":
        colbert_flag = True
        bm25_flag = False
		
    elif args.entityretrieval == "bm25":
        colbert_flag = False
        bm25_flag = True

    entity_retriever = EntityRetriever(engine=engine, colbert=colbert_flag, bm25=bm25_flag)

    while True:
        query = input("Enter a query: ")
        ranked_nodes, subgraph, table_dict, nodes_dataframe, tables = \
            entity_retriever.get_candidate_nodes_better(query, full_graph=True)
        generated_sexp = conditional_generation(ranked_nodes, subgraph, table_dict, query, enc_tokenizer, model, nodes_dataframe)[0]

        print(f"Generated S-expression: {generated_sexp}")