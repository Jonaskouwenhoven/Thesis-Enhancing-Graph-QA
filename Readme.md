
# Enhancing Knowledge Graph QA through constrained Dynamic Decoding and Low Rank Adaptation

This code repostitory contains the code the my thesis. 

# Installation


```python
pip install -r requirements.txt
```


# Dataset Documentation

In order to run this code you need to download the relevant data sets from this [Link](https://drive.google.com/file/d/12XhF587bQhpAQovlVp01Ixj6Ej0gmx2A/view?usp=sharing)

## Table Dataframe


The file data/tabledf.pkl holds a  set of information and embeddings about the different tables used in this research. It's organized to give a detailed view of each table, such as its title, summary, embeddings linked to it, Measure, and dimension groups.


### Dataset Schema

The dataset comprises the following columns:

- `table_id`: A unique identifier for each table.
- `Table Title`: The name or title of the table.
- `Summary`: A openai-generated summary of the table's content.
- `embeddings`: The embedding vector associated with the generated summary
- `Measure`: The measure(s) associated with the table
- `DimensionGroups`: The groups of dimensions represented in the table

## Measure and Dimensions Dataframe

The dataset named measure_dim_df.pkl contains details about the Measures and dimensions in the different tables, along with their respective IDs, titles, and embeddings. The column table_ids lists the tables where each measure or dimension is used. The embeddings are created by converting the titles to lowercase using lower() function and utilizing GroNLP embeddings.

### Dataset Schema

The structure of the dataset is outlined as follows:

- `id`: A unique identifier for each measure or dimension.
- `title`: The name or title of the measure or dimension.
- `table_ids`: A list of table IDs where the measure or dimension is present, linking back to the tables they are used in.
- `embeddings`: The GroNLP embeddings generated from the lowercased titles, providing a vector representation for each title.

## Query Dataframe


The dataset called querydf contains queries either manually annotated or generated based on the different tables, along with their corresponding S-expression (original_sexp), (Query), table ID (table_id), embeddings (Embeddings), prompts (Prompts), Measure (Measure), and dimensions (Dimension). If the value in the Prompts column of a row is "manual," it signifies that the query is derived from a manually annotated data annotation process.
### Dataset Schema

- `original_sexp`: The original s-expression of the query.
- `Query`: The natural language representation of the s-expression
- `table_id`: The correct table identifier
- `Embeddings`: The GroNLP embeddings
- `Prompts`: Indicates the origin of the query, with "manual" signifying manually annotated queries.
- `Measure`: The measure associated with the query.
- `Dimension`: The dimension associated with the query.

## TestDf and TrainDf

The datasets testdf and traindf split the T500 dataset for experimental use, acting as the basis for model training and evaluation. 


## OODdf

The Out-Of-Domain (OOD) dataset, `OODdf`, is used to evaluate the models perfomance on out-of-domain data.


## T138

The T138 dataset is used for experiments on the T138 dataset. This dataset will be divided into training and testing sets using a test size of 0.2 and a random state of 42 .

### Preparing the Dataset

To split the dataset into training and testing sets, use the following approach:

```python
from sklearn.model_selection import train_test_split

# Assuming T138_df is your DataFrame
train_df, test_df = train_test_split(T138_df, test_size=0.2, random_state=42)

# Now, train_df and test_df are ready for training and testing respectively.
```

# Retrieval Train Documentation

This repository contains code to train three sets of encoders; ColBERT, Query and Dual encoder

## Training Dual and Query Encoder

### Command-line Arguments

- `--model`: Specifies the model to train. Choose `DualEncoder` for a dual encoder model or `QueryEncoder` for a query encoder model. The default is `QueryEncoder`.
- `--hnm_technique`: Defines the hard negative mining technique to use for selecting hard negatives. Options include `topn`, `bm25`, `random`, `cluster`, and `cbsgroups`. The default technique is `cbsgroups`.
- `--epochs`: Sets the number of epochs for training the model. The default number of epochs is 5.
- `--batch_size`: Determines the size of each batch of data used during training. The default batch size is 8.
- `--margin`: Specifies the margin value for the triplet loss function. The default margin is 0.5.
- `--traindf`: The path to the training dataframe stored as a `.pkl` file. The default path is `data/traindf.pkl`.
- `--testdf`: The path to the testing dataframe stored as a `.pkl` file. The default path is `data/testdf.pkl`.
- `--tabledf`: The path to the table dataframe stored as a `.pkl` file. The default path is `data/tabledf.pkl`.

### Running the Script

To train your model, execute the script from the command line by providing values for the above arguments as needed. Below is an example command to train a Query Encoder model with the `cbsgroups` hard negative mining technique:

```python
python trainencoder.py --model QueryEncoder --hnm_technique cbsgroups --epochs 10 --batch_size 16 --margin 0.5 --traindf path/to/your/traindf.pkl --testdf path/to/your/testdf.pkl --tabledf path/to/your/tabledf.pkl
```

## Training ColBERT
### Command-line Arguments

- `--model_name`: Identifier for the pre-trained model to use. Defaults to a specific BERT model but can be adjusted to any compatible model identifier.
- `--negative_n`: Number of negative samples to generate for each positive sample. Default is 5.
- `--output_name`: The name under which the trained model will be saved. Default is "ColBERT".


```python
python traincolbert.py --model_name textgain/allnli-GroNLP-bert-base-dutch-cased --negative_n 5 --output_name MyTrainedModel
```
**Training Mistral-7b LoRA Model:**

To train the Mistral-7b LoRA model, please follow these steps:

1. **Prepare Your Data:** Upload a `.pickle` file containing the instruction, input, and output data.
2. **Run the Training Notebook:** Access and run the training notebook by following this link: [Mistral-7b LoRA Training Notebook](https://colab.research.google.com/drive/1CtKHsQxoe7Fm7i7PaZ6GfrPz_Uo_fOyq?usp=sharing).
3. **Save the Model:** Once training is complete, save the trained LoRA to your Hugging Face account.

For more information on the setup and training process, please refer to the [Unsloth GitHub Repository](https://github.com/unslothai/unsloth).

## Running Mistral QA

To run the trained Contrained Dynamic decoding Mistral Model, through the following command-line arguments:


### Command-line Arguments

- `--entityretrieval`: Specifies the entity retriever function. Choose `dualencoder` for a dual encoder model or `colbert` for colbert, or `bm25` for bm25. The default is `colbert`.

```python
python qa_mistral.py --entityretrieval bm25
```

## Generating Summaries

The utils folder contains a notebook for generating the summaries and keywords based on a table description

## Evaluation

The utils folder contains a notebook for evaluating the different generated S-expressions. It generates scores for rouge-2, Bleu,Table Exact Match, Measure Exact Match, Dimension F1, Overall Exact Match, and a automated Relevancy Score

