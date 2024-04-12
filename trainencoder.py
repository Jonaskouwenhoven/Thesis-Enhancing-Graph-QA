import argparse
import os
import torch
from torch import optim
# import utils  # Assuming utils has necessary functions
# import train  # Assuming train has the model definitions
import utils.hardnegativemining as hnm
from utils.DualEncoder import DualEncoder
from utils.QueryEncoder import QueryEncoder
import utils.utils as util
from tqdm import tqdm
import pandas as pd

import json


def parse_args():
    parser = argparse.ArgumentParser(description="Train a dual encoder model with triplet loss.")
    parser.add_argument("--model", type=str, default="QueryEncoder", choices=["DualEncoder", "QueryEncoder"],
                        help="The model type to use.")
    parser.add_argument("--hnm_technique", type=str, required=False, default="cbsgroups",
                        help="The hard negative mining technique to use.", choices=["topn", "bm25", 'random', 'cluster', 'cbsgroups'])
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Training batch size.")
    parser.add_argument("--margin", type=float, default=0.5,
                        help="Margin for triplet loss.")
    parser.add_argument("--traindf", type=str, default='data/traindf.pkl',
                        help="path to traindataframe")
    parser.add_argument("--testdf", type=str, default='data/testdf.pkl',
                        help="path to test dataframe")
    parser.add_argument("--tabledf", type=str, default='data/tabledf.pkl',
                        help="path to table dataframe")
    return parser.parse_args()


def train_triplet_model(model, train_dataloader, num_epochs=10, early_stopping_limit=10, learning_rate=5e-6, margin=0.5):
    """
    Train a Siamese triplet model using triplet loss.

    Args:
    - model: The Siamese model to be trained.
    - train_dataloader (DataLoader): DataLoader for training triplets.
    - num_epochs (int): Number of training epochs (default is 10).
    - early_stopping_limit (int): Number of epochs for early stopping (default is 10).
    - learning_rate (float): Learning rate for optimizer (default is 5e-6).
    - margin (float): Margin for triplet loss (default is 0.5).

    Returns:
    - model: Trained Siamese model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     
    print(" Training Started ".center(60, "#"))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = util.TripletLoss(margin=margin).to(device)

    best_val_loss = float('inf')
    early_stopping_counter = 0

    for epoch in tqdm(range(num_epochs), total = num_epochs):
    # for epoch in range(num_epochs):
        
        model.train()
        total_train_loss = 0

        for batch in train_dataloader:
            anchor_emb, positive_emb, negative_emb = batch['anchor'].to(device), batch['positive'].to(device), batch['negative'].to(device)

            optimizer.zero_grad()

            # Process through the model
            output_query = model.query_encoder(anchor_emb.to(torch.float32))
            output_positive = model.table_encoder(positive_emb.to(torch.float32))
            output_negative = model.table_encoder(negative_emb.to(torch.float32))

            # Calculate triplet loss
            loss = criterion(output_query, output_positive, output_negative)
            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        # print(f'Epoch: {epoch + 1}, Training Loss: {avg_train_loss:.4f}')

        # Early stopping
        if avg_train_loss < best_val_loss:
            best_val_loss = avg_train_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= early_stopping_limit:
            print("Early stopping triggered.")
            break

    return model

def main():
    args = parse_args()

    # Read dataframes
    traindf = pd.read_pickle(args.traindf)
    table_df = pd.read_pickle(args.tabledf)


    table_df = util.add_tensor_embedding_column(table_df, "embeddings")
    traindf = util.add_tensor_embedding_column(traindf, "embedding")

    # Setup experiment directory
    experiment_name = args.hnm_technique
    directory_name = f"experiments/{experiment_name}"

    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
        print(f"Directory '{directory_name}' was created.")
    else:
        print(f"Directory '{directory_name}' already exists.")

    if args.hnm_technique == 'topn':
        df_triplets = hnm.hardnegativemining_topn_bottomn(traindf, table_df, n=5)

    elif args.hnm_technique == 'bm25':
        table_df = table_df.rename(columns={"Summary":"desc"})

        df_triplets = hnm.hardnegativemining_bm25(traindf, table_df, n=5)

    elif args.hnm_technique == 'random':
        df_triplets = hnm.hardnegativemining_random(traindf, table_df, n=5)

    elif args.hnm_technique == 'cluster':
        df_triplets = hnm.hardnegativemining_clusters(traindf, table_df, n_clusters=5, n_negatives=2)

    elif args.hnm_technique == 'cbsgroups':
        with open('./data/table_themes.json', 'r') as file:
            table_themes = json.loads(file.read())
            table_df['Groepnaam'] = table_df['table_id'].map(table_themes)

        df_triplets = hnm.hardnegativemining_cbs_groups(traindf, table_df, n=5)

    
    # Assuming the existence of functions to prepare your dataloaders and models
    _, train_dataloader = util.create_dataloader(df_triplets, batch_size=args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Model selection
    if args.model == "DualEncoder":
        model = DualEncoder(768, 768)  # Adjust to how you instantiate your normal dual encoder
    elif args.model == "QueryEncoder":
        model = QueryEncoder(768, 768)  # Adjust to how you instantiate your modified dual encoder
    model.to(device)
    
    # Train the model
    trained_model = train_triplet_model(model, train_dataloader, num_epochs=args.epochs, margin=args.margin)
    
    # Save trained model, losses, and calculate accuracy
    torch.save(trained_model.state_dict(), f"{directory_name}/Model.pt")
    # Additional code to save losses and calculate accuracy as per your original script
    print(f"Training completed.\nModel saved to '{directory_name}/Model.pt'.\n")


if __name__ == "__main__":
    main()
