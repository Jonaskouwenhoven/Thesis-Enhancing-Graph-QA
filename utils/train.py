import torch
import torch.optim as optim
import utils.utils as util
import torch
import torch.nn as nn
from tqdm import tqdm 
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, embedding_size, transformed_size, dropout_rate=0.5):
        super(Encoder, self).__init__()
        self.fc = nn.Linear(embedding_size, transformed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class SiameseNetwork(nn.Module):
    def __init__(self, embedding_size, transformed_size, dropout_rate=0.5):
        super(SiameseNetwork, self).__init__()
        self.encoder = Encoder(embedding_size, transformed_size, dropout_rate)

    def forward(self, input1, input2):
        encoded_input1 = self.encoder(input1)
        encoded_input2 = self.encoder(input2)
        return encoded_input1, encoded_input2


class EnhancedSiameseNetworkTripletGroNLP(nn.Module):
    def __init__(self, embedding_size=768, intermediate_size=512, transformed_size=512, dropout_rate=0.5):
        super(EnhancedSiameseNetworkTripletGroNLP, self).__init__()
        
        # First linear transformation layer
        self.fc1 = nn.Linear(embedding_size, intermediate_size)
        self.bn1 = nn.BatchNorm1d(intermediate_size)

        # Second linear transformation layer
        self.fc2 = nn.Linear(intermediate_size, transformed_size)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

    def forward_once(self, x):
        # Apply first linear transformation and activation
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)

        # Apply second linear transformation
        x = self.fc2(x)

        return x

    def forward(self, anchor, positive, negative):
        # Processing each input through the network independently
        output_anchor = self.forward_once(anchor)
        output_positive = self.forward_once(positive)
        output_negative = self.forward_once(negative)

        return output_anchor, output_positive, output_negative


    def forward(self, anchor, positive, negative):
        # Processing each input through the network independently
        output_anchor = self.forward_once(anchor)
        output_positive = self.forward_once(positive)
        output_negative = self.forward_once(negative)

        return output_anchor, output_positive, output_negative

class PassThroughTableEncoder(nn.Module):
    def __init__(self, embedding_size):
        super(PassThroughTableEncoder, self).__init__()
        self.embedding_size = embedding_size

    def forward(self, x):
        if x.shape[-1] != self.embedding_size:
            raise ValueError("Input shape does not match expected embedding size.")
        return x

class CustomQueryEncoder(nn.Module):
    def __init__(self, embedding_size, transformed_size, dropout_rate=0.5):
        super(CustomQueryEncoder, self).__init__()
        self.fc = nn.Linear(embedding_size, transformed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class ModifiedDualEncoder(nn.Module):
    def __init__(self, embedding_size, transformed_size, dropout_rate=0.5):
        super(ModifiedDualEncoder, self).__init__()
        self.query_encoder = CustomQueryEncoder(embedding_size, transformed_size, dropout_rate)
        self.table_encoder = PassThroughTableEncoder(transformed_size)

    def forward(self, query_emb, table_emb):
        encoded_query = self.query_encoder(query_emb)
        encoded_table = self.table_encoder(table_emb)
        return encoded_query, encoded_table



# import torch.nn as nn

# class CustomQueryEncoder(nn.Module):
#     def __init__(self, embedding_size, transformed_size, bottleneck_size, dropout_rate=0.5):
#         super(CustomQueryEncoder, self).__init__()
#         # First layer: original embedding size to transformed size
#         self.fc1 = nn.Linear(embedding_size, transformed_size)
#         # Bottleneck layer: transformed size to bottleneck size
#         self.bottleneck = nn.Linear(transformed_size, bottleneck_size)
#         # Expansion layer: bottleneck size back to transformed size
#         self.fc2 = nn.Linear(bottleneck_size, transformed_size)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout_rate)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.bottleneck(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         return x
# class ModifiedDualEncoder(nn.Module):
#     def __init__(self, embedding_size, transformed_size, bottleneck_size, dropout_rate=0.5):
#         super(ModifiedDualEncoder, self).__init__()
#         self.query_encoder = CustomQueryEncoder(embedding_size, transformed_size, bottleneck_size, dropout_rate)
#         self.table_encoder = PassThroughTableEncoder(transformed_size)

#     def forward(self, query_emb, table_emb):
#         encoded_query = self.query_encoder(query_emb)
#         encoded_table = self.table_encoder(table_emb)
#         return encoded_query, encoded_table


class TableEncoder(nn.Module):
    def __init__(self, embedding_size, transformed_size, dropout_rate=0.5):
        super(TableEncoder, self).__init__()
        self.fc = nn.Linear(embedding_size, transformed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class QueryEncoder(nn.Module):
    def __init__(self, embedding_size, transformed_size, dropout_rate=0.5):
        super(QueryEncoder, self).__init__()
        self.fc = nn.Linear(embedding_size, transformed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class DualEncoder(nn.Module):
    def __init__(self, embedding_size, transformed_size, dropout_rate=0.5):
        super(DualEncoder, self).__init__()
        self.query_encoder = QueryEncoder(embedding_size, transformed_size, dropout_rate)
        self.table_encoder = TableEncoder(embedding_size, transformed_size, dropout_rate)

    def forward(self, query_emb, table_emb):
        encoded_query = self.query_encoder(query_emb)
        encoded_table = self.table_encoder(table_emb)
        return encoded_query, encoded_table
    

# import torch
# import torch.nn as nn

# class TransformerEncoder(nn.Module):
#     def __init__(self, embedding_size, num_heads, num_layers, dropout_rate=0.5):
#         super(TransformerEncoder, self).__init__()
#         self.embedding_size = embedding_size
#         self.num_heads = num_heads
#         self.num_layers = num_layers
#         self.dropout_rate = dropout_rate

#         self.pos_encoder = nn.Sequential(
#             nn.Linear(embedding_size, 768),
#             nn.ReLU(),
#             nn.Dropout(dropout_rate)
#         )

#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=768,
#             nhead=num_heads,
#             dim_feedforward=2048,
#             dropout=dropout_rate
#         )
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

#     def forward(self, x):
#         x = self.pos_encoder(x)
#         x = x.permute(1, 0, 2)  # Reshape to (seq_len, batch_size, embedding_size)
#         x = self.transformer_encoder(x)
#         x = x.permute(1, 0, 2)  # Reshape back to (batch_size, seq_len, embedding_size)
#         return x[:, -1, :]  # Return the last token's embedding

# class DualEncoderTransformer(nn.Module):
#     def __init__(self, embedding_size, num_heads, num_layers, dropout_rate=0.5):
#         super(DualEncoderTransformer, self).__init__()
#         self.query_encoder = TransformerEncoder(embedding_size, num_heads, num_layers, dropout_rate)
#         self.table_encoder = TransformerEncoder(embedding_size, num_heads, num_layers, dropout_rate)

#     def forward(self, query_emb, table_emb):
#         encoded_query = self.query_encoder(query_emb)
#         encoded_table = self.table_encoder(table_emb)
#         return encoded_query, encoded_table

class TransformerEncoder(nn.Module):
    def __init__(self, embedding_size, num_heads, num_layers, dropout_rate=0.5):
        super(TransformerEncoder, self).__init__()
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.pos_encoder = nn.Sequential(
            nn.Linear(embedding_size, 768),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=768,
            nhead=num_heads,
            dim_feedforward=2048,
            dropout=dropout_rate
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # Adjust input x to have a sequence length dimension if it's not present
        if x.dim() == 2:  # if shape is [batch_size, embedding_size]
            x = x.unsqueeze(1)  # reshape to [batch_size, 1, embedding_size]
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)  # Reshape to [seq_len, batch_size, embedding_size]
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # Reshape back to [batch_size, seq_len, embedding_size]
        return x[:, -1, :]  # Return the last token's embedding

class DualEncoderTransformer(nn.Module):
    def __init__(self, embedding_size, num_heads, num_layers, dropout_rate=0.5):
        super(DualEncoderTransformer, self).__init__()
        self.query_encoder = TransformerEncoder(embedding_size, num_heads, num_layers, dropout_rate)
        self.table_encoder = TransformerEncoder(embedding_size, num_heads, num_layers, dropout_rate)

    def forward(self, query_emb, table_emb):
        encoded_query = self.query_encoder(query_emb)
        encoded_table = self.table_encoder(table_emb)
        return encoded_query, encoded_table


def train_triplet_model(model, train_dataloader, num_epochs=10, early_stopping_limit=10, learning_rate=5e-6, margin=10):
    """
    Train a Siamese triplet model using triplet loss.

    Args:
    - model: The Siamese model to be trained.
    - train_dataloader (DataLoader): DataLoader for training triplets.
    - num_epochs (int): Number of training epochs (default is 10).
    - early_stopping_limit (int): Number of epochs for early stopping (default is 10).
    - learning_rate (float): Learning rate for optimizer (default is 5e-6).
    - margin (float): Margin for triplet loss (default is 10).

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

def trained_siamese(model, train_dataloader, num_epochs=5, learning_rate=1e-5, margin=1.0):
    """
    Train a Siamese triplet model using triplet loss.

    Args:
        model (nn.Module): The Siamese model to be trained.
        train_dataloader (DataLoader): DataLoader for training triplets.
        val_dataloader (DataLoader): DataLoader for validation triplets.
        num_epochs (int): Number of training epochs (default is 250).
        learning_rate (float): Learning rate for optimizer (default is 1e-5).
        margin (float): Margin for triplet loss (default is 1.0).
        early_stopping_limit (int): Number of epochs for early stopping (default is 25).

    Returns:
        nn.Module: Trained Siamese triplet model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = util.TripletLoss(margin=margin).to(device)



    print("Training Started".center(60, "#"))

    for epoch in tqdm(range(num_epochs), total=num_epochs, desc='Epochs'):
        model.train()
        total_train_loss = 0

        for batch in train_dataloader:
            anchor_emb, positive_emb, negative_emb = batch['anchor'].to(device), batch['positive'].to(device), batch['negative'].to(device)

            optimizer.zero_grad()

            # Model outputs embeddings for anchor, positive, and negative inputs
            output_anchor, output_positive, output_negative = model(anchor_emb, positive_emb, negative_emb)

            # Calculate loss using the triplet loss function
            loss = criterion(output_anchor, output_positive, output_negative)
            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f'Epoch: {epoch + 1}, Training Loss: {avg_train_loss:.4f}')


    return model
