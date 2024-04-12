import torch
import torch.nn as nn
import torch.nn.functional as F


class TableEncoder(nn.Module):
    def __init__(self, embedding_size, transformed_size):
        super(TableEncoder, self).__init__()
        self.fc = nn.Linear(embedding_size, transformed_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        return x


class QueryEncoder(nn.Module):
    def __init__(self, embedding_size, transformed_size):
        super(QueryEncoder, self).__init__()
        self.fc = nn.Linear(embedding_size, transformed_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        return x


class DualEncoder(nn.Module):
    def __init__(self, embedding_size, transformed_size):
        super(DualEncoder, self).__init__()
        self.query_encoder = QueryEncoder(embedding_size, transformed_size)
        self.table_encoder = TableEncoder(embedding_size, transformed_size)

    def forward(self, query_emb, table_emb):
        encoded_query = self.query_encoder(query_emb)
        encoded_table = self.table_encoder(table_emb)
        return encoded_query, encoded_table


class EnhancedSiameseNetworkTripletGroNLLSTM(nn.Module):
    def __init__(self, embedding_size=768, hidden_size=512, num_layers=2, dropout_rate=0.5):
        super(EnhancedSiameseNetworkTripletGroNLLSTM, self).__init__()

        # LSTM layer
        self.lstm = nn.LSTM(input_size=embedding_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers, 
                            batch_first=True, 
                            dropout=dropout_rate)

        # Linear transformation layer
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

    def forward_once(self, x):
        # LSTM layer
        x, (hn, cn) = self.lstm(x)

        # Selecting the output of the last LSTM sequence
        x = x[:, -1, :]

        # Linear transformation with Batch Normalization and Dropout
        x = F.relu(self.bn(self.fc(x)))
        x = self.dropout(x)

        return x

    def forward(self, anchor, positive, negative):
        # Processing each input through the network independently
        output_anchor = self.forward_once(anchor)
        output_positive = self.forward_once(positive)
        output_negative = self.forward_once(negative)

        return output_anchor, output_positive, output_negative

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


class EnhancedSiameseNetworkTriplet(nn.Module):
    def __init__(self, embedding_size=1536, intermediate_size=1024, transformed_size=512, dropout_rate=0.5):
        super(EnhancedSiameseNetworkTriplet, self).__init__()
        
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


import torch.nn as nn
import torch.nn.functional as F

class EnhancedSiameseNetworkTripletGroNLPAttention(nn.Module):
    def __init__(self, embedding_size=768, intermediate_size=512, transformed_size=512, dropout_rate=0.5, num_heads=8):
        super(EnhancedSiameseNetworkTripletGroNLPAttention, self).__init__()
        
        # First linear transformation layer
        self.fc1 = nn.Linear(embedding_size, intermediate_size)
        self.bn1 = nn.BatchNorm1d(intermediate_size)

        # Attention layer
        self.attention = nn.MultiheadAttention(intermediate_size, num_heads=num_heads)

        # Second linear transformation layer
        self.fc2 = nn.Linear(intermediate_size, transformed_size)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)

    def forward_once(self, x):
        # Apply first linear transformation and activation
        x = F.relu(self.bn1(self.fc1(x)))
        
        # Apply attention
        attn_output, _ = self.attention(x, x, x)
        x = attn_output + x  # Optionally, add a residual connection
        
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
