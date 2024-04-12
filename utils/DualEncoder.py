import torch.nn as nn



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
    