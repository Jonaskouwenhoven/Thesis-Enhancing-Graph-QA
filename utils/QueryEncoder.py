import torch.nn as nn

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

class QueryEncoder(nn.Module):
    def __init__(self, embedding_size, transformed_size, dropout_rate=0.5):
        super(QueryEncoder, self).__init__()
        self.query_encoder = CustomQueryEncoder(embedding_size, transformed_size, dropout_rate)
        self.table_encoder = PassThroughTableEncoder(transformed_size)

    def forward(self, query_emb, table_emb):
        encoded_query = self.query_encoder(query_emb)
        encoded_table = self.table_encoder(table_emb)
        return encoded_query, encoded_table



