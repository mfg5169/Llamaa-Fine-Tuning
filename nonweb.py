import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0), :]
    
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
         super(ScaledDotProductAttention, self).__init__()
         self.dropout = nn.Dropout(p=dropout)
         self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value, mask=None):
        # Calculate QK^T
        scores = torch.matmul(query, key.transpose(-2, -1)) /math.sqrt(query.size(-1))
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = self.softmax(scores)
        attn = self.dropout(attn)
        
        # Apply attention to value vectors
        output = torch.matmul(attn, value)
        return output, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        
        self.attn = ScaledDotProductAttention(dropout)
        self.fc_out = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear projections
        query = self.query_linear(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = self.key_linear(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.value_linear(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention
        output, attn = self.attn(query, key, value, mask)
        
        # Concat heads and pass through final linear layer
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        output = self.fc_out(output)
        
        return output, attn

class PositionwiseFeedforward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedforward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.layernorm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        return self.layernorm(out + x)  # residual connection

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feedforward = PositionwiseFeedforward(d_model, d_ff, dropout)
        
    def forward(self, src, mask=None):
        # Self-attention with residual connection and layer norm
        attention_output, _ = self.self_attention(src, src, src, mask)
        out1 = self.feedforward(attention_output)
        return out1

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.encoder_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feedforward = PositionwiseFeedforward(d_model, d_ff, dropout)
        
    def forward(self, tgt, memory, src_mask=None, tgt_mask=None):
        # Self-attention
        self_attn_output, _ = self.self_attention(tgt, tgt, tgt, tgt_mask)
        
        # Encoder-decoder attention
        attention_output, _ = self.encoder_attention(self_attn_output, memory, memory, src_mask)
        
        # Feed-forward with residual connection
        out = self.feedforward(attention_output)
        return out

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_ff, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.layernorm = nn.LayerNorm(d_model)
        
    def forward(self, src, mask=None):
        out = src
        for layer in self.layers:
            out = layer(out, mask)
        return self.layernorm(out)

class TransformerDecoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_ff, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.layernorm = nn.LayerNorm(d_model)
        
    def forward(self, tgt, memory, src_mask=None, tgt_mask=None):
        out = tgt
        for layer in self.layers:
            out = layer(out, memory, src_mask, tgt_mask)
        return self.layernorm(out)

class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, num_heads, num_layers, d_ff, dropout=0.1):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.encoder = TransformerEncoder(d_model, num_heads, num_layers, d_ff, dropout)
        self.decoder = TransformerDecoder(d_model, num_heads, num_layers, d_ff, dropout)
        self.fc_out = nn.Linear(d_model, output_dim)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src = self.positional_encoding(self.embedding(src))
        tgt = self.positional_encoding(self.embedding(tgt))
        
        memory = self.encoder(src, src_mask)
        output = self.decoder(tgt, memory, src_mask, tgt_mask)
        return self.fc_out(output)
    
model = Transformer(input_dim=10000, output_dim=10000, d_model=512, num_heads=8, num_layers=6, d_ff=2048)

# Example input (batch_size, seq_len)
src = torch.randint(0, 10000, (32, 20))  # Batch of 32, sequence length 20
tgt = torch.randint(0, 10000,(32, 20))
