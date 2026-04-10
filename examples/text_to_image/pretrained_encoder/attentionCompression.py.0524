import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionCompression(nn.Module):
    def __init__(self, input_dim, output_dim, num_tokens, num_compressed_tokens):
        super(AttentionCompression, self).__init__()
        self.input_dim = input_dim
        self.num_tokens = num_tokens
        self.num_compressed_tokens = num_compressed_tokens

        # Learnable query tokens
        self.query_tokens = nn.Parameter(torch.randn(num_compressed_tokens, input_dim))

        # Linear projections for keys and values
        self.key_proj = nn.Linear(input_dim, input_dim)
        self.value_proj = nn.Linear(input_dim, input_dim)
        self.dim_compress = nn.Linear(input_dim, output_dim) 

    def forward(self, x):
        # x shape: (batch_size, num_tokens, input_dim)
        #print('<><><> x.shape in attentionCompression.py: ', x.shape) #torch.Size([1, 4000, 1024])
        batch_size = x.shape[0]

        # Project input to keys and values
        keys = self.key_proj(x)  # (batch_size, num_tokens, input_dim)
        values = self.value_proj(x)  # (batch_size, num_tokens, input_dim)

        # Repeat query tokens for batch size
        queries = self.query_tokens.unsqueeze(0).repeat(batch_size, 1, 1)  # (batch_size, num_compressed_tokens, input_dim)

        # Compute attention scores
        attn_scores = torch.bmm(queries, keys.transpose(1, 2)) / (self.input_dim ** 0.5)  # (batch_size, num_compressed_tokens, num_tokens)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch_size, num_compressed_tokens, num_tokens)

        # Apply attention to values
        compressed_output = torch.bmm(attn_weights, values)  # (batch_size, num_compressed_tokens, input_dim)
        compressed_output = self.dim_compress(compressed_output)
        #print('<><><> compressed_output shape: ', compressed_output.shape) # torch.Size([1, 51, 768]) # changed to 51 X 768 05/18/2025
        return compressed_output



