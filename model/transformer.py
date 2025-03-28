import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        # Split embedding dimension into heads, each head gets d_model/num_heads dimensions
        self.head_dim = d_model // num_heads
        assert (
            self.head_dim * num_heads == d_model
        ), "d_model must be divisible by num_heads"

        # Key, query, value projections
        # These linear transformations project the input embedding into different spaces
        # for keys, queries, and values - the core components of attention
        self.key = nn.Linear(d_model, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        # Output projection
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size = x.size(0)

        # Linear projections and reshape for multi-head attention
        # 1. Project input (x) into key, query, value spaces
        # 2. Reshape to [batch_size, num_heads, seq_len, head_dim]
        # 3. Transpose to prepare for batch matrix multiplication
        k = (
            self.key(x)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )  # (B, nh, T, hd)
        q = (
            self.query(x)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )  # (B, nh, T, hd)
        v = (
            self.value(x)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )  # (B, nh, T, hd)

        # Compute attention scores: Q * K^T / sqrt(d_k)
        # This gives a similarity measure between each query and all keys
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(
            self.head_dim
        )  # (B, nh, T, T)

        # Apply mask if provided (for causal/autoregressive attention)
        # In causal attention, tokens can only attend to previous tokens
        if mask is not None:
            # Mask future positions with a large negative value (effectively zero after softmax)
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax to get attention weights (probabilities summing to 1)
        # Each query will have a probability distribution over all keys
        attn_weights = F.softmax(scores, dim=-1)  # (B, nh, T, T)
        attn_weights = self.dropout(attn_weights)

        # Compute weighted sum of values based on attention weights
        # This is where the information aggregation happens
        out = torch.matmul(attn_weights, v)  # (B, nh, T, hd)

        # Reshape and project back to original dimension
        # 1. Transpose heads and sequence length dimensions
        # 2. Reshape back to [batch_size, seq_len, d_model]
        # 3. Apply final projection
        out = (
            out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        )  # (B, T, d_model)
        out = self.proj(out)

        return out


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        # Two-layer MLP with GELU activation
        # The hidden layer is typically 4x larger than embedding dimension
        # This adds non-linearity and increases the model's capacity to learn complex patterns
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # GELU activation is smoother than ReLU and performs better in transformers
        x = F.gelu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)

        # Layer normalization is applied before self-attention and feed-forward
        # This is the "Pre-LN" pattern that improves training stability
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual connection and layer norm
        # Residual connections (x + ...) help with the vanishing gradient problem
        # and allow the model to learn incremental improvements
        attn_output = self.attn(self.ln1(x), mask)
        x = x + self.dropout(attn_output)  # Residual connection

        # Feed forward with residual connection and layer norm
        ff_output = self.ff(self.ln2(x))
        x = x + self.dropout(ff_output)  # Residual connection

        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Create positional encoding matrix using sine and cosine functions
        # This is a key innovation in transformers since they have no inherent
        # sense of position/order of tokens (unlike RNNs)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)  # (max_len, 1)

        # Compute division term using exponential function for numerical stability
        # This creates different frequencies for different dimensions
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )

        # Apply sine to even indices and cosine to odd indices
        # This creates a unique encoding for each position and dimension
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices

        # Add batch dimension for broadcasting
        pe = pe.unsqueeze(0)

        # Register the positional encoding as a buffer (not a parameter)
        # This means it won't be updated during training
        self.register_buffer("pe", pe)

    def forward(self, x):
        # Add the positional encoding to the input embeddings
        # Only uses the first seq_len positions of the pre-computed table
        return x + self.pe[:, : x.size(1)]


class Transformer(nn.Module):
    def __init__(
        self, vocab_size, d_model, num_heads, d_ff, num_layers, max_seq_len, dropout=0.1
    ):
        super().__init__()
        # Token embedding layer converts token IDs to vectors
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

        # Stack multiple transformer blocks
        # Each block contains self-attention and feed-forward networks
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ]
        )

        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)  # Maps to vocabulary distribution
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Create causal mask to ensure autoregressive property
        # This ensures the model can only attend to previous tokens, not future ones
        # Critical for language modeling and generation tasks
        seq_len = x.size(1)
        # Lower triangular matrix (1s in the lower triangle, 0s elsewhere)
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
        mask = mask.to(x.device)

        # Token embeddings and positional encoding
        # 1. Convert token IDs to embeddings
        # 2. Add positional information
        # 3. Apply dropout for regularization
        x = self.token_embedding(x)  # (B, T, d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # Apply transformer blocks sequentially
        for block in self.transformer_blocks:
            x = block(x, mask)

        # Apply final layer norm and project to vocabulary
        x = self.ln_f(x)
        x = self.fc_out(x)  # (B, T, vocab_size)

        return x

    def generate(self, idx, max_new_tokens, block_size=None, temp=1.0):
        """Generate new tokens autoregressively"""
        # If block_size not provided, use the model's max_seq_len
        if block_size is None:
            block_size = self.pos_encoding.pe.shape[1]

        # idx is (B, T) tensor of indices in the current context
        for _ in range(max_new_tokens):
            # Crop context to block_size if needed to avoid exceeding the positional encoding
            # This implements a sliding window if the generated sequence gets too long
            idx_cond = idx[:, -block_size:] if idx.size(1) > block_size else idx

            # Get the predictions from the model
            logits = self(idx_cond)

            # Focus only on the last time step - we only need to predict the next token
            logits = logits[:, -1, :] / temp  # (B, vocab_size)

            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample from the distribution
            # This introduces randomness controlled by temperature
            # Higher temp = more randomness, lower temp = more deterministic
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx
