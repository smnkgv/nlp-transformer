import torch
import numpy as np


def load_data(file_path):
    """Load and preprocess the text data"""
    with open(file_path, "r") as f:
        text = f.read()
    return text


def build_vocab(text):
    """
    Build character-level vocabulary from text

    This is the simplest form of tokenization - character level.
    More advanced transformers use subword tokenization (BPE, WordPiece, etc.)
    """
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return chars, vocab_size, stoi, itos


def encode(text, stoi):
    """
    Encode text to integer tokens

    This converts each character to its corresponding integer ID
    based on the vocabulary mapping (stoi).
    """
    return [stoi[ch] for ch in text]


def decode(tokens, itos):
    """
    Decode integers back to text

    This converts integer IDs back to their corresponding characters
    and joins them into a single string.
    """
    return "".join([itos[token] for token in tokens])


def get_batch(data, batch_size, block_size, device):
    """
    Generate a small batch of inputs and targets for training

    In language modeling, the target is the next token after each input token,
    creating a supervised learning task from unsupervised data.
    """
    # Ensure we can at least get one sample by clamping the range to at least 1
    max_range = max(1, len(data) - block_size)
    ix = torch.randint(max_range, (min(batch_size, max_range),))

    # Handle the case where data length is less than block_size
    if len(data) <= block_size:
        # Repeat the data to match block_size if needed
        # This handles small datasets by creating synthetic sequences
        padded_data = data.repeat((block_size // len(data)) + 1)

        # Create input sequences (x) and target sequences (y)
        # The target is shifted by one position (next token prediction)
        x = padded_data[:block_size].unsqueeze(0).repeat(min(batch_size, max_range), 1)
        y = (
            torch.roll(padded_data[:block_size], -1)
            .unsqueeze(0)
            .repeat(min(batch_size, max_range), 1)
        )
    else:
        # Normal case: extract subsequences of length block_size from data
        # Each x[i] is a context window of block_size tokens
        x = torch.stack([data[i : i + block_size] for i in ix])

        # Each y[i] is the corresponding target: the next token after each position in x[i]
        # This creates the classic language modeling task: predict the next token
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])

    # Move tensors to the specified device (CPU/GPU)
    x, y = x.to(device), y.to(device)
    return x, y
