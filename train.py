import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse
import time
import os
from model.transformer import Transformer
from model.utils import load_data, build_vocab, encode, decode, get_batch

# Command-line arguments
parser = argparse.ArgumentParser(description="Train a simple transformer model")
parser.add_argument(
    "--data",
    type=str,
    default="data/tiny_shakespeare.txt",
    help="path to the data file",
)
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument(
    "--block_size", type=int, default=64, help="context size for predictions"
)
parser.add_argument("--d_model", type=int, default=128, help="embedding dimension")
parser.add_argument(
    "--num_heads", type=int, default=4, help="number of attention heads"
)
parser.add_argument("--d_ff", type=int, default=512, help="feed-forward dimension")
parser.add_argument(
    "--num_layers", type=int, default=4, help="number of transformer layers"
)
parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
parser.add_argument(
    "--max_iters", type=int, default=5000, help="maximum training iterations"
)
parser.add_argument(
    "--eval_interval", type=int, default=100, help="interval to evaluate the model"
)
parser.add_argument(
    "--save_interval", type=int, default=1000, help="interval to save the model"
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="model_checkpoints",
    help="directory to save model",
)
args = parser.parse_args()

# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# Device configuration - use GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
text = load_data(args.data)
print(f"Data loaded: {len(text)} characters")

# Build vocabulary
# This creates a mapping between characters and unique integer IDs
chars, vocab_size, stoi, itos = build_vocab(text)
print(f"Vocabulary size: {vocab_size}")

# Encode data - convert text to integer tokens
data = encode(text, stoi)
data = torch.tensor(data, dtype=torch.long)

# Split data into train/val sets (90/10 split)
# This allows us to monitor for overfitting during training
n = int(0.9 * len(data))
train_data = data[:n].to(device)
val_data = data[n:].to(device)
print(f"Train data size: {len(train_data)}, Val data size: {len(val_data)}")

# Initialize model
# This creates a transformer with the specified hyperparameters
model = Transformer(
    vocab_size=vocab_size,  # Size of the vocabulary (number of unique tokens)
    d_model=args.d_model,  # Embedding dimension - larger means more capacity but more params
    num_heads=args.num_heads,  # Number of attention heads - allows model to focus on different parts
    d_ff=args.d_ff,  # Feed-forward network dimension - typically 4x the embedding dim
    num_layers=args.num_layers,  # Number of transformer blocks - deeper = more capacity
    max_seq_len=args.block_size,  # Maximum sequence length for positional encoding
    dropout=args.dropout,  # Dropout rate for regularization
).to(device)

# Optimizer - AdamW is the standard for transformers
# It's an improved version of Adam with better weight decay handling
optimizer = optim.AdamW(model.parameters(), lr=args.lr)

# Count parameters - This gives an idea of model size
param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of parameters: {param_count:,}")

# Define block_size for the model's generate method
block_size = args.block_size


# Training loop
def train():
    best_val_loss = float("inf")
    for iter in range(args.max_iters):
        t0 = time.time()

        # Training step
        model.train()
        optimizer.zero_grad()  # Zero gradients before each step

        # Get a batch of data
        # x: input sequences, y: target sequences (shifted by 1)
        x, y = get_batch(train_data, args.batch_size, args.block_size, device)
        logits = model(x)  # Forward pass through the model

        # Compute loss (averaged over all dimensions except batch)
        # CrossEntropyLoss combines log softmax and NLL loss
        # We reshape to (batch_size*seq_len, vocab_size) and (batch_size*seq_len)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()  # Backward pass - compute gradients

        # Clip gradients to prevent exploding gradients
        # This is a common practice in training transformers
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()  # Update weights based on gradients

        # Evaluation
        if iter % args.eval_interval == 0:
            model.eval()  # Set model to evaluation mode (disables dropout)
            with torch.no_grad():  # No need to track gradients during evaluation
                # Get validation batch
                val_x, val_y = get_batch(
                    val_data, args.batch_size, args.block_size, device
                )
                val_logits = model(val_x)
                val_loss = F.cross_entropy(
                    val_logits.view(-1, vocab_size), val_y.view(-1)
                )

                # Log training and validation loss
                print(
                    f"Iter {iter}/{args.max_iters} | Train loss: {loss.item():.4f} | "
                    f"Val loss: {val_loss.item():.4f} | Time: {time.time() - t0:.2f}s"
                )

                # Track best validation loss
                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()

                # Generate sample text to check progress
                # Start with a single token and generate 100 more
                context = torch.zeros((1, 1), dtype=torch.long, device=device)
                context[0, 0] = data[0]  # Start with the first character of the dataset
                generated = model.generate(
                    context, max_new_tokens=100, block_size=args.block_size, temp=0.8
                )[0].tolist()
                generated_text = decode(generated, itos)
                print(f"\nSample generated text:\n{generated_text}\n")

        # Save model checkpoint
        if iter % args.save_interval == 0 and iter > 0:
            checkpoint = {
                "model_state_dict": model.state_dict(),  # Model weights
                "optimizer_state_dict": optimizer.state_dict(),  # Optimizer state
                "iter": iter,  # Current iteration
                "vocab": {  # Vocabulary info for generation
                    "chars": chars,
                    "stoi": stoi,
                    "itos": itos,
                },
                "args": vars(args),  # Training arguments
            }
            checkpoint_path = os.path.join(args.output_dir, f"model_iter_{iter}.pt")
            torch.save(checkpoint, checkpoint_path)
            print(f"Model saved to {checkpoint_path}")


if __name__ == "__main__":
    train()
