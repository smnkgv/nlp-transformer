import torch
import argparse
import os
from model.transformer import Transformer
from model.utils import decode

parser = argparse.ArgumentParser(
    description="Generate text with a trained transformer model"
)
parser.add_argument(
    "--checkpoint", type=str, required=True, help="path to the model checkpoint"
)
parser.add_argument(
    "--prompt", type=str, default="", help="starting prompt for generation"
)
parser.add_argument(
    "--max_tokens", type=int, default=500, help="maximum number of tokens to generate"
)
parser.add_argument(
    "--temperature",
    type=float,
    default=0.8,
    help="sampling temperature (higher = more random)",
)
args = parser.parse_args()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load checkpoint
if not os.path.exists(args.checkpoint):
    raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")

# Load the saved model state and vocabulary
checkpoint = torch.load(args.checkpoint, map_location=device)
model_args = checkpoint["args"]
chars = checkpoint["vocab"]["chars"]
stoi = checkpoint["vocab"]["stoi"]
itos = checkpoint["vocab"]["itos"]
vocab_size = len(chars)
block_size = model_args["block_size"]

# Initialize model with the same architecture as during training
model = Transformer(
    vocab_size=vocab_size,
    d_model=model_args["d_model"],
    num_heads=model_args["num_heads"],
    d_ff=model_args["d_ff"],
    num_layers=model_args["num_layers"],
    max_seq_len=model_args["block_size"],
    dropout=model_args["dropout"],
).to(device)

# Load model parameters
model.load_state_dict(checkpoint["model_state_dict"])

# Set model to evaluation mode
# This disables dropout and other training-specific behaviors
model.eval()

print(f"Model loaded from checkpoint: {args.checkpoint}")

# Prepare the prompt
if args.prompt:
    # Convert prompt to tensor of token IDs
    # Each character is mapped to its corresponding vocabulary ID
    prompt_ids = [
        stoi.get(c, 0) for c in args.prompt
    ]  # Use 0 as default for unknown chars
    context = torch.tensor([prompt_ids], dtype=torch.long, device=device)
else:
    # Start with a single token (first char in the vocabulary)
    # We need at least one token to start the generation process
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    context[0, 0] = 0

# Generate text
print("\nGenerating text...")
with torch.no_grad():  # No need to track gradients during inference
    # The generate method will:
    # 1. Take the current context as input
    # 2. Predict the probability distribution for the next token
    # 3. Sample from this distribution (controlled by temperature)
    # 4. Add the sampled token to the context
    # 5. Repeat steps 1-4 until max_tokens is reached
    generated = model.generate(
        context,
        max_new_tokens=args.max_tokens,
        block_size=block_size,
        temp=args.temperature,
    )[0].tolist()

    # Convert token IDs back to characters
    generated_text = decode(generated, itos)

    # If we used a prompt, print separately from the generated text
    if args.prompt:
        print(f"Prompt: {args.prompt}")
        print(f"Generated: {generated_text[len(args.prompt):]}")
    else:
        print(f"Generated: {generated_text}")
