# Simple NLP Transformer from Scratch

This project implements a simple transformer model from scratch for language modeling. It's designed for educational purposes to understand how transformers work.

## Project Structure

```
.
├── data/                  # Directory for data
│   └── tiny_shakespeare.txt  # Small Shakespeare text dataset
├── model/                 # Model implementation
│   ├── transformer.py     # Core transformer implementation
│   └── utils.py           # Utility functions
├── train.py               # Training script
├── generate.py            # Text generation script
├── requirements.txt       # Dependencies
└── README.md              # This file
```

## Key Components

The transformer implementation includes:

1. **Multi-Head Attention**: The core mechanism that allows the model to focus on different parts of the input sequence.
2. **Positional Encoding**: Since transformers don't have recurrence, positional encoding adds information about token positions.
3. **Feed-Forward Networks**: Applied to each position separately.
4. **Layer Normalization & Residual Connections**: For stable training.
5. **Autoregressive Generation**: For generating text one token at a time.

## How to Use

### Installation

```bash
pip install -r requirements.txt
```

### Training

Train the model with default parameters:

```bash
python train.py
```

Or customize with arguments:

```bash
python train.py --batch_size 64 --d_model 256 --num_heads 8 --num_layers 6 --max_iters 10000
```

### Text Generation

Generate text using a trained model:

```bash
python generate.py --checkpoint model_checkpoints/model_iter_1000.pt --prompt "First Citizen:" --max_tokens 200
```

## Key Parameters

- `d_model`: Embedding dimension
- `num_heads`: Number of attention heads
- `d_ff`: Feed-forward network dimension
- `num_layers`: Number of transformer layers
- `block_size`: Maximum context length for predictions
- `batch_size`: Number of sequences per batch
- `dropout`: Dropout rate for regularization

## How the Transformer Works

1. **Input Processing**: Text is tokenized at character level, converted to embeddings, and positional encoding is added.
2. **Self-Attention**: Each token attends to all previous tokens (causal attention).
3. **Layer Stack**: Multiple transformer blocks are stacked for deeper representations.
4. **Output Layer**: Predicts the probability distribution for the next token.
5. **Generation**: Autoregressively generates text by sampling from the predicted distributions.

## Learning Resources

To learn more about transformers:

- ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (original transformer paper)
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
# nlp-transformer
