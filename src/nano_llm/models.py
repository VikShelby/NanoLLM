import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention module."""
    # Added seq_length to the constructor
    def __init__(self, d_model, n_heads, dropout, seq_length):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads # Dimension of keys, queries, values per head
        self.n_heads = n_heads
        self.d_model = d_model
        self.seq_length = seq_length # Store maximum sequence length

        self.wq = nn.Linear(d_model, d_model) # Query weights
        self.wk = nn.Linear(d_model, d_model) # Key weights
        self.wv = nn.Linear(d_model, d_model) # Value weights
        self.wo = nn.Linear(d_model, d_model) # Output weights

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Mask to prevent attention to future tokens (for causal language modeling)
        # Use the passed seq_length for the mask size
        # Create the lower triangular mask (True for allowed connections)
        # Ensure it's 3D: (1, seq_length, seq_length)
        causal_mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool().logical_not()
        # Register it as a buffer, ensuring it has the (1, seq_length, seq_length) shape
        self.register_buffer('mask', causal_mask.view(1, seq_length, seq_length))


    def forward(self, x):
        # x shape: (batch_size, current_seq_length, d_model)
        batch_size, current_seq_length, d_model = x.size()

        # Ensure the input sequence length doesn't exceed the model's max seq_length
        # This check is useful, though typically handled by cropping the input before calling forward
        # assert current_seq_length <= self.seq_length, f"Input sequence length {current_seq_length} exceeds model's max seq_length {self.seq_length}"
        # The forward method handles current_seq_length <= self.seq_length gracefully due to slicing the mask.

        # 1) Linear projections (wq, wk, wv) and split into heads
        # Shape after view and transpose: (batch_size, n_heads, current_seq_length, d_k)
        q = self.wq(x).view(batch_size, current_seq_length, self.n_heads, self.d_k).transpose(1, 2)
        k = self.wk(x).view(batch_size, current_seq_length, self.n_heads, self.d_k).transpose(1, 2)
        v = self.wv(x).view(batch_size, current_seq_length, self.n_heads, self.d_k).transpose(1, 2)


        # 2) Scaled Dot-Product Attention
        # (q @ k.transpose(-2, -1)) / sqrt(d_k)
        # Result shape: (batch_size, n_heads, current_seq_length, current_seq_length)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_k))

        # Apply causal mask
        # Slice the registered mask (1, max_seq_length, max_seq_length)
        # to match the current sequence length (current_seq_length)
        # The slicing `[:, :current_seq_length, :current_seq_length]` is now valid
        # because self.mask is 3D (1, ..., ...)
        mask = self.mask[:, :current_seq_length, :current_seq_length] # Already boolean due to init
        # Apply the mask: fill with -inf where mask is False (upper triangle)
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        # 3) Multiply by values and concatenate heads
        # (batch_size, n_heads, current_seq_length, current_seq_length) @ (batch_size, n_heads, current_seq_length, d_k)
        # -> (batch_size, n_heads, current_seq_length, d_k)
        attention = torch.matmul(attn_probs, v)
        # Transpose and reshape back to (batch_size, current_seq_length, d_model)
        attention = attention.transpose(1, 2).contiguous().view(batch_size, current_seq_length, d_model)

        # 4) Final linear layer
        output = self.wo(attention)
        output = self.resid_dropout(output)
        return output


class FeedForward(nn.Module):
    """Simple two-layer feed-forward network."""
    def __init__(self, d_model, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), # Inner layer typically 4x d_model
            nn.GELU(), # Or ReLU
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """A single Transformer decoder block."""
    # Added seq_length to the constructor
    def __init__(self, d_model, n_heads, dropout, seq_length):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        # Pass seq_length to MultiHeadAttention
        self.attn = MultiHeadAttention(d_model, n_heads, dropout, seq_length)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, dropout)

    def forward(self, x):
        # x shape: (batch_size, seq_length, d_model)
        # Apply self-attention with skip connection and normalization (pre-norm)
        x = x + self.attn(self.norm1(x)) # Add & Norm (pre-norm style) followed by attention result
        # Apply feed-forward with skip connection and normalization (pre-norm)
        x = x + self.ff(self.norm2(x))   # Add & Norm (pre-norm style) followed by FF result
        return x


class SimpleTransformerDecoder(nn.Module):
    """A minimal Transformer Decoder model for language modeling."""
    def __init__(self, vocab_size, d_model, n_layers, n_heads, seq_length, dropout):
        super().__init__()
        self.seq_length = seq_length # Store the maximum sequence length
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # Positional embeddings dimension should match the max sequence length
        self.position_embedding = nn.Embedding(seq_length, d_model)
        self.dropout = nn.Dropout(dropout)

        # Stack of transformer blocks
        self.blocks = nn.Sequential(*[
            # Pass seq_length to each TransformerBlock
            TransformerBlock(d_model, n_heads, dropout, seq_length)
            for _ in range(n_layers)
        ])

        # Final layer norm before the output projection
        self.norm_f = nn.LayerNorm(d_model)

        # Output layer: projects back to vocab size to get logits
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying between token embeddings and the language model head (optional but common)
        # self.token_embedding.weight = self.lm_head.weight # Requires vocab_size x d_model shapes match exactly

        # Report number of parameters
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Model has {n_params/1e6:.2f} Million parameters")

        # Initialize weights (optional, but can help convergence)
        self.apply(self._init_weights)


    def _init_weights(self, module):
        """Initializes weights with a small normal distribution."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        # Special initialization for the language model head if weight tying is not used
        # if isinstance(module, nn.Linear) and module is self.lm_head and not hasattr(self, 'token_embedding'):
        #     torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx):
        """
        Forward pass.
        Args:
            idx (torch.Tensor): Input tensor of shape (batch_size, current_seq_length)
                                containing token IDs.
        Returns:
            torch.Tensor: Logits for the next token, shape (batch_size, current_seq_length, vocab_size).
        """
        batch_size, current_seq_length = idx.size()

        # Ensure current sequence length doesn't exceed the maximum
        # This check is crucial if you are NOT cropping the input sequence before calling forward
        # If you ARE cropping in generate(), this check is redundant for forward()
        # For simplicity, we assume inputs to forward() are already cropped or are within seq_length
        # assert current_seq_length <= self.seq_length, f"Input sequence length {current_seq_length} exceeds model's max seq_length {self.seq_length}"


        # Token embeddings
        token_emb = self.token_embedding(idx) # (b, s, d_model)

        # Positional embeddings
        # Need positions from 0 up to current_seq_length - 1
        position = torch.arange(0, current_seq_length, dtype=torch.long, device=idx.device)
        position_emb = self.position_embedding(position) # (current_seq_length, d_model)

        # Combine embeddings
        # Broadcasting position_emb (s, d_model) to (b, s, d_model)
        x = self.dropout(token_emb + position_emb)

        # Pass through transformer blocks
        x = self.blocks(x) # (b, s, d_model)

        # Apply final layer norm
        x = self.norm_f(x)

        # Project to vocabulary size
        logits = self.lm_head(x) # (b, s, vocab_size)

        return logits

    @torch.no_grad() # Disable gradient calculations for inference
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generates new tokens based on a prompt.

        Args:
            idx (torch.Tensor): Input prompt token IDs, shape (1, current_seq_length).
            max_new_tokens (int): Maximum number of tokens to generate.
            temperature (float): Sampling temperature.
            top_k (int, optional): If not None, sample from the top_k most likely tokens.

        Returns:
            torch.Tensor: Generated token IDs, shape (1, initial_current_seq_length + max_new_tokens).
        """
        # idx is (batch_size, current_seq_length) - should be batch_size=1 for simple inference
        if idx.size(0) != 1:
             # Allow batch_size > 1 for batched generation if needed, but the core logic is the same
             # raise ValueError("Batch size must be 1 for simple generation.")
             pass # Keep current behavior allowing batch > 1 if your use case needs it

        for _ in range(max_new_tokens):
            # Crop idx to the last block_size tokens (model's context window)
            # This prevents sequence from growing infinitely and keeps it within the model's positional embedding limit
            idx_cond = idx[:, max(0, idx.size(1) - self.seq_length):] # (batch_size, min(current_seq_length, self.seq_length))

            # Get the predictions (logits) for the next token based on the cropped sequence
            # The forward pass will now handle the actual sequence length (idx_cond.size(1)) correctly
            logits = self(idx_cond) # (batch_size, current_seq_length_cond, vocab_size)

            # Focus only on the logits for the *last* token prediction in the sequence
            # This is the prediction for the token *after* idx_cond
            logits = logits[:, -1, :] # (batch_size, vocab_size)

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Apply top-k sampling if specified
            if top_k is not None:
                # Ensure top_k is not larger than the vocab size
                top_k = min(top_k, logits.size(-1))
                # Get top_k values and indices
                v, _ = torch.topk(logits, top_k)
                # Set logits of tokens not in top_k to -infinity
                logits[logits < v[:, [-1]]] = float('-inf')

            # Get probabilities
            probs = F.softmax(logits, dim=-1) # (batch_size, vocab_size)

            # Sample the next token
            # torch.multinomial expects input to be non-negative and sum to 1 along last dimension (probs)
            # If there are any NaNs or issues in probs, this might fail.
            # Add a small epsilon or check for NaNs if needed, but softmax usually handles this.
            idx_next = torch.multinomial(probs, num_samples=1) # (batch_size, 1)

            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (batch_size, current_seq_length + 1)

        return idx