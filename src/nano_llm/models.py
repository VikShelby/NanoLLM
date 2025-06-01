
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention module."""
    def __init__(self, d_model, n_heads, dropout, seq_length):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.d_model = d_model


        self.max_seq_length = seq_length

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)







        mask = torch.triu(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool), diagonal=1)
        self.register_buffer('mask', mask.view(1, 1, self.max_seq_length, self.max_seq_length))




    def forward(self, x):
        batch_size, current_seq_length, _ = x.size()

        q = self.wq(x).view(batch_size, current_seq_length, self.n_heads, self.d_k).transpose(1, 2)
        k = self.wk(x).view(batch_size, current_seq_length, self.n_heads, self.d_k).transpose(1, 2)
        v = self.wv(x).view(batch_size, current_seq_length, self.n_heads, self.d_k).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(self.d_k))






        causal_mask_for_scores = self.mask[:, :, :current_seq_length, :current_seq_length]
        attn_scores = attn_scores.masked_fill(causal_mask_for_scores, float('-inf'))










        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        attention = torch.matmul(attn_probs, v)
        attention = attention.transpose(1, 2).contiguous().view(batch_size, current_seq_length, self.d_model)

        output = self.wo(attention)
        output = self.resid_dropout(output)
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout, seq_length):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout, seq_length)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

class SimpleTransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, seq_length, dropout):
        super().__init__()

        if vocab_size is None:
            raise ValueError("vocab_size must be specified for SimpleTransformerDecoder.")

        self.max_seq_length = seq_length
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(self.max_seq_length, d_model)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout, self.max_seq_length)
            for _ in range(n_layers)
        ])

        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)




        self.apply(self._init_weights)

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_nonemb_params = sum(p.numel() for n, p in self.named_parameters()
                              if p.requires_grad and "token_embedding" not in n and "lm_head" not in n and "position_embedding" not in n)

        print(f"Model initialized:")
        print(f"  Total parameters: {n_params/1e6:.2f}M")
        print(f"  - Token Embedding parameters: {self.token_embedding.weight.numel()/1e6:.2f}M")
        print(f"  - Positional Embedding parameters: {self.position_embedding.weight.numel()/1e6:.2f}M")
        print(f"  - Transformer Blocks (core) parameters: {n_nonemb_params/1e6:.2f}M")
        print(f"  - LM Head parameters: {self.lm_head.weight.numel()/1e6:.2f}M")
        if hasattr(self.token_embedding, 'weight') and self.token_embedding.weight is self.lm_head.weight:
            print("  (Note: Token Embedding and LM Head weights are tied)")


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)



    def forward(self, idx):
        batch_size, current_seq_length = idx.size()

        if current_seq_length > self.max_seq_length:
            raise ValueError(
                f"Input sequence length ({current_seq_length}) "
                f"exceeds model's maximum configured sequence length ({self.max_seq_length}). "
                f"Input should be cropped to self.max_seq_length before calling forward."
            )

        token_emb = self.token_embedding(idx)
        position = torch.arange(0, current_seq_length, dtype=torch.long, device=idx.device)
        position_emb = self.position_embedding(position)

        x = self.dropout(token_emb + position_emb.unsqueeze(0))

        for block in self.blocks:
            x = block(x)

        x = self.norm_f(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):

            idx_cond = idx[:, -self.max_seq_length:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]

            if temperature == 0.0:
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)
        self.train()
        return idx