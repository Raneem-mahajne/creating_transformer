"""Transformer LM and components."""
import torch
import torch.nn as nn
from torch.nn import functional as F


class Head(nn.Module):
    """Single self-attention head with causal mask and scaling."""

    def __init__(self, n_embd: int, head_size: int, block_size: int):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        weight = q @ k.transpose(-2, -1)
        # Scale by sqrt(d_k) to prevent attention from becoming too peaked
        weight = weight / (self.head_size ** 0.5)
        weight = weight.masked_fill(self.tril[:T, :T].to(x.device) == 0, float("-inf"))
        weight = F.softmax(weight, dim=-1)
        v = self.value(x)
        out = weight @ v
        return out, weight


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention."""

    def __init__(self, num_heads: int, n_embd: int, head_size: int, block_size: int):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size) for _ in range(num_heads)])

    def forward(self, x):
        outs, weights = zip(*[h(x) for h in self.heads])
        return torch.cat(outs, dim=-1), weights


class FeedForward(nn.Module):
    def __init__(self, n_embd: int, ffwd_mult: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, ffwd_mult * n_embd),
            nn.ReLU(),
            nn.Linear(ffwd_mult * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)


class BigramLanguageModel(nn.Module):
    """
    - token_embedding: maps token id -> embedding vector (n_embd)
    - lm_head: maps attention output -> logits over vocab
    """
    def __init__(self, vocab_size: int, n_embd: int, block_size: int, num_heads: int, head_size: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)           # (vocab, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)  # (block_size, n_embd)
        self.sa_heads = MultiHeadAttention(num_heads, n_embd, head_size, block_size)
        # Projection removed: with single head and head_size=n_embd, attention output already matches n_embd
        self.ffwd = FeedForward(n_embd, ffwd_mult=16)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.block_size = block_size

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, return_wei: bool = False):
        B, T = idx.shape

        token_emb = self.token_embedding(idx)  # (B,T,C)
        positions = torch.arange(T, device=idx.device) % self.block_size
        pos_emb = self.position_embedding_table(positions)  # (T,C)

        x = token_emb + pos_emb  # (B,T,n_embd)

        attn_out, wei = self.sa_heads(x)  # (B,T,n_embd) - already correct dimension
        x = x + attn_out                  # residual connection
        x = x + self.ffwd(x)              # feedforward + residual

        logits = self.lm_head(x)  # (B,T,vocab)

        loss = None
        if targets is not None:
            Bt, Tt, Cc = logits.shape
            loss = F.cross_entropy(logits.view(Bt * Tt, Cc), targets.view(Bt * Tt))

        if return_wei:
            return logits, loss, wei
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int):
        """
        Autoregressive generation.
        idx: (B, T)
        returns idx extended to (B, T + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)            # (B, T, vocab)
            last_logits = logits[:, -1, :]   # (B, vocab)
            probs = F.softmax(last_logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat([idx, idx_next], dim=1)             # (B, T+1)
        return idx


# Alias for backward compatibility
TransformerLM = BigramLanguageModel
