import torch
import torch.nn as nn
from torch.nn import functional as F
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, leaves_list
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster
import numpy as np

# -----------------------------
# Hyperparameters (same values)
# -----------------------------
MAX_STEPS = 8000
EVAL_ITERATIONS = 100
EVAL_INTERVAL = 300

N_EMBD = 40
BLOCK_SIZE = 16
BATCH_SIZE = 4

def annotate_sequence(ax, chars, y=1.02, fontsize=10):
    s = "".join(chars)
    ax.text(
        0.5, y, s,
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=fontsize,
        fontfamily="monospace",
        clip_on=False,
    )
def sparse_ticks(chars, every=2):
    """Return a list same length as chars, with only every Nth label shown."""
    out = []
    for i, ch in enumerate(chars):
        out.append(ch if i % every == 0 else "")
    return out
# -----------------------------
# Data helpers
# -----------------------------
def load_text(path: str = "./input.txt") -> str:
    """Read the whole dataset text file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
def build_encoder(text: str):
    """
    Builds:
    - string_to_index: char -> int
    - index_to_string: int -> char
    - encode/decode functions
    """
    chars = sorted(list(set(text)))
    string_to_index = {ch: i for i, ch in enumerate(chars)}
    index_to_string = {i: ch for ch, i in string_to_index.items()}

    def encode(s: str):
        return [string_to_index[c] for c in s]

    def decode(indices):
        return "".join(index_to_string[i] for i in indices)

    vocab_size = len(chars)
    return encode, decode, vocab_size, index_to_string, string_to_index
def split_train_val(data: torch.Tensor, train_ratio: float = 0.9):
    """Split data into training and validation tensors."""
    n_train = int(train_ratio * data.size(0))
    train_data = data[:n_train]
    val_data = data[n_train:]
    return train_data, val_data
def get_batch(split: str, train_data: torch.Tensor, val_data: torch.Tensor):
    """
    Create a mini-batch of inputs/targets.

    X: (B, T) tokens
    Y: (B, T) next tokens (shifted by 1)
    """
    data = train_data if split == "train" else val_data

    # random starting indices
    ix = torch.randint(0, len(data) - BLOCK_SIZE, (BATCH_SIZE,))

    X = torch.stack([data[i : i + BLOCK_SIZE] for i in ix])
    Y = torch.stack([data[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
    return X, Y

# -----------------------------
# Plotting helpers (same plots)
# -----------------------------
def plot_bigram_logits_heatmap(model, itos):
    """
    Heatmap of the model's output weights (logits) per token.
    Note: In THIS model version, token_embedding is (vocab_size, N_EMBD),
    so this plot will not be a "bigram next-token table".
    (Keeping it unchanged because you asked not to add/change behavior.)
    """
    with torch.no_grad():
        weights = model.token_embedding.weight.detach().cpu().numpy()  # (vocab, N_EMBD)

    y_labels = [itos[i] for i in range(len(itos))]

    plt.figure(figsize=(12, 10))
    sns.heatmap(weights, yticklabels=y_labels)
    plt.xlabel("Embedding dimension")
    plt.ylabel("Token (current char)")
    plt.title("Token Embedding Weights Heatmap")
    plt.tight_layout()
    plt.show()
def plot_token_embeddings_pca_2d_with_hclust(
    model,
    itos,
    metric="cosine",
    method="average",
    k_clusters=6,
):
    with torch.no_grad():
        W = model.token_embedding.weight.detach().cpu().numpy()

    # --- HClust on original 40D ---
    d = pdist(W, metric=metric)
    Z = linkage(d, method=method)
    clusters = fcluster(Z, t=k_clusters, criterion="maxclust")

    # --- PCA to 2D ---
    X = W.astype(np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    X2 = X @ Vt[:2].T

    # --- Plot (ONE figure) ---
    plt.figure(figsize=(10, 8))
    sc = plt.scatter(
        X2[:, 0],
        X2[:, 1],
        c=clusters,
        s=45,
        alpha=0.85,
        cmap="tab10"
    )

    plt.title(f"PCA 2D of Token Embeddings + HClust coloring (k={k_clusters})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True, alpha=0.2)

    # Optional labels
    if len(itos) <= 80:
        for i in range(len(itos)):
            plt.text(X2[i, 0], X2[i, 1], itos[i], fontsize=9)

    # Colorbar on SAME figure
    cbar = plt.colorbar(sc)
    cbar.set_label("Cluster ID")

    plt.tight_layout()
    plt.show()
def plot_bigram_probability_heatmap_hclust(
    model,
    itos,
    metric="cosine",
    method="average",
    cluster_on="embeddings",   # "embeddings" | "probs"
    zscore_rows=False,
):
    """
    Softmax heatmap (over embedding dims) but with token rows reordered by
    hierarchical clustering so similar tokens appear close together.

    cluster_on:
      - "embeddings": cluster using raw token_embedding.weight (recommended)
      - "probs":      cluster using the softmax probabilities themselves
    """
    with torch.no_grad():
        W = model.token_embedding.weight.detach().cpu().numpy()   # (vocab, N_EMBD)
        P = torch.softmax(model.token_embedding.weight, dim=1).detach().cpu().numpy()

    # Choose what you cluster on
    X = W if cluster_on == "embeddings" else P

    # Optional row-wise z-score (often helpful for clustering)
    if zscore_rows:
        X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)

    # Hierarchical clustering to get row order
    if X.shape[0] > 2:
        d = pdist(X, metric=metric)
        Z = linkage(d, method=method)
        row_order = leaves_list(Z)
    else:
        row_order = np.arange(X.shape[0])

    # Reorder the SOFTMAX matrix by that order
    P_ord = P[row_order]
    y_labels = [itos[i] for i in row_order]

    plt.figure(figsize=(12, 10))
    sns.heatmap(P_ord, yticklabels=y_labels, cmap="magma")
    plt.xlabel("Embedding dimension")
    plt.ylabel("Token (clustered)")
    plt.title(f"Token Embedding Softmax Heatmap (rows clustered on {cluster_on}; {metric}/{method})")
    plt.tight_layout()
    plt.show()
def plot_bigram_probability_heatmap(model, itos):
    """
    This softmax is applied across embedding dimensions in this version
    (since token_embedding is vocab_size x N_EMBD).
    Keeping it as-is (no behavior changes).
    """
    with torch.no_grad():
        weights = model.token_embedding.weight  # (vocab, N_EMBD)
        probs = torch.softmax(weights, dim=1).cpu().numpy()

    y_labels = [itos[i] for i in range(len(itos))]

    plt.figure(figsize=(12, 10))
    sns.heatmap(probs, yticklabels=y_labels, cmap="magma")
    plt.xlabel("Embedding dimension")
    plt.ylabel("Token (current char)")
    plt.title("Token Embedding Softmax (over embedding dims)")
    plt.tight_layout()
    plt.show()
def plot_learning_curve(steps, train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(steps, train_losses, label="Training Loss")
    plt.plot(steps, val_losses, label="Validation Loss")
    plt.title("Learning Curve: Training vs Validation Loss")
    plt.xlabel(f"Training Steps (evaluated every {EVAL_INTERVAL} steps)")
    plt.ylabel("Cross-entropy loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
def plot_heatmaps(out, tril, wei, chars):
    # 1) Causal mask heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(tril.numpy(), cmap="gray_r", cbar=False)
    plt.title("Causal Mask (tril) — allowed=1 blocked=0")
    plt.xlabel("Key position (look at)")
    plt.ylabel("Query position (looking from)")
    plt.tight_layout()
    plt.show()

    # 2) Attention weights heatmap (batch 0)
    plt.figure(figsize=(6, 5))
    sns.heatmap(wei[0].detach().cpu().numpy(), cmap="magma", vmin=0.0, vmax=1.0)
    plt.title("Attention Weights (wei) — batch 0")
    plt.xlabel("Key position (look at)")
    plt.ylabel("Query position (looking from)")
    plt.tight_layout()
    plt.show()

    # 3) Output after attention heatmap (batch 0)
    plt.figure(figsize=(8, 4))
    sns.heatmap(out[0].detach().cpu().numpy(), cmap="viridis")
    plt.title("Output after attention (out) — batch 0 (T x head_size)")
    plt.xlabel("Feature index (head_size)")
    plt.ylabel("Time position (T)")
    plt.tight_layout()
    plt.show()

    # 4) Attention weights — readable version
    plt.figure(figsize=(7, 6))
    ax = plt.gca()

    W = wei[0].detach().cpu().numpy()

    # SHOW ALL CHARS (T=16)
    xt = chars
    yt = chars

    sns.heatmap(
        W,
        ax=ax,
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
        xticklabels=xt,
        yticklabels=yt,
        square=True,
        cbar=True
    )

    ax.tick_params(axis="x", labelrotation=0)
    ax.tick_params(axis="y", labelrotation=0)

    # FIXED call signature
    annotate_sequence(ax, chars, y=1.02, fontsize=12)

    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")
    plt.tight_layout()
    plt.show()

def plot_all_heads_snapshot(snap, title=""):
    """
    Grid: H rows (heads) × 8 columns:
    q, W_Q, k, W_K, v, W_V, wei, out

    Fixes readability by:
      - turning OFF char tick labels in WEI plots (too cramped)
      - writing the full token string above every WEI panel
      - using constrained_layout instead of tight_layout
    """
    chars = snap["chars"]
    H, T, hs = snap["q"].shape
    C = snap["W_Q"].shape[1]

    cols = ["q", "W_Q", "k", "W_K", "v", "W_V", "wei", "out"]
    col_titles = [
        "Q (T×hs)", "W_Q (C×hs)",
        "K (T×hs)", "W_K (C×hs)",
        "V (T×hs)", "W_V (C×hs)",
        "Attention WEI (T×T)",
        "Head OUT (T×hs)",
    ]

    # shared color scales per column
    vlims = {}
    for c in cols:
        A = np.asarray(snap[c])
        if c == "wei":
            vlims[c] = (0.0, 1.0)
        else:
            vlims[c] = (float(A.min()), float(A.max()))

    fig, axes = plt.subplots(
        H, len(cols),
        figsize=(4.3 * len(cols), 2.8 * H),
        constrained_layout=True
    )
    if H == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(H):
        for j, c in enumerate(cols):
            ax = axes[i, j]
            data = np.asarray(snap[c][i])

            cmap = "magma" if c == "wei" else "viridis"
            vmin, vmax = vlims[c]

            # axis tick labels
            if c in ("q", "k", "v", "out"):
                # these are still cramped; show none or only on first col if you want
                ytick = False
                xtick = False
            elif c in ("W_Q", "W_K", "W_V"):
                ytick = False
                xtick = list(range(hs)) if i == H - 1 else False
            elif c == "wei":
                # KEY FIX: no per-char ticks in the grid
                ytick = False
                xtick = False
            else:
                ytick = xtick = False

            sns.heatmap(
                data,
                ax=ax,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                cbar=True,
                yticklabels=ytick,
                xticklabels=xtick,
                square=True if c == "wei" else False,
            )

            # Titles only on top row
            if i == 0:
                ax.set_title(col_titles[j], fontsize=11)

            # Labels
            if j == 0:
                ax.set_ylabel(f"head {i}", fontsize=11)
            else:
                ax.set_ylabel("")

            if i == H - 1:
                if c == "wei":
                    ax.set_xlabel("Key position (T)", fontsize=10)
                elif c in ("q", "k", "v", "out"):
                    ax.set_xlabel("Head feature (hs)", fontsize=10)
                elif c in ("W_Q", "W_K", "W_V"):
                    ax.set_xlabel("Output feature (hs)", fontsize=10)
            else:
                ax.set_xlabel("")

            # KEY FIX: show the full sequence above EVERY WEI panel
            if c == "wei":
                annotate_sequence(ax, chars, y=1.02, fontsize=12)
                # also label the y-axis meaning on WEI column
                ax.set_ylabel(f"head {i}\nQuery pos", fontsize=10)

    fig.suptitle(title, fontsize=16)
    plt.show()

@torch.no_grad()
def collect_epoch_stats(model, train_data, val_data):
    model.eval()

    # single representative batch
    X, _ = get_batch("train", train_data, val_data)

    # embeddings
    token_emb = model.token_embedding(X)  # (B,T,C)
    pos_emb = model.position_embedding_table(
        torch.arange(token_emb.size(1), device=X.device)
    )
    x = token_emb + pos_emb

    # attention internals
    out, wei = model.sa_heads(x)

    # q,k,v (recompute explicitly)
    q = model.sa_heads.query(x)
    k = model.sa_heads.key(x)
    v = model.sa_heads.value(x)

    stats = {
        "emb_norm": token_emb.norm(dim=-1).mean().item(),
        "x_norm": x.norm(dim=-1).mean().item(),
        "q_norm": q.norm(dim=-1).mean().item(),
        "k_norm": k.norm(dim=-1).mean().item(),
        "v_norm": v.norm(dim=-1).mean().item(),
        "attn_entropy": (-wei * (wei + 1e-9).log()).sum(dim=-1).mean().item(),
        "out_norm": out.norm(dim=-1).mean().item(),
    }

    return stats
@torch.no_grad()
def get_attention_snapshot_from_X(model, X, itos):
    model.eval()
    B, T = X.shape
    chars = [itos[i.item()] for i in X[0]]

    token_emb = model.token_embedding(X)
    pos = torch.arange(T, device=X.device) % BLOCK_SIZE
    pos_emb = model.position_embedding_table(pos)
    x = token_emb + pos_emb

    out, wei = model.sa_heads(x)  # wei will be a tuple/list of per-head weights

    snap = {
        "chars": chars,
        "token_emb": token_emb[0].cpu().numpy(),
        "x": x[0].cpu().numpy(),
        "wei": np.stack([w[0].cpu().numpy() for w in wei], axis=0),  # (num_heads, T, T)
        "out": out[0].cpu().numpy(),
    }
    model.train()
    return snap
@torch.no_grad()
def get_multihead_snapshot_from_X(model, X, itos):
    model.eval()
    B, T = X.shape
    chars = [itos[i.item()] for i in X[0]]

    token_emb = model.token_embedding(X)                          # (B,T,C)
    pos = torch.arange(T, device=X.device) % BLOCK_SIZE
    pos_emb = model.position_embedding_table(pos)                 # (T,C)
    x = token_emb + pos_emb                                       # (B,T,C)

    q_all, k_all, v_all = [], [], []
    wei_all, out_all = [], []

    # NEW: store weights
    Wq_all, Wk_all, Wv_all = [], [], []

    for h in model.sa_heads.heads:
        # activations
        q = h.query(x)                  # (B,T,hs)
        k = h.key(x)                    # (B,T,hs)
        v = h.value(x)                  # (B,T,hs)
        out, wei = h(x)                 # out (B,T,hs), wei (B,T,T)

        q_all.append(q[0].cpu().numpy())
        k_all.append(k[0].cpu().numpy())
        v_all.append(v[0].cpu().numpy())
        out_all.append(out[0].cpu().numpy())
        wei_all.append(wei[0].cpu().numpy())

        # weights (parameters)
        Wq_all.append(h.query.weight.cpu().numpy())  # (hs, C)
        Wk_all.append(h.key.weight.cpu().numpy())    # (hs, C)
        Wv_all.append(h.value.weight.cpu().numpy())  # (hs, C)

    snap = {
        "chars": chars,

        # embeddings
        "token_emb": token_emb[0].cpu().numpy(),     # (T,C)
        "pos_emb": pos_emb.cpu().numpy(),             # (T,C)
        "x": x[0].cpu().numpy(),                      # (T,C)

        # activations
        "q": np.stack(q_all, axis=0),                 # (H,T,hs)
        "k": np.stack(k_all, axis=0),                 # (H,T,hs)
        "v": np.stack(v_all, axis=0),                 # (H,T,hs)
        "out": np.stack(out_all, axis=0),             # (H,T,hs)
        "wei": np.stack(wei_all, axis=0),             # (H,T,T)

        # parameters (NOTE transpose for readability)
        "W_Q": np.stack(Wq_all, axis=0).transpose(0, 2, 1),  # (H,C,hs)
        "W_K": np.stack(Wk_all, axis=0).transpose(0, 2, 1),  # (H,C,hs)
        "W_V": np.stack(Wv_all, axis=0).transpose(0, 2, 1),  # (H,C,hs)
    }

    model.train()
    return snap

def plot_embedding_triplet_matrix(model, X, itos, title="Embeddings: pos, token, x"):
    model.eval()
    B, T = X.shape

    chars = [itos[i.item()] for i in X[0]]
    ytick = chars

    token_emb = model.token_embedding(X)
    pos = torch.arange(T, device=X.device) % BLOCK_SIZE
    pos_emb = model.position_embedding_table(pos)
    x = token_emb + pos_emb

    tok0 = token_emb[0].detach().cpu().numpy()
    pos0 = pos_emb.detach().cpu().numpy()
    x0   = x[0].detach().cpu().numpy()

    mats = [pos0, tok0, x0]
    titles = ["pos_emb (T×C)", "token_emb (T×C)", "x = token + pos (T×C)"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # and REMOVE fig.tight_layout(...) entirely

    # and REMOVE fig.tight_layout(...) entirely

    for j in range(3):
        ax = axes[j]
        sns.heatmap(
            mats[j],
            ax=ax,
            cmap="viridis",
            cbar=True,
            yticklabels=ytick,
            xticklabels=False
        )
        ax.tick_params(axis="y", labelrotation=0)
        ax.set_title(titles[j], fontsize=12)
        ax.set_xlabel("Embedding dim (C)")
        ax.set_ylabel("")

        # put full sequence once (only on first panel)
        if j == 0:
            annotate_sequence(ax, chars, y=1.02, fontsize=12)
            ax.set_ylabel("Token position")

    fig.suptitle(title, fontsize=16)
    # fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    model.train()

# -----------------------------
# Loss evaluation helper
# -----------------------------
@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    """
    Average loss on 'train' and 'validation' splits.
    """
    out = {}
    model.eval()

    for split in ["train", "validation"]:
        losses = torch.zeros(EVAL_ITERATIONS)
        for i in range(EVAL_ITERATIONS):
            X, Y = get_batch(split, train_data, val_data)
            _, loss = model(X, Y)
            losses[i] = loss.item()
        out[split] = losses.mean()

    model.train()
    return out

class Head(nn.Module):
        """One head of self-attention (Version 4)."""

        def __init__(self, n_embd: int, head_size: int):
            super().__init__()
            self.key = nn.Linear(n_embd, head_size, bias=False)
            self.query = nn.Linear(n_embd, head_size, bias=False)
            self.value = nn.Linear(n_embd, head_size, bias=False)

            # causal mask
            self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

        def forward(self, x):
            # x: (B, T, C)
            B, T, C = x.shape

            k = self.key(x)  # (B, T, head_size)
            q = self.query(x)  # (B, T, head_size)


            weight = q @ k.transpose(-2, -1)  # (B, T, T)
            weight = weight.masked_fill(self.tril[:T, :T].to(x.device) == 0, float("-inf"))
            weight = F.softmax(weight, dim=-1)  # (B, T, T)

            v = self.value(x)  # (B, T, head_size)
            out = weight @ v  # (B, T, head_size)

            return out, weight



# -----------------------------
# Model
# -----------------------------


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, n_embd, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size) for _ in range(num_heads)])

    def forward(self, x):
        outs, weights = zip(*[h(x) for h in self.heads])   # each head returns (out, wei)
        out = torch.cat(outs, dim=-1)
        return out, weights


class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)

class BigramLanguageModel(nn.Module):
    """
    - token_embedding: maps token id -> embedding vector (N_EMBD)
    - lm_head: maps embedding -> logits over vocab
    """
    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, N_EMBD)           # (vocab, N_EMBD)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)  # (block_size, N_EMBD)
        self.sa_heads = MultiHeadAttention(4, N_EMBD, N_EMBD // 4)
        self.lm_head = nn.Linear(N_EMBD, vocab_size)                      # (N_EMBD -> vocab)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, return_wei: bool = False):
        B, T = idx.shape

        token_emb = self.token_embedding(idx)  # (B,T,C)
        positions = torch.arange(T, device=idx.device) % BLOCK_SIZE
        pos_emb = self.position_embedding_table(positions)  # (T,C)

        x = token_emb + pos_emb  # (B,T,C)

        x, wei = self.sa_heads(x)  # (B,T,C) + (B,T,T)

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
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)            # (B, T, vocab)
            last_logits = logits[:, -1, :]   # (B, vocab)
            probs = F.softmax(last_logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat([idx, idx_next], dim=1)             # (B, T+1)
        return idx

# -----------------------------
# Main training script
# -----------------------------
def main():
    torch.manual_seed(0)

    # 1) Load + encode data
    text = load_text()
    print("Raw text length:", len(text))

    encode, decode, vocab_size, itos, stoi = build_encoder(text)
    print("Vocabulary size:", vocab_size)

    data = torch.tensor(encode(text), dtype=torch.long)
    train_data, val_data = split_train_val(data, train_ratio=0.9)
    print("Train shape:", train_data.shape, "Val shape:", val_data.shape)

    # 2) Create model + optimizer
    model = BigramLanguageModel(vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # 3) Training loop
    steps_for_plot = []
    train_loss_history = []
    val_loss_history = []


    X_fixed, _ = get_batch("train", train_data, val_data)
    plot_embedding_triplet_matrix(model, X_fixed, itos, title="Before training: pos vs token vs x")

    snap_start = get_multihead_snapshot_from_X(model, X_fixed, itos)
    plot_all_heads_snapshot(snap_start, "BEFORE: all heads (Q,K,V,WEI,OUT + weights)")

    for step in range(MAX_STEPS):
        # Evaluate occasionally
        if step % EVAL_INTERVAL == 0:
            losses = estimate_loss(model, train_data, val_data)

            steps_for_plot.append(step)
            train_loss_history.append(losses["train"])
            val_loss_history.append(losses["validation"])

            print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['validation']:.4f}")

        # One batch
        X, Y = get_batch("train", train_data, val_data)

        # Forward + backward + update
        _, loss = model(X, Y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    plot_embedding_triplet_matrix(model, X_fixed, itos, title="AFTER: pos vs token vs x")

    snap_after = get_multihead_snapshot_from_X(model, X_fixed, itos)
    plot_all_heads_snapshot(snap_after, "AFTER: all heads (Q,K,V,WEI,OUT + weights)")


    # plot_two_snapshots_grid(snap_start, snap_end, title_top="Step 0 snapshot", title_bottom="Final snapshot")
    # 4) Show results
    print("Final loss:", loss.item())

    # Generate some text
    start = torch.zeros((1, 1), dtype=torch.long)
    sample = model.generate(start, max_new_tokens=1000)[0].tolist()
    generated_text = decode(sample)  # print(decode(sample))
    with open("generated_text_ver3.txt", "w", encoding="utf-8") as f:
        f.write(generated_text)

    # 5) Plots
    plot_learning_curve(steps_for_plot, train_loss_history, val_loss_history)
    plot_bigram_logits_heatmap(model, itos)
    plot_bigram_probability_heatmap(model, itos)
    plot_token_embeddings_pca_2d_with_hclust(model, itos)

    with torch.no_grad():
        probs = torch.softmax(model.token_embedding.weight, dim=1).cpu().numpy()

    sns.clustermap(
        probs,
        metric="cosine",    # good for embeddings
        method="average",
        row_cluster=True,   # cluster tokens
        col_cluster=False,  # don’t cluster embedding dims unless you want
        yticklabels=[itos[i] for i in range(len(itos))],
        cmap="magma",
        figsize=(12,10)
    )
    plt.show()

    plot_embedding_triplet_matrix(model, X_fixed, itos, title="After training: pos vs token vs x")



if __name__ == "__main__":
    main()
