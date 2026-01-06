import torch
import torch.nn as nn
from torch.nn import functional as F
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, leaves_list
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.cluster.hierarchy import fcluster
import numpy as np
import os
import random
import sys
from IntegerStringGenerator import IntegerStringGenerator, OddEvenIndexRule
from config_loader import load_config, get_generator_from_config

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
def generate_integer_string_data(generator: IntegerStringGenerator, num_sequences: int = 1000, 
                                  min_length: int = 50, max_length: int = 200) -> list[list[int]]:
    """
    Generate integer sequences using the generator and return as a list of sequences.
    
    Args:
        generator: IntegerStringGenerator instance
        num_sequences: Number of sequences to generate
        min_length: Minimum sequence length
        max_length: Maximum sequence length
        
    Returns:
        List of sequences (each sequence is a list of integers)
    """
    sequences = generator.generate_dataset(num_sequences, min_length, max_length)
    return sequences

def build_encoder_for_integers(min_value: int = 0, max_value: int = 20):
    """
    Builds encoder/decoder for integer tokens.
    Each integer value (min_value to max_value) is a unique token.
    Works directly with integers - no text conversion.
    
    Args:
        min_value: Minimum integer value (inclusive)
        max_value: Maximum integer value (inclusive)
        
    Returns:
        encode, decode, vocab_size, index_to_string, string_to_index
    """
    # Vocabulary: all integers from min_value to max_value
    vocab_size = max_value - min_value + 1
    
    # index_to_string: maps token index -> integer value string (only for plotting/display)
    # string_to_index: maps integer value string -> token index (for compatibility)
    index_to_string = {i: str(min_value + i) for i in range(vocab_size)}
    string_to_index = {str(min_value + i): i for i in range(vocab_size)}
    
    def encode(integers: list[int]) -> list[int]:
        """
        Encode integer values directly to token indices.
        Args:
            integers: List of integer values (e.g., [0, 1, 2, 15])
        Returns:
            List of token indices (integers)
        """
        return [val - min_value for val in integers]
    
    def decode(token_indices) -> list[int]:
        """
        Decode token indices directly back to integer values.
        Args:
            token_indices: List of token indices (integers)
        Returns:
            List of integer values
        """
        return [idx + min_value for idx in token_indices]
    
    return encode, decode, vocab_size, index_to_string, string_to_index
def split_train_val_sequences(sequences: list[list[int]], train_ratio: float = 0.9):
    """Split sequences into training and validation sets."""
    n_train = int(train_ratio * len(sequences))
    train_sequences = sequences[:n_train]
    val_sequences = sequences[n_train:]
    return train_sequences, val_sequences

def get_batch_from_sequences(sequences: list[list[int]], block_size: int, batch_size: int):
    """
    Sample batches that respect sequence boundaries.
    Only samples from sequences long enough to contain a block.
    
    Args:
        sequences: List of sequences (each is a list of integers)
        block_size: Size of the context block
        batch_size: Number of samples in the batch
        
    Returns:
        X, Y tensors of shape (batch_size, block_size)
    """
    # Filter sequences that are long enough
    valid_sequences = [seq for seq in sequences if len(seq) >= block_size + 1]
    if not valid_sequences:
        raise ValueError(f"No sequences long enough for block_size {block_size}")
    
    batch_x, batch_y = [], []
    for _ in range(batch_size):
        seq = random.choice(valid_sequences)
        if len(seq) <= block_size + 1:
            start_idx = 0
        else:
            start_idx = random.randint(0, len(seq) - block_size - 1)
        
        x = seq[start_idx:start_idx + block_size]
        y = seq[start_idx + 1:start_idx + block_size + 1]
        batch_x.append(x)
        batch_y.append(y)
    
    return torch.tensor(batch_x, dtype=torch.long), torch.tensor(batch_y, dtype=torch.long)

# -----------------------------
# Plotting helpers (same plots)
# -----------------------------
def plot_token_embeddings_heatmap(model, itos, save_path=None):
    """
    Plot token embeddings as a heatmap: tokens (rows) x embedding dimensions (columns).
    """
    with torch.no_grad():
        embeddings = model.token_embedding.weight.detach().cpu().numpy()  # (vocab, N_EMBD)

    y_labels = [itos[i] for i in range(len(itos))]

    plt.figure(figsize=(14, 10))
    x_labels = list(range(embeddings.shape[1]))
    vocab_size, n_embd = embeddings.shape
    ax = sns.heatmap(embeddings, yticklabels=y_labels, xticklabels=x_labels, cmap="RdBu_r", center=0)
    plt.xlabel("Embedding dimension")
    plt.ylabel("Token")
    plt.title(f"Token Embeddings Heatmap ({vocab_size}×{n_embd})")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_bigram_logits_heatmap(model, itos, save_path=None):
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
    x_labels = list(range(weights.shape[1]))
    vocab_size, n_embd = weights.shape
    sns.heatmap(weights, yticklabels=y_labels, xticklabels=x_labels)
    plt.xlabel("Embedding dimension")
    plt.ylabel("Token")
    plt.title(f"Token Embedding Weights Heatmap ({vocab_size}×{n_embd})")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
def plot_token_embeddings_pca_2d_with_hclust(
    model,
    itos,
    metric="cosine",
    method="average",
    k_clusters=6,
    save_path=None,
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

    vocab_size = W.shape[0]
    plt.title(f"PCA 2D of Token Embeddings + HClust coloring (vocab={vocab_size}, k={k_clusters})")
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
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
def plot_bigram_probability_heatmap_hclust(
    model,
    itos,
    metric="cosine",
    method="average",
    cluster_on="embeddings",   # "embeddings" | "probs"
    zscore_rows=False,
    save_path=None,
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
    x_labels = list(range(P_ord.shape[1]))
    vocab_size, n_embd = P_ord.shape
    sns.heatmap(P_ord, yticklabels=y_labels, xticklabels=x_labels, cmap="magma")
    plt.xlabel("Embedding dimension")
    plt.ylabel("Token (clustered)")
    plt.title(f"Token Embedding Softmax Heatmap ({vocab_size}×{n_embd}, rows clustered on {cluster_on}; {metric}/{method})")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
def plot_bigram_probability_heatmap(model, itos, save_path=None):
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
    x_labels = list(range(probs.shape[1]))
    vocab_size, n_embd = probs.shape
    sns.heatmap(probs, yticklabels=y_labels, xticklabels=x_labels, cmap="magma")
    plt.xlabel("Embedding dimension")
    plt.ylabel("Token")
    plt.title(f"Token Embedding Softmax ({vocab_size}×{n_embd}, over embedding dims)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
def plot_learning_curve(steps, train_losses, val_losses, save_path=None, eval_interval=None):
    plt.figure(figsize=(10, 6))
    plt.plot(steps, train_losses, label="Training Loss")
    plt.plot(steps, val_losses, label="Validation Loss")
    plt.title("Learning Curve: Training vs Validation Loss")
    eval_interval_str = f" (evaluated every {eval_interval} steps)" if eval_interval else ""
    plt.xlabel(f"Training Steps{eval_interval_str}")
    plt.ylabel("Cross-entropy loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
def plot_heatmaps(out, tril, wei, chars, save_dir=None):
    # 1) Causal mask heatmap
    plt.figure(figsize=(6, 5))
    tril_np = tril.numpy()
    x_labels = list(range(tril_np.shape[1]))
    y_labels = list(range(tril_np.shape[0]))
    sns.heatmap(tril_np, cmap="gray_r", cbar=False, xticklabels=x_labels, yticklabels=y_labels)
    t_size = tril_np.shape[0]
    plt.title(f"Causal Mask (tril) {t_size}×{t_size} — allowed=1 blocked=0")
    plt.xlabel("Key position (look at)")
    plt.ylabel("Query position (looking from)")
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "causal_mask.png"))
        plt.close()
    else:
        plt.show()

    # 2) Attention weights heatmap (batch 0)
    plt.figure(figsize=(6, 5))
    wei_np = wei[0].detach().cpu().numpy()
    x_labels = list(range(wei_np.shape[1]))
    y_labels = list(range(wei_np.shape[0]))
    sns.heatmap(wei_np, cmap="magma", vmin=0.0, vmax=1.0, xticklabels=x_labels, yticklabels=y_labels)
    t_size = wei_np.shape[0]
    plt.title(f"Attention Weights (wei) {t_size}×{t_size} — batch 0")
    plt.xlabel("Key position (look at)")
    plt.ylabel("Query position (looking from)")
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "attention_weights.png"))
        plt.close()
    else:
        plt.show()

    # 3) Output after attention heatmap (batch 0)
    plt.figure(figsize=(8, 4))
    out_np = out[0].detach().cpu().numpy()
    x_labels = list(range(out_np.shape[1]))
    y_labels = list(range(out_np.shape[0]))
    sns.heatmap(out_np, cmap="viridis", xticklabels=x_labels, yticklabels=y_labels)
    t_size, head_size = out_np.shape
    plt.title(f"Output after attention (out) {t_size}×{head_size} — batch 0")
    plt.xlabel("Feature index (head_size)")
    plt.ylabel("Time position (T)")
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "output_after_attention.png"))
        plt.close()
    else:
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

    t_size = W.shape[0]
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")
    plt.title(f"Attention Weights {t_size}×{t_size} — readable")
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "attention_weights_readable.png"))
        plt.close()
    else:
        plt.show()

def plot_all_heads_snapshot(snap, title="", save_path=None):
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
        f"Q {T}×{hs}", f"W_Q {C}×{hs}",
        f"K {T}×{hs}", f"W_K {C}×{hs}",
        f"V {T}×{hs}", f"W_V {C}×{hs}",
        f"Attention WEI {T}×{T}",
        f"Head OUT {T}×{hs}",
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

            # axis tick labels - show on all rows and columns
            ytick = list(range(data.shape[0]))
            xtick = list(range(data.shape[1]))

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
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

@torch.no_grad()
def collect_epoch_stats(model, train_sequences, block_size, batch_size):
    model.eval()

    # single representative batch
    X, _ = get_batch_from_sequences(train_sequences, block_size, batch_size)

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
    pos = torch.arange(T, device=X.device) % model.block_size
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
    pos = torch.arange(T, device=X.device) % model.block_size
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

@torch.no_grad()
def plot_embeddings_pca(model, itos, save_path=None):
    """
    Plot embeddings: heatmap, clustered heatmap, and PCA 2D
    """
    model.eval()
    
    # Get embeddings
    embeddings = model.token_embedding.weight.detach().cpu().numpy()  # (vocab, N_EMBD)
    vocab_size, n_embd = embeddings.shape
    y_labels = [itos[i] for i in range(vocab_size)]
    
    # Hierarchical clustering for embeddings
    d = pdist(embeddings, metric="cosine")
    Z = linkage(d, method="average")
    row_order = leaves_list(Z)
    embeddings_clustered = embeddings[row_order]
    y_labels_clustered = [itos[i] for i in row_order]
    
    # PCA
    X_emb = embeddings.astype(np.float64)
    X_emb = X_emb - X_emb.mean(axis=0, keepdims=True)
    clusters = fcluster(Z, t=6, criterion="maxclust")
    
    # Create figure: 1 row, 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Embeddings heatmap
    ax1 = axes[0]
    x_labels = list(range(embeddings.shape[1]))
    sns.heatmap(embeddings, yticklabels=y_labels, xticklabels=x_labels, cmap="RdBu_r", center=0, ax=ax1)
    ax1.set_title(f"Embeddings (vocab×embd={vocab_size}×{n_embd})", fontsize=11)
    ax1.set_xlabel("Embedding dim")
    ax1.set_ylabel("Token")
    
    # Clustered embeddings
    ax2 = axes[1]
    x_labels = list(range(embeddings_clustered.shape[1]))
    sns.heatmap(embeddings_clustered, yticklabels=y_labels_clustered, xticklabels=x_labels, cmap="RdBu_r", center=0, ax=ax2)
    ax2.set_title(f"Embeddings Clustered (vocab×embd={vocab_size}×{n_embd})", fontsize=11)
    ax2.set_xlabel("Embedding dim")
    ax2.set_ylabel("Token (clustered)")
    
    # PCA - handle 1D case
    ax3 = axes[2]
    if n_embd >= 2:
        _, _, Vt = np.linalg.svd(X_emb, full_matrices=False)
        X2 = X_emb @ Vt[:2].T
        sc = ax3.scatter(X2[:, 0], X2[:, 1], c=clusters, s=45, alpha=0.85, cmap="tab10")
        ax3.set_title(f"PCA 2D (vocab={vocab_size})", fontsize=11)
        ax3.set_xlabel("PC1")
        ax3.set_ylabel("PC2")
        ax3.grid(True, alpha=0.2)
        if vocab_size <= 80:
            for i in range(vocab_size):
                ax3.text(X2[i, 0], X2[i, 1], itos[i], fontsize=8)
        plt.colorbar(sc, ax=ax3, label="Cluster ID")
    else:
        # For 1D embeddings, just plot the single dimension
        X1 = X_emb[:, 0]
        sc = ax3.scatter(X1, np.zeros_like(X1), c=clusters, s=45, alpha=0.85, cmap="tab10")
        ax3.set_title(f"Embeddings 1D (vocab={vocab_size})", fontsize=11)
        ax3.set_xlabel("Embedding value")
        ax3.set_ylabel("")
        ax3.grid(True, alpha=0.2)
        if vocab_size <= 80:
            for i in range(vocab_size):
                ax3.text(X1[i], 0, itos[i], fontsize=8, ha='center')
        plt.colorbar(sc, ax=ax3, label="Cluster ID")
        ax3.set_yticks([])
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
    model.train()

@torch.no_grad()
def plot_weights_qkv(model, X, itos, save_path=None, sequence_str=None):
    """
    Plot weights and QKV activations:
    Row 1: W_Q, W_K, QK^T, Attention, W_V
    Row 2: Q, K, softmax(QK^T), Output, V
    """
    model.eval()
    B, T = X.shape
    
    # Get actual tokens for tick labels
    tokens = [itos[i.item()] for i in X[0]]
    
    # Get weights - average across heads
    Wq_all, Wk_all, Wv_all = [], [], []
    for h in model.sa_heads.heads:
        Wq_all.append(h.query.weight.cpu().numpy())  # (hs, C)
        Wk_all.append(h.key.weight.cpu().numpy())    # (hs, C)
        Wv_all.append(h.value.weight.cpu().numpy())  # (hs, C)
    
    # Average weights across heads
    W_Q = np.stack(Wq_all, axis=0).mean(axis=0).T  # (C, hs) averaged
    W_K = np.stack(Wk_all, axis=0).mean(axis=0).T  # (C, hs) averaged
    W_V = np.stack(Wv_all, axis=0).mean(axis=0).T  # (C, hs) averaged
    
    # Get activations Q, K, V from input X
    token_emb = model.token_embedding(X)
    pos = torch.arange(T, device=X.device) % model.block_size
    pos_emb = model.position_embedding_table(pos)
    x = token_emb + pos_emb
    
    q_all, k_all, v_all = [], [], []
    for h in model.sa_heads.heads:
        q_all.append(h.query(x)[0].cpu().numpy())  # (T, hs)
        k_all.append(h.key(x)[0].cpu().numpy())    # (T, hs)
        v_all.append(h.value(x)[0].cpu().numpy())  # (T, hs)
    
    # Average activations across heads
    Q = np.stack(q_all, axis=0).mean(axis=0)  # (T, hs)
    K = np.stack(k_all, axis=0).mean(axis=0)  # (T, hs)
    V = np.stack(v_all, axis=0).mean(axis=0)  # (T, hs)
    
    # Compute QK^T (attention formula: Q @ K^T)
    QK_T = Q @ K.T  # (T, hs) @ (hs, T) = (T, T)
    
    # Compute softmax(QK^T) without masking
    QK_T_torch = torch.from_numpy(QK_T).float()
    Softmax_QK_T = F.softmax(QK_T_torch, dim=-1).numpy()  # (T, T)
    
    # Get attention weights (softmax(QK^T) after masking) from all heads
    wei_all = []
    out_all = []
    for h in model.sa_heads.heads:
        out, wei = h(x)  # out: (B, T, head_size), wei: (B, T, T)
        wei_all.append(wei[0].cpu().numpy())  # (T, T)
        out_all.append(out[0].cpu().numpy())  # (T, head_size)
    
    # Average attention weights across heads
    Attention = np.stack(wei_all, axis=0).mean(axis=0)  # (T, T)
    
    # Average output across heads
    Output = np.stack(out_all, axis=0).mean(axis=0)  # (T, head_size)
    
    # Create figure: 2 rows, 5 columns
    # Column order: W_Q/Q, W_K/K, QK^T/softmax(QK^T), Attention/Output, W_V/V (switched last two)
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    
    # Column 1: W_Q (top), Q (bottom)
    # Column 2: W_K (top), K (bottom)
    # Column 3: QK^T (top), softmax(QK^T) (bottom)
    # Column 4: Attention weights (top), Output matrix (bottom)
    # Column 5: W_V (top), V (bottom) - RIGHTMOST
    
    # Row 1: W_Q, W_K, QK^T, W_V, Attention (switched last two columns)
    row1_titles = ["W_Q", "W_K", "QK^T", "W_V", "Attention"]
    row1_data = [W_Q, W_K, QK_T, W_V, Attention]
    
    for j, (data, title) in enumerate(zip(row1_data, row1_titles)):
        ax = axes[0, j]
        if title in ["QK^T", "Attention"]:
            # QK^T and Attention are (T, T) matrices - use tokens for both axes
            x_labels = tokens
            y_labels_local = tokens
            dim_str = f"(T×T={data.shape[0]}×{data.shape[1]})"
            
            cmap = "magma" if title == "Attention" else "viridis"
            vmin, vmax = (0.0, 1.0) if title == "Attention" else (None, None)
            
            sns.heatmap(data, cmap=cmap, vmin=vmin, vmax=vmax, 
                        xticklabels=x_labels, yticklabels=y_labels_local, cbar=True, ax=ax)
            ax.set_title(f"{title} {dim_str}", fontsize=11)
            ax.set_ylabel("T", fontsize=10)
            ax.set_xlabel("T", fontsize=10)
        else:
            # W_Q, W_K, W_V are (C, hs) matrices - keep numeric labels
            x_labels = list(range(data.shape[1]))
            y_labels_local = list(range(data.shape[0]))
            dim_str = f"(C×hs={data.shape[0]}×{data.shape[1]})"
            
            sns.heatmap(data, cmap="viridis", xticklabels=x_labels, yticklabels=y_labels_local, cbar=True, ax=ax)
            ax.set_title(f"{title} {dim_str}", fontsize=11)
            if j == 0:
                ax.set_ylabel("C", fontsize=10)
            ax.set_xlabel("hs", fontsize=10)
    
    # Row 2: Q, K, softmax(QK^T), V, Output (switched last two columns)
    row2_titles = ["Q", "K", "softmax(QK^T)", "V", "Output"]
    row2_data = [Q, K, Softmax_QK_T, V, Output]
    
    for j, (data, title) in enumerate(zip(row2_data, row2_titles)):
        ax = axes[1, j]
        if title == "softmax(QK^T)":
            # softmax(QK^T) is (T, T) matrix - use tokens for both axes
            x_labels = tokens
            y_labels_local = tokens
            dim_str = f"(T×T={data.shape[0]}×{data.shape[1]})"
            
            sns.heatmap(data, cmap="magma", vmin=0.0, vmax=1.0,
                        xticklabels=x_labels, yticklabels=y_labels_local, cbar=True, ax=ax)
            ax.set_title(f"{title} {dim_str}", fontsize=11)
            if j == 0:
                ax.set_ylabel("T", fontsize=10)
            ax.set_xlabel("T", fontsize=10)
        elif title == "Output":
            # Output is (T, head_size) matrix - use tokens for y-axis
            x_labels = list(range(data.shape[1]))
            y_labels_local = tokens
            dim_str = f"(T×hs={data.shape[0]}×{data.shape[1]})"
            
            sns.heatmap(data, cmap="viridis", xticklabels=x_labels, yticklabels=y_labels_local, cbar=True, ax=ax)
            ax.set_title(f"{title} {dim_str}", fontsize=11)
            if j == 0:
                ax.set_ylabel("T", fontsize=10)
            ax.set_xlabel("hs", fontsize=10)
        else:
            # Q, K, V are (T, hs) matrices - use tokens for y-axis
            x_labels = list(range(data.shape[1]))
            y_labels_local = tokens
            dim_str = f"(T×hs={data.shape[0]}×{data.shape[1]})"
            
            sns.heatmap(data, cmap="viridis", xticklabels=x_labels, yticklabels=y_labels_local, cbar=True, ax=ax)
            ax.set_title(f"{title} {dim_str}", fontsize=11)
            if j == 0:
                ax.set_ylabel("T", fontsize=10)
            ax.set_xlabel("hs", fontsize=10)
    
    # Add sequence as supertitle
    if sequence_str:
        fig.suptitle(sequence_str, fontsize=14, y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])  # Leave space for supertitle
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()
    model.train()

@torch.no_grad()
def plot_weights_qkv_two_sequences(model, X_list, itos, save_path=None, num_sequences=3):
    """
    Plot QKV for multiple sequences shown as different rows.
    Weights (W_Q, W_K, W_V) are shown once at the top since they're shared model parameters.
    Activations (Q, K, V, etc.) are shown for each sequence.
    
    Args:
        model: The model
        X_list: List of input sequences, or single sequence (will be converted to list)
        itos: Index to string mapping
        save_path: Path to save figure
        num_sequences: Number of sequences to show (if X_list is single sequence, will use it multiple times)
    """
    model.eval()
    
    # Handle single sequence input
    if not isinstance(X_list, list):
        X_list = [X_list]
    
    # Use provided sequences up to num_sequences
    sequences_to_plot = X_list[:num_sequences]
    num_sequences = len(sequences_to_plot)
    
    # Get weights once (they're the same for all sequences)
    Wq_all, Wk_all, Wv_all = [], [], []
    for h in model.sa_heads.heads:
        Wq_all.append(h.query.weight.cpu().numpy())  # (hs, C)
        Wk_all.append(h.key.weight.cpu().numpy())    # (hs, C)
        Wv_all.append(h.value.weight.cpu().numpy())  # (hs, C)
    
    # Average weights across heads
    W_Q = np.stack(Wq_all, axis=0).mean(axis=0).T  # (C, hs) averaged
    W_K = np.stack(Wk_all, axis=0).mean(axis=0).T  # (C, hs) averaged
    W_V = np.stack(Wv_all, axis=0).mean(axis=0).T  # (C, hs) averaged
    
    # Create figure: 1 row for weights + (num_sequences * 2) rows for activations, 5 columns
    # Total: (1 + num_sequences * 2) rows, 5 columns
    fig = plt.figure(figsize=(25, 3 + 5 * num_sequences))
    gs = GridSpec(1 + num_sequences * 2, 5, figure=fig, hspace=0.3, wspace=0.3)
    
    # Row 0: Show weights once (W_Q, W_K, W_V, empty, empty)
    for j, (data, title) in enumerate(zip([W_Q, W_K, W_V], ["W_Q", "W_K", "W_V"])):
        ax = fig.add_subplot(gs[0, j])
        x_labels = list(range(data.shape[1]))
        y_labels_local = list(range(data.shape[0]))
        dim_str = f"(C×hs={data.shape[0]}×{data.shape[1]})"
        
        sns.heatmap(data, cmap="viridis", xticklabels=x_labels, yticklabels=y_labels_local, cbar=True, ax=ax)
        ax.set_title(f"{title} {dim_str}", fontsize=11)
        if j == 0:
            ax.set_ylabel("C", fontsize=10)
        ax.set_xlabel("hs", fontsize=10)
    
    # Leave columns 3 and 4 empty in the weights row
    for j in [3, 4]:
        ax = fig.add_subplot(gs[0, j])
        ax.axis('off')
    
    # For each sequence, show activations
    for seq_idx, X in enumerate(sequences_to_plot):
        # Get sequence string
        seq_str = " ".join([itos[i.item()] for i in X[0]])
        
        # Calculate row offset for this sequence (1 row for weights + seq_idx * 2 rows for previous sequences)
        row_offset = 1 + seq_idx * 2
        
        # Plot this sequence's activations
        plot_weights_qkv_single_rows(model, X, itos, fig, gs, row_offset, seq_str, seq_idx, show_weights=False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()
    model.train()

@torch.no_grad()
def plot_weights_qkv_single_rows(model, X, itos, fig, gs, row_offset, sequence_str=None, seq_idx=0, show_weights=True):
    """
    Plot a single QKV figure in a subplot grid, using row_offset to position it.
    row_offset: which row to start at (1, 3, 5, etc. for multiple sequences, since row 0 is weights)
    show_weights: if False, don't show weights (they're shown once at the top)
    """
    B, T = X.shape
    
    # Get actual tokens for tick labels
    tokens = [itos[i.item()] for i in X[0]]
    
    # Get activations Q, K, V from input X
    token_emb = model.token_embedding(X)
    pos = torch.arange(T, device=X.device) % model.block_size
    pos_emb = model.position_embedding_table(pos)
    x = token_emb + pos_emb
    
    q_all, k_all, v_all = [], [], []
    for h in model.sa_heads.heads:
        q_all.append(h.query(x)[0].cpu().numpy())  # (T, hs)
        k_all.append(h.key(x)[0].cpu().numpy())    # (T, hs)
        v_all.append(h.value(x)[0].cpu().numpy())  # (T, hs)
    
    # Average activations across heads
    Q = np.stack(q_all, axis=0).mean(axis=0)  # (T, hs)
    K = np.stack(k_all, axis=0).mean(axis=0)  # (T, hs)
    V = np.stack(v_all, axis=0).mean(axis=0)  # (T, hs)
    
    # Compute QK^T (attention formula: Q @ K^T)
    QK_T = Q @ K.T  # (T, hs) @ (hs, T) = (T, T)
    
    # Compute softmax(QK^T) without masking
    QK_T_torch = torch.from_numpy(QK_T).float()
    Softmax_QK_T = F.softmax(QK_T_torch, dim=-1).numpy()  # (T, T)
    
    # Get attention weights (softmax(QK^T) after masking) from all heads
    wei_all = []
    out_all = []
    for h in model.sa_heads.heads:
        out, wei = h(x)  # out: (B, T, head_size), wei: (B, T, T)
        wei_all.append(wei[0].cpu().numpy())  # (T, T)
        out_all.append(out[0].cpu().numpy())  # (T, head_size)
    
    # Average attention weights across heads
    Attention = np.stack(wei_all, axis=0).mean(axis=0)  # (T, T)
    
    # Average output across heads
    Output = np.stack(out_all, axis=0).mean(axis=0)  # (T, head_size)
    
    if show_weights:
        # Row 1: W_Q, W_K, QK^T, W_V, Attention (old behavior for backward compatibility)
        # Get weights - average across heads
        Wq_all, Wk_all, Wv_all = [], [], []
        for h in model.sa_heads.heads:
            Wq_all.append(h.query.weight.cpu().numpy())  # (hs, C)
            Wk_all.append(h.key.weight.cpu().numpy())    # (hs, C)
            Wv_all.append(h.value.weight.cpu().numpy())  # (hs, C)
        
        W_Q = np.stack(Wq_all, axis=0).mean(axis=0).T  # (C, hs) averaged
        W_K = np.stack(Wk_all, axis=0).mean(axis=0).T  # (C, hs) averaged
        W_V = np.stack(Wv_all, axis=0).mean(axis=0).T  # (C, hs) averaged
        
        row1_titles = ["W_Q", "W_K", "QK^T", "W_V", "Attention"]
        row1_data = [W_Q, W_K, QK_T, W_V, Attention]
        
        for j, (data, title) in enumerate(zip(row1_data, row1_titles)):
            ax = fig.add_subplot(gs[row_offset, j])
            if title in ["QK^T", "Attention"]:
                # QK^T and Attention are (T, T) matrices - use tokens for both axes
                x_labels = tokens
                y_labels_local = tokens
                dim_str = f"(T×T={data.shape[0]}×{data.shape[1]})"
                
                cmap = "magma" if title == "Attention" else "viridis"
                vmin, vmax = (0.0, 1.0) if title == "Attention" else (None, None)
                
                sns.heatmap(data, cmap=cmap, vmin=vmin, vmax=vmax, 
                            xticklabels=x_labels, yticklabels=y_labels_local, cbar=True, ax=ax)
                ax.set_title(f"{title} {dim_str}", fontsize=11)
                ax.set_ylabel("T", fontsize=10)
                ax.set_xlabel("T", fontsize=10)
            else:
                # W_Q, W_K, W_V are (C, hs) matrices - keep numeric labels
                x_labels = list(range(data.shape[1]))
                y_labels_local = list(range(data.shape[0]))
                dim_str = f"(C×hs={data.shape[0]}×{data.shape[1]})"
                
                sns.heatmap(data, cmap="viridis", xticklabels=x_labels, yticklabels=y_labels_local, cbar=True, ax=ax)
                ax.set_title(f"{title} {dim_str}", fontsize=11)
                if j == 0:
                    if sequence_str:
                        ax.set_ylabel(f"Seq {seq_idx+1}: {sequence_str}\nC", fontsize=9)
                    else:
                        ax.set_ylabel("C", fontsize=10)
                ax.set_xlabel("hs", fontsize=10)
    else:
        # Row 1: Q, K, QK^T, V, Attention (no weights)
        row1_titles = ["Q", "K", "QK^T", "V", "Attention"]
        row1_data = [Q, K, QK_T, V, Attention]
        
        for j, (data, title) in enumerate(zip(row1_data, row1_titles)):
            ax = fig.add_subplot(gs[row_offset, j])
            if title in ["QK^T", "Attention"]:
                # QK^T and Attention are (T, T) matrices - use tokens for both axes
                x_labels = tokens
                y_labels_local = tokens
                dim_str = f"(T×T={data.shape[0]}×{data.shape[1]})"
                
                cmap = "magma" if title == "Attention" else "viridis"
                vmin, vmax = (0.0, 1.0) if title == "Attention" else (None, None)
                
                sns.heatmap(data, cmap=cmap, vmin=vmin, vmax=vmax, 
                            xticklabels=x_labels, yticklabels=y_labels_local, cbar=True, ax=ax)
                ax.set_title(f"{title} {dim_str}", fontsize=11)
                if j == 0:
                    if sequence_str:
                        ax.set_ylabel(f"Seq {seq_idx+1}: {sequence_str}\nT", fontsize=9)
                    else:
                        ax.set_ylabel("T", fontsize=10)
                ax.set_xlabel("T", fontsize=10)
            else:
                # Q, K, V are (T, hs) matrices - use tokens for y-axis
                x_labels = list(range(data.shape[1]))
                y_labels_local = tokens
                dim_str = f"(T×hs={data.shape[0]}×{data.shape[1]})"
                
                sns.heatmap(data, cmap="viridis", xticklabels=x_labels, yticklabels=y_labels_local, cbar=True, ax=ax)
                ax.set_title(f"{title} {dim_str}", fontsize=11)
                if j == 0:
                    if sequence_str:
                        ax.set_ylabel(f"Seq {seq_idx+1}: {sequence_str}\nT", fontsize=9)
                    else:
                        ax.set_ylabel("T", fontsize=10)
                ax.set_xlabel("hs", fontsize=10)
    
    # Row 2: Q, K, softmax(QK^T), V, Output
    row2_titles = ["Q", "K", "softmax(QK^T)", "V", "Output"]
    row2_data = [Q, K, Softmax_QK_T, V, Output]
    
    for j, (data, title) in enumerate(zip(row2_data, row2_titles)):
        ax = fig.add_subplot(gs[row_offset + 1, j])
        if title == "softmax(QK^T)":
            # softmax(QK^T) is (T, T) matrix - use tokens for both axes
            x_labels = tokens
            y_labels_local = tokens
            dim_str = f"(T×T={data.shape[0]}×{data.shape[1]})"
            
            sns.heatmap(data, cmap="magma", vmin=0.0, vmax=1.0,
                        xticklabels=x_labels, yticklabels=y_labels_local, cbar=True, ax=ax)
            ax.set_title(f"{title} {dim_str}", fontsize=11)
            if j == 0:
                ax.set_ylabel("T", fontsize=10)
            ax.set_xlabel("T", fontsize=10)
        elif title == "Output":
            # Output is (T, head_size) matrix - use tokens for y-axis
            x_labels = list(range(data.shape[1]))
            y_labels_local = tokens
            dim_str = f"(T×hs={data.shape[0]}×{data.shape[1]})"
            
            sns.heatmap(data, cmap="viridis", xticklabels=x_labels, yticklabels=y_labels_local, cbar=True, ax=ax)
            ax.set_title(f"{title} {dim_str}", fontsize=11)
            if j == 0:
                ax.set_ylabel("T", fontsize=10)
            ax.set_xlabel("hs", fontsize=10)
        else:
            # Q, K, V are (T, hs) matrices - use tokens for y-axis
            x_labels = list(range(data.shape[1]))
            y_labels_local = tokens
            dim_str = f"(T×hs={data.shape[0]}×{data.shape[1]})"
            
            sns.heatmap(data, cmap="viridis", xticklabels=x_labels, yticklabels=y_labels_local, cbar=True, ax=ax)
            ax.set_title(f"{title} {dim_str}", fontsize=11)
            if j == 0:
                ax.set_ylabel("T", fontsize=10)
            ax.set_xlabel("hs", fontsize=10)

@torch.no_grad()
def plot_weights_qkv_single(model, X, itos, fig, col_offset, sequence_str=None):
    """
    Plot a single QKV figure in a subplot grid
    col_offset: 0 for left, 1 for right (each sequence takes 5 columns)
    """
    B, T = X.shape
    
    # Get actual tokens for tick labels
    tokens = [itos[i.item()] for i in X[0]]
    
    # Get weights - average across heads
    Wq_all, Wk_all, Wv_all = [], [], []
    for h in model.sa_heads.heads:
        Wq_all.append(h.query.weight.cpu().numpy())  # (hs, C)
        Wk_all.append(h.key.weight.cpu().numpy())    # (hs, C)
        Wv_all.append(h.value.weight.cpu().numpy())  # (hs, C)
    
    # Average weights across heads
    W_Q = np.stack(Wq_all, axis=0).mean(axis=0).T  # (C, hs) averaged
    W_K = np.stack(Wk_all, axis=0).mean(axis=0).T  # (C, hs) averaged
    W_V = np.stack(Wv_all, axis=0).mean(axis=0).T  # (C, hs) averaged
    
    # Get activations Q, K, V from input X
    token_emb = model.token_embedding(X)
    pos = torch.arange(T, device=X.device) % model.block_size
    pos_emb = model.position_embedding_table(pos)
    x = token_emb + pos_emb
    
    q_all, k_all, v_all = [], [], []
    for h in model.sa_heads.heads:
        q_all.append(h.query(x)[0].cpu().numpy())  # (T, hs)
        k_all.append(h.key(x)[0].cpu().numpy())    # (T, hs)
        v_all.append(h.value(x)[0].cpu().numpy())  # (T, hs)
    
    # Average activations across heads
    Q = np.stack(q_all, axis=0).mean(axis=0)  # (T, hs)
    K = np.stack(k_all, axis=0).mean(axis=0)  # (T, hs)
    V = np.stack(v_all, axis=0).mean(axis=0)  # (T, hs)
    
    # Compute QK^T (attention formula: Q @ K^T)
    QK_T = Q @ K.T  # (T, hs) @ (hs, T) = (T, T)
    
    # Compute softmax(QK^T) without masking
    QK_T_torch = torch.from_numpy(QK_T).float()
    Softmax_QK_T = F.softmax(QK_T_torch, dim=-1).numpy()  # (T, T)
    
    # Get attention weights (softmax(QK^T) after masking) from all heads
    wei_all = []
    out_all = []
    for h in model.sa_heads.heads:
        out, wei = h(x)  # out: (B, T, head_size), wei: (B, T, T)
        wei_all.append(wei[0].cpu().numpy())  # (T, T)
        out_all.append(out[0].cpu().numpy())  # (T, head_size)
    
    # Average attention weights across heads
    Attention = np.stack(wei_all, axis=0).mean(axis=0)  # (T, T)
    
    # Average output across heads
    Output = np.stack(out_all, axis=0).mean(axis=0)  # (T, head_size)
    
    # Create axes for this sequence (2 rows, 5 columns, starting at col_offset*5)
    gs = GridSpec(2, 10, figure=fig, hspace=0.3, wspace=0.3)
    
    # Row 1: W_Q, W_K, QK^T, W_V, Attention (switched last two columns)
    row1_titles = ["W_Q", "W_K", "QK^T", "W_V", "Attention"]
    row1_data = [W_Q, W_K, QK_T, W_V, Attention]
    
    for j, (data, title) in enumerate(zip(row1_data, row1_titles)):
        ax = fig.add_subplot(gs[0, col_offset*5 + j])
        if title in ["QK^T", "Attention"]:
            # QK^T and Attention are (T, T) matrices - use tokens for both axes
            x_labels = tokens
            y_labels_local = tokens
            dim_str = f"(T×T={data.shape[0]}×{data.shape[1]})"
            
            cmap = "magma" if title == "Attention" else "viridis"
            vmin, vmax = (0.0, 1.0) if title == "Attention" else (None, None)
            
            sns.heatmap(data, cmap=cmap, vmin=vmin, vmax=vmax, 
                        xticklabels=x_labels, yticklabels=y_labels_local, cbar=True, ax=ax)
            ax.set_title(f"{title} {dim_str}", fontsize=11)
            ax.set_ylabel("T", fontsize=10)
            ax.set_xlabel("T", fontsize=10)
        else:
            # W_Q, W_K, W_V are (C, hs) matrices - keep numeric labels
            x_labels = list(range(data.shape[1]))
            y_labels_local = list(range(data.shape[0]))
            dim_str = f"(C×hs={data.shape[0]}×{data.shape[1]})"
            
            sns.heatmap(data, cmap="viridis", xticklabels=x_labels, yticklabels=y_labels_local, cbar=True, ax=ax)
            ax.set_title(f"{title} {dim_str}", fontsize=11)
            if j == 0:
                ax.set_ylabel("C", fontsize=10)
            ax.set_xlabel("hs", fontsize=10)
    
    # Row 2: Q, K, softmax(QK^T), V, Output (switched last two columns)
    row2_titles = ["Q", "K", "softmax(QK^T)", "V", "Output"]
    row2_data = [Q, K, Softmax_QK_T, V, Output]
    
    for j, (data, title) in enumerate(zip(row2_data, row2_titles)):
        ax = fig.add_subplot(gs[1, col_offset*5 + j])
        if title == "softmax(QK^T)":
            # softmax(QK^T) is (T, T) matrix - use tokens for both axes
            x_labels = tokens
            y_labels_local = tokens
            dim_str = f"(T×T={data.shape[0]}×{data.shape[1]})"
            
            sns.heatmap(data, cmap="magma", vmin=0.0, vmax=1.0,
                        xticklabels=x_labels, yticklabels=y_labels_local, cbar=True, ax=ax)
            ax.set_title(f"{title} {dim_str}", fontsize=11)
            if j == 0:
                ax.set_ylabel("T", fontsize=10)
            ax.set_xlabel("T", fontsize=10)
        elif title == "Output":
            # Output is (T, head_size) matrix - use tokens for y-axis
            x_labels = list(range(data.shape[1]))
            y_labels_local = tokens
            dim_str = f"(T×hs={data.shape[0]}×{data.shape[1]})"
            
            sns.heatmap(data, cmap="viridis", xticklabels=x_labels, yticklabels=y_labels_local, cbar=True, ax=ax)
            ax.set_title(f"{title} {dim_str}", fontsize=11)
            if j == 0:
                ax.set_ylabel("T", fontsize=10)
            ax.set_xlabel("hs", fontsize=10)
        else:
            # Q, K, V are (T, hs) matrices - use tokens for y-axis
            x_labels = list(range(data.shape[1]))
            y_labels_local = tokens
            dim_str = f"(T×hs={data.shape[0]}×{data.shape[1]})"
            
            sns.heatmap(data, cmap="viridis", xticklabels=x_labels, yticklabels=y_labels_local, cbar=True, ax=ax)
            ax.set_title(f"{title} {dim_str}", fontsize=11)
            if j == 0:
                ax.set_ylabel("T", fontsize=10)
            ax.set_xlabel("hs", fontsize=10)
    
    # Add sequence as title for this column block
    if sequence_str:
        # Add text at the top of this column block (centered over the 5 columns)
        center_x = 0.1 + col_offset*0.5 + 0.2  # Center of the 5-column block
        fig.text(center_x, 0.98, sequence_str, 
                fontsize=12, ha='center', va='top', weight='bold')

@torch.no_grad()
def plot_attention_matrix(model, X_list, itos, save_path=None, num_sequences=3):
    """
    Plot attention weights matrix (averaged across heads) for multiple sequences,
    and show linear/logit/softmax layers after attention.
    
    Args:
        model: The model
        X_list: List of input sequences, or single sequence (will be converted to list)
        itos: Index to string mapping
        save_path: Path to save figure
        num_sequences: Number of sequences to show (if X_list is single sequence, will generate more)
    """
    model.eval()
    
    # Handle single sequence input
    if not isinstance(X_list, list):
        X_list = [X_list]
    
    # If only one sequence provided, we'll use it multiple times (or could generate more)
    # For now, just use what's provided up to num_sequences
    sequences_to_plot = X_list[:num_sequences]
    num_sequences = len(sequences_to_plot)
    
    # Collect data for all sequences
    attention_matrices = []
    linear_outputs = []
    logits_list = []
    probs_list = []
    tokens_list = []
    
    for X in sequences_to_plot:
        B, T = X.shape
        
        # Get tokens for this sequence
        tokens = [itos[i.item()] for i in X[0]]
        tokens_list.append(tokens)
        
        # Get embeddings and positional encodings
        token_emb = model.token_embedding(X)
        pos = torch.arange(T, device=X.device) % model.block_size
        pos_emb = model.position_embedding_table(pos)
        x = token_emb + pos_emb
        
        # Get attention weights from all heads
        wei_all = []
        for h in model.sa_heads.heads:
            _, wei = h(x)  # wei: (B, T, T)
            wei_all.append(wei[0].cpu().numpy())  # (T, T)
        
        # Average across heads
        attention_matrix = np.stack(wei_all, axis=0).mean(axis=0)  # (T, T)
        attention_matrices.append(attention_matrix)
        
        # Get attention output (before linear layer)
        out_all = []
        for h in model.sa_heads.heads:
            out, _ = h(x)  # out: (B, T, head_size)
            out_all.append(out[0].cpu().numpy())  # (T, head_size)
        
        # Concatenate heads and average (or just concatenate)
        attention_out = np.concatenate(out_all, axis=-1)  # (T, num_heads * head_size)
        linear_outputs.append(attention_out)
        
        # Get logits (after linear layer)
        logits, _ = model(X)
        logits_np = logits[0].cpu().numpy()  # (T, vocab_size)
        logits_list.append(logits_np)
        
        # Get probabilities (softmax of logits)
        probs = F.softmax(logits[0], dim=-1).cpu().numpy()  # (T, vocab_size)
        probs_list.append(probs)
    
    # Create figure: num_sequences rows, 4 columns (Attention, Linear Out, Logits, Probs)
    fig, axes = plt.subplots(num_sequences, 4, figsize=(20, 5 * num_sequences))
    if num_sequences == 1:
        axes = axes.reshape(1, -1)
    
    for i, (attn, lin_out, logits_np, probs_np, tokens) in enumerate(
        zip(attention_matrices, linear_outputs, logits_list, probs_list, tokens_list)
    ):
        T = attn.shape[0]
        
        # Column 1: Attention matrix
        ax = axes[i, 0]
        sns.heatmap(attn, cmap="magma", vmin=0.0, vmax=1.0, 
                    xticklabels=tokens, yticklabels=tokens, cbar=True, ax=ax)
        ax.set_title(f"Attention (T×T={T}×{T})", fontsize=11)
        ax.set_xlabel("Key position")
        ax.set_ylabel("Query position")
        
        # Column 2: Linear output (attention output before lm_head)
        ax = axes[i, 1]
        sns.heatmap(lin_out, cmap="viridis", 
                    xticklabels=list(range(lin_out.shape[1])), 
                    yticklabels=tokens, cbar=True, ax=ax)
        ax.set_title(f"Linear Input (T×{lin_out.shape[1]})", fontsize=11)
        ax.set_xlabel("Head dim")
        ax.set_ylabel("Position")
        
        # Column 3: Logits
        ax = axes[i, 2]
        sns.heatmap(logits_np, cmap="RdBu_r", center=0,
                    xticklabels=[itos[j] for j in range(logits_np.shape[1])], 
                    yticklabels=tokens, cbar=True, ax=ax)
        ax.set_title(f"Logits (T×vocab={T}×{logits_np.shape[1]})", fontsize=11)
        ax.set_xlabel("Vocabulary")
        ax.set_ylabel("Position")
        
        # Column 4: Probabilities (softmax)
        ax = axes[i, 3]
        sns.heatmap(probs_np, cmap="magma", vmin=0.0, vmax=1.0,
                    xticklabels=[itos[j] for j in range(probs_np.shape[1])], 
                    yticklabels=tokens, cbar=True, ax=ax)
        ax.set_title(f"Probabilities (T×vocab={T}×{probs_np.shape[1]})", fontsize=11)
        ax.set_xlabel("Vocabulary")
        ax.set_ylabel("Position")
        
        # Add sequence string as ylabel on leftmost plot
        if i == 0:
            seq_str = " ".join(tokens)
            axes[i, 0].set_ylabel(f"Seq {i+1}: {seq_str}\nQuery position", fontsize=10)
        else:
            seq_str = " ".join(tokens)
            axes[i, 0].set_ylabel(f"Seq {i+1}: {seq_str}\nQuery position", fontsize=10)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()
    model.train()

@torch.no_grad()
def plot_position_embeddings(model, X_list, itos, save_path=None, num_sequences=3):
    """
    Plot token embeddings, positional embeddings, their sum, and PCAs for multiple sequences
    Each sequence gets 2 rows: Row 1: Heatmaps (token, position, combined), Row 2: PCAs (token, position, combined)
    
    Args:
        model: The model
        X_list: List of input sequences, or single sequence (will be converted to list)
        itos: Index to string mapping
        save_path: Path to save figure
        num_sequences: Number of sequences to show
    """
    model.eval()
    
    # Handle single sequence input
    if not isinstance(X_list, list):
        X_list = [X_list]
    
    # Use provided sequences up to num_sequences
    sequences_to_plot = X_list[:num_sequences]
    num_sequences = len(sequences_to_plot)
    
    # Get embedding dimension from the model
    n_embd = model.position_embedding_table.weight.shape[1]
    
    # Create figure: (num_sequences * 2) rows, 3 columns
    fig, axes = plt.subplots(num_sequences * 2, 3, figsize=(20, 6 * num_sequences))
    if num_sequences == 1:
        axes = axes.reshape(2, -1)
    
    for seq_idx, X in enumerate(sequences_to_plot):
        B, T = X.shape
        
        # Get position indices (0, 1, 2, ..., T-1, wrapped by block_size)
        positions = torch.arange(T, device=X.device) % model.block_size
        positions_np = positions.cpu().numpy()  # (T,)
        
        # Get positional embeddings
        pos_emb = model.position_embedding_table(positions)  # (T, n_embd)
        pos_emb_np = pos_emb.detach().cpu().numpy()  # (T, n_embd)
        
        # Get token embeddings
        token_emb = model.token_embedding(X)  # (B, T, n_embd)
        token_emb_np = token_emb[0].detach().cpu().numpy()  # (T, n_embd)
        
        # Get sum (token + position)
        combined_emb = token_emb + pos_emb  # (B, T, n_embd)
        combined_emb_np = combined_emb[0].detach().cpu().numpy()  # (T, n_embd)
        
        # Get tokens for labels
        tokens = [itos[i.item()] for i in X[0]]
        
        # Calculate row indices for this sequence
        row_heatmap = seq_idx * 2
        row_pca = seq_idx * 2 + 1
        
        # Get sequence string for title
        seq_str = " ".join(tokens[:20])  # First 20 tokens
        if len(tokens) > 20:
            seq_str += "..."
        
        # Row 1: Heatmaps
        # Token embeddings heatmap
        ax1 = axes[row_heatmap, 0]
        sns.heatmap(token_emb_np, cmap="RdBu_r", center=0, xticklabels=list(range(n_embd)),
                    yticklabels=tokens, cbar=True, ax=ax1)
        ax1.set_title(f"Token Embeddings (T×n_embd={T}×{n_embd}) - Seq {seq_idx+1}", fontsize=11)
        ax1.set_xlabel("Embedding dim")
        ax1.set_ylabel("Position (token)")
        
        # Positional embeddings heatmap
        ax2 = axes[row_heatmap, 1]
        sns.heatmap(pos_emb_np, cmap="RdBu_r", center=0, xticklabels=list(range(n_embd)),
                    yticklabels=tokens, cbar=True, ax=ax2)
        ax2.set_title(f"Positional Embeddings (T×n_embd={T}×{n_embd}) - Seq {seq_idx+1}", fontsize=11)
        ax2.set_xlabel("Embedding dim")
        ax2.set_ylabel("Position (token)")
        
        # Combined embeddings heatmap
        ax3 = axes[row_heatmap, 2]
        sns.heatmap(combined_emb_np, cmap="RdBu_r", center=0, xticklabels=list(range(n_embd)),
                    yticklabels=tokens, cbar=True, ax=ax3)
        ax3.set_title(f"Token + Position (T×n_embd={T}×{n_embd}) - Seq {seq_idx+1}", fontsize=11)
        ax3.set_xlabel("Embedding dim")
        ax3.set_ylabel("Position (token)")
        
        # Row 2: PCAs
        def plot_pca(ax, data, title, color_by_position=True):
            if n_embd >= 2:
                X_data = data.astype(np.float64)
                X_data = X_data - X_data.mean(axis=0, keepdims=True)
                _, _, Vt = np.linalg.svd(X_data, full_matrices=False)
                X2 = X_data @ Vt[:2].T
                colors = positions_np if color_by_position else range(T)
                sc = ax.scatter(X2[:, 0], X2[:, 1], c=colors, s=100, alpha=0.7, cmap="viridis")
                ax.set_title(title, fontsize=11)
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
                ax.grid(True, alpha=0.2)
                for i, token in enumerate(tokens):
                    ax.text(X2[i, 0], X2[i, 1], token, fontsize=8, ha='center', va='center')
                plt.colorbar(sc, ax=ax, label="Position mod k")
            else:
                X1 = data[:, 0]
                sc = ax.scatter(X1, positions_np, c=positions_np, s=100, alpha=0.7, cmap="viridis")
                ax.set_title(title, fontsize=11)
                ax.set_xlabel("Embedding value")
                ax.set_ylabel("Position index")
                ax.grid(True, alpha=0.2)
                for i, token in enumerate(tokens):
                    ax.text(X1[i], positions_np[i], token, fontsize=8, ha='center', va='bottom')
                plt.colorbar(sc, ax=ax, label="Position index")
        
        # PCA of token embeddings
        plot_pca(axes[row_pca, 0], token_emb_np, f"PCA: Token Embeddings (T={T}) - Seq {seq_idx+1}")
        
        # PCA of positional embeddings
        plot_pca(axes[row_pca, 1], pos_emb_np, f"PCA: Positional Embeddings (T={T}) - Seq {seq_idx+1}")
        
        # PCA of combined embeddings
        plot_pca(axes[row_pca, 2], combined_emb_np, f"PCA: Combined Embeddings (T={T}) - Seq {seq_idx+1}")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()
    model.train()

@torch.no_grad()
def plot_learned_lookup(model, vocab_size, itos, save_path=None):
    """
    Plot what the model predicts for each input token (the learned lookup table).
    This directly shows the V matrix's role in encoding token-to-token mappings.
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Create a sequence with each token appearing once, with context
    # We'll query position 1 for each pair (context_token, query_token)
    predictions = np.zeros((vocab_size, vocab_size))
    
    # For each possible input token, what does the model predict next?
    for token in range(vocab_size):
        # Create input: [token] and predict next
        # Use a minimal context with just the token
        x = torch.tensor([[token]], dtype=torch.long, device=device)
        logits, _ = model(x)
        probs = torch.softmax(logits[0, 0], dim=-1).cpu().numpy()
        predictions[token] = probs
    
    # Find the argmax prediction for each token
    predicted_next = np.argmax(predictions, axis=1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Probability matrix (what probabilities does model assign)
    ax1 = axes[0]
    sns.heatmap(predictions, cmap="viridis", xticklabels=[itos[i] for i in range(vocab_size)],
                yticklabels=[itos[i] for i in range(vocab_size)], cbar=True, ax=ax1,
                vmin=0, vmax=1)
    ax1.set_title("Learned Lookup Table\n(P(next_token | current_token))", fontsize=12)
    ax1.set_xlabel("Predicted Next Token")
    ax1.set_ylabel("Current Token")
    
    # Right: The argmax mapping as a simple table
    ax2 = axes[1]
    # Create a mapping visualization
    mapping_text = "Learned Mapping:\n\n"
    for i in range(vocab_size):
        pred = predicted_next[i]
        conf = predictions[i, pred] * 100
        mapping_text += f"{itos[i]} → {itos[pred]} ({conf:.1f}%)\n"
    
    ax2.text(0.1, 0.95, mapping_text, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    ax2.axis('off')
    ax2.set_title("Token Mappings (argmax)", fontsize=12)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()
    model.train()


@torch.no_grad()
def plot_output_matrix(model, X, itos, save_path=None):
    """
    Plot output matrix after attention (averaged across heads)
    """
    model.eval()
    B, T = X.shape
    
    # Get embeddings and positional encodings
    token_emb = model.token_embedding(X)
    pos = torch.arange(T, device=X.device) % model.block_size
    pos_emb = model.position_embedding_table(pos)
    x = token_emb + pos_emb
    
    # Get output from all heads
    out_all = []
    for h in model.sa_heads.heads:
        out, _ = h(x)  # out: (B, T, head_size)
        out_all.append(out[0].cpu().numpy())  # (T, head_size)
    
    # Average across heads
    output_matrix = np.stack(out_all, axis=0).mean(axis=0)  # (T, head_size)
    
    # Plot
    plt.figure(figsize=(8, 7))
    x_labels = list(range(output_matrix.shape[1]))
    y_labels = list(range(output_matrix.shape[0]))
    sns.heatmap(output_matrix, cmap="viridis", 
                xticklabels=x_labels, yticklabels=y_labels, cbar=True)
    plt.title(f"Output Matrix (T×hs={T}×{output_matrix.shape[1]})", fontsize=12)
    plt.xlabel("Head dimension (hs)")
    plt.ylabel("Time position (T)")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
    model.train()

def plot_embedding_triplet_matrix(model, X, itos, title="Embeddings: pos, token, x", save_path=None):
    model.eval()
    B, T = X.shape

    chars = [itos[i.item()] for i in X[0]]
    ytick = chars

    token_emb = model.token_embedding(X)
    pos = torch.arange(T, device=X.device) % model.block_size
    pos_emb = model.position_embedding_table(pos)
    x = token_emb + pos_emb

    tok0 = token_emb[0].detach().cpu().numpy()
    pos0 = pos_emb.detach().cpu().numpy()
    x0   = x[0].detach().cpu().numpy()

    mats = [pos0, tok0, x0]
    t_size, c_size = pos0.shape
    titles = [f"pos_emb {t_size}×{c_size}", f"token_emb {t_size}×{c_size}", f"x = token + pos {t_size}×{c_size}"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # and REMOVE fig.tight_layout(...) entirely

    # and REMOVE fig.tight_layout(...) entirely

    for j in range(3):
        ax = axes[j]
        x_labels = list(range(mats[j].shape[1]))
        sns.heatmap(
            mats[j],
            ax=ax,
            cmap="viridis",
            cbar=True,
            yticklabels=ytick,
            xticklabels=x_labels
        )
        ax.tick_params(axis="y", labelrotation=0)
        ax.tick_params(axis="x", labelrotation=0)
        ax.set_title(titles[j], fontsize=12)
        ax.set_xlabel("Embedding dim (C)")
        ax.set_ylabel("")

        # put full sequence once (only on first panel)
        if j == 0:
            annotate_sequence(ax, chars, y=1.02, fontsize=12)
            ax.set_ylabel("Token position")

    fig.suptitle(title, fontsize=16)
    # fig.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
    model.train()

# -----------------------------
# Loss evaluation helper
# -----------------------------
@torch.no_grad()
def estimate_loss(model, train_sequences, val_sequences, block_size, batch_size, eval_iterations):
    """
    Average loss on 'train' and 'validation' splits.
    """
    out = {}
    model.eval()

    for split_name, sequences in [("train", train_sequences), ("validation", val_sequences)]:
        losses = torch.zeros(eval_iterations)
        for i in range(eval_iterations):
            X, Y = get_batch_from_sequences(sequences, block_size, batch_size)
            _, loss = model(X, Y)
            losses[i] = loss.item()
        out[split_name] = losses.mean()

    model.train()
    return out

class Head(nn.Module):
        """One head of self-attention (Version 4)."""

        def __init__(self, n_embd: int, head_size: int, block_size: int):
            super().__init__()
            self.key = nn.Linear(n_embd, head_size, bias=False)
            self.query = nn.Linear(n_embd, head_size, bias=False)
            self.value = nn.Linear(n_embd, head_size, bias=False)

            # causal mask
            self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

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

    def __init__(self, num_heads, n_embd, head_size, block_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size) for _ in range(num_heads)])

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
    - token_embedding: maps token id -> embedding vector (n_embd)
    - lm_head: maps attention output -> logits over vocab
    """
    def __init__(self, vocab_size: int, n_embd: int, block_size: int, num_heads: int, head_size: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)           # (vocab, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)  # (block_size, n_embd)
        self.sa_heads = MultiHeadAttention(num_heads, n_embd, head_size, block_size)
        # lm_head takes attention output (num_heads * head_size), not n_embd
        self.lm_head = nn.Linear(num_heads * head_size, vocab_size)       # (num_heads*head_size -> vocab)
        self.block_size = block_size

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, return_wei: bool = False):
        B, T = idx.shape

        token_emb = self.token_embedding(idx)  # (B,T,C)
        positions = torch.arange(T, device=idx.device) % self.block_size
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
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)            # (B, T, vocab)
            last_logits = logits[:, -1, :]   # (B, vocab)
            probs = F.softmax(last_logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat([idx, idx_next], dim=1)             # (B, T+1)
        return idx

# -----------------------------
# Main training script
# -----------------------------
def main(config_name: str = "copy_modulo"):
    """
    Main training function.
    
    Args:
        config_name: Name of the config file (without .yaml extension) in the configs folder
    """
    torch.manual_seed(0)

    # Load configuration
    config = load_config(config_name)
    print(f"Loaded configuration: {config['name']}")
    
    # Extract config values
    config_name_actual = config['name']
    data_config = config['data']
    model_config = config['model']
    training_config = config['training']
    
    # Create plots directory with config name subfolder
    plots_dir = os.path.join("plots", config_name_actual)
    os.makedirs(plots_dir, exist_ok=True)

    # 1) Generate integer string data using generator from config
    generator = get_generator_from_config(config)
    sequences = generate_integer_string_data(
        generator, 
        num_sequences=data_config['num_sequences'],
        min_length=data_config['min_length'],
        max_length=data_config['max_length']
    )
    print(f"Generated {len(sequences)} sequences")
    print(f"Sequence lengths: min={min(len(s) for s in sequences)}, max={max(len(s) for s in sequences)}, avg={sum(len(s) for s in sequences)/len(sequences):.1f}")
    
    # 2) Build encoder/decoder for integers
    min_value = data_config['min_value']
    max_value = data_config['max_value']
    encode, decode, vocab_size, itos, stoi = build_encoder_for_integers(min_value=min_value, max_value=max_value)
    print("Vocabulary size:", vocab_size)
    print("Vocabulary (integers):", [itos[i] for i in range(vocab_size)])

    # 3) Encode sequences (integer values -> token indices)
    encoded_sequences = [encode(seq) for seq in sequences]
    
    # 4) Split sequences into train/val
    train_sequences, val_sequences = split_train_val_sequences(encoded_sequences, train_ratio=0.9)
    print(f"Train: {len(train_sequences)} sequences, Val: {len(val_sequences)} sequences")

    # 5) Create model + optimizer
    n_embd = model_config['n_embd']
    block_size = model_config['block_size']
    num_heads = model_config['num_heads']
    head_size = model_config['head_size']
    
    model = BigramLanguageModel(vocab_size, n_embd, block_size, num_heads, head_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config['learning_rate'])

    # 6) Training loop
    steps_for_plot = []
    train_loss_history = []
    val_loss_history = []

    batch_size = training_config['batch_size']
    max_steps = training_config['max_steps']
    eval_interval = training_config['eval_interval']
    eval_iterations = training_config['eval_iterations']
    
    X_fixed, _ = get_batch_from_sequences(train_sequences, block_size, batch_size)

    for step in range(max_steps):
        # Evaluate occasionally
        if step % eval_interval == 0:
            losses = estimate_loss(model, train_sequences, val_sequences, block_size, batch_size, eval_iterations)

            steps_for_plot.append(step)
            train_loss_history.append(losses["train"])
            val_loss_history.append(losses["validation"])

            print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['validation']:.4f}")

        # One batch
        X, Y = get_batch_from_sequences(train_sequences, block_size, batch_size)

        # Forward + backward + update
        _, loss = model(X, Y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # 4) Show results
    print("Final loss:", loss.item())

    # Generate multiple integer sequences
    num_sequences_to_generate = 10  # Number of sequences to generate
    generated_sequences = []
    for _ in range(num_sequences_to_generate):
        # Random sequence length between min_length and max_length
        seq_length = random.randint(data_config['min_length'], data_config['max_length'])
        # Start with a random token instead of always 0 to avoid bias
        start_token = random.randint(0, vocab_size - 1)
        start = torch.tensor([[start_token]], dtype=torch.long)
        sample = model.generate(start, max_new_tokens=seq_length - 1)[0].tolist()  # -1 because start token counts
        generated_integers = decode(sample)  # Decode token indices back to integer values
        generated_sequences.append(generated_integers)
    
    # Write sequences, one per line, with space-separated integers
    with open(os.path.join(plots_dir, "generated_integer_sequence.txt"), "w", encoding="utf-8") as f:
        for seq in generated_sequences:
            f.write(" ".join(str(i) for i in seq) + "\n")
    print(f"Generated {num_sequences_to_generate} sequences")
    print(f"First sequence (length {len(generated_sequences[0])}): {generated_sequences[0][:50]}...")

    # 5) Plots - keep only: QKV (2 rows: weights W_Q/W_K/W_V, activations Q/K/V), embeddings (1x3: raw/hierarchical/PCA), attention matrix, output matrix, learning curve
    plot_learning_curve(steps_for_plot, train_loss_history, val_loss_history, 
                       save_path=os.path.join(plots_dir, "learning_curve.png"), 
                       eval_interval=eval_interval)
    
    # Create multiple example sequences for plotting
    X1, _ = get_batch_from_sequences(train_sequences, block_size, 1)
    X2, _ = get_batch_from_sequences(train_sequences, block_size, 1)
    X3, _ = get_batch_from_sequences(train_sequences, block_size, 1)
    X_list = [X1, X2, X3]
    
    # Plot QKV for multiple sequences (shown as rows)
    plot_weights_qkv_two_sequences(model, X_list, itos, save_path=os.path.join(plots_dir, "qkv.png"), num_sequences=3)
    plot_embeddings_pca(model, itos, save_path=os.path.join(plots_dir, "embeddings.png"))
    plot_attention_matrix(model, X_list, itos, save_path=os.path.join(plots_dir, "attention_matrix.png"), num_sequences=3)
    plot_output_matrix(model, X_fixed, itos, save_path=os.path.join(plots_dir, "output_matrix.png"))
    plot_learned_lookup(model, vocab_size, itos, save_path=os.path.join(plots_dir, "learned_lookup.png"))
    plot_position_embeddings(model, X_list, itos, save_path=os.path.join(plots_dir, "position_embeddings.png"), num_sequences=3)



if __name__ == "__main__":
    # Get config name from command line argument, default to "default"
    config_name = sys.argv[1] if len(sys.argv) > 1 else "copy_modulo"
    main(config_name=config_name)
