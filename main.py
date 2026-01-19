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
from IntegerStringGenerator import IntegerStringGenerator, OddEvenIndexRule, OperatorBasedGenerator
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


def build_encoder_with_operators(min_value: int, max_value: int, operators: list[str]):
    """
    Build encoder/decoder for mixed vocabulary (integers + operators).
    
    Args:
        min_value: Minimum integer value (inclusive)
        max_value: Maximum integer value (inclusive)
        operators: List of operator strings (e.g., ["+", "-"])
        
    Returns:
        encode, decode, vocab_size, index_to_string, string_to_index
    """
    # Vocabulary: integers first, then operators
    # Integers: indices 0 to (max_value - min_value)
    # Operators: indices after integers
    num_integers = max_value - min_value + 1
    vocab_size = num_integers + len(operators)
    
    # Build mappings
    # token_to_index: maps actual token (int or str) -> index
    # index_to_token: maps index -> actual token (int or str)
    token_to_index = {}
    index_to_token = {}
    
    # Add integers
    for i in range(num_integers):
        val = min_value + i
        token_to_index[val] = i
        index_to_token[i] = val
    
    # Add operators
    for i, op in enumerate(operators):
        idx = num_integers + i
        token_to_index[op] = idx
        index_to_token[idx] = op
    
    # index_to_string for display purposes
    index_to_string = {i: str(index_to_token[i]) for i in range(vocab_size)}
    string_to_index = {str(index_to_token[i]): i for i in range(vocab_size)}
    
    def encode(tokens: list) -> list[int]:
        """
        Encode mixed tokens (integers and operators) to token indices.
        Args:
            tokens: List of tokens (can be int or str)
        Returns:
            List of token indices
        """
        return [token_to_index[t] for t in tokens]
    
    def decode(token_indices) -> list:
        """
        Decode token indices back to original tokens (int or str).
        Args:
            token_indices: List of token indices
        Returns:
            List of tokens (can be int or str)
        """
        return [index_to_token[idx] for idx in token_indices]
    
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
def plot_learning_curve(steps, train_losses, val_losses, rule_error_history=None, save_path=None, eval_interval=None):
    """
    Plot learning curve with loss and optionally rule error on the same figure.
    Uses dual y-axes: left for loss, right for error (both should decrease).
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Left y-axis: Loss
    color1 = 'tab:blue'
    ax1.set_xlabel(f"Training Steps{' (evaluated every ' + str(eval_interval) + ' steps)' if eval_interval else ''}", fontsize=11)
    ax1.set_ylabel("Cross-entropy loss", color=color1, fontsize=11)
    line1 = ax1.plot(steps, train_losses, label="Training Loss", color='tab:blue', linewidth=2)
    line2 = ax1.plot(steps, val_losses, label="Validation Loss", color='tab:orange', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    # Right y-axis: Rule Error (if provided)
    if rule_error_history is not None:
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel("Rule Error (fraction of constrained positions wrong)", color=color2, fontsize=11)
        line3 = ax2.plot(steps, rule_error_history, label="Rule Error", color=color2, linewidth=2, linestyle='--')
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.set_ylim(0, 1.05)
        
        # Combine legends
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='best')
        
        plt.title("Learning Curve: Loss and Rule Error During Training", fontsize=13)
    else:
        ax1.legend(loc='best')
        plt.title("Learning Curve: Training vs Validation Loss", fontsize=13)
    
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
    Plot embeddings: token embeddings (heatmap, clustered, PCA), position embeddings, and token+position combinations.
    """
    model.eval()
    
    # Get token embeddings
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
    
    # Get position embeddings for all positions
    block_size = model.block_size
    pos_emb_all = model.position_embedding_table.weight.detach().cpu().numpy()  # (block_size, n_embd)
    
    # Hierarchical clustering for position embeddings
    d_pos = pdist(pos_emb_all, metric="cosine")
    Z_pos = linkage(d_pos, method="average")
    row_order_pos = leaves_list(Z_pos)
    pos_emb_clustered = pos_emb_all[row_order_pos]
    pos_y_labels_clustered = [f"pos {i}" for i in row_order_pos]
    
    # PCA for position embeddings
    X_pos = pos_emb_all.astype(np.float64)
    X_pos = X_pos - X_pos.mean(axis=0, keepdims=True)
    clusters_pos = fcluster(Z_pos, t=6, criterion="maxclust")
    
    # Create figure: 3 rows, 3 columns
    # Row 1: Token embeddings (raw, clustered, PCA)
    # Row 2: Position embeddings (raw, clustered, PCA)
    # Row 3: Token+Position embeddings (dim 0 heatmap, dim 1 heatmap, PCA)
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    
    # Row 1: Token embeddings
    # Embeddings heatmap
    ax1 = axes[0, 0]
    x_labels = list(range(embeddings.shape[1]))
    sns.heatmap(embeddings, yticklabels=y_labels, xticklabels=x_labels, cmap="RdBu_r", center=0, ax=ax1)
    ax1.set_title(f"Token Embeddings (vocab×embd={vocab_size}×{n_embd})", fontsize=11)
    ax1.set_xlabel("Embedding dim")
    ax1.set_ylabel("Token")
    
    # Clustered embeddings
    ax2 = axes[0, 1]
    x_labels = list(range(embeddings_clustered.shape[1]))
    sns.heatmap(embeddings_clustered, yticklabels=y_labels_clustered, xticklabels=x_labels, cmap="RdBu_r", center=0, ax=ax2)
    ax2.set_title(f"Token Embeddings Clustered (vocab×embd={vocab_size}×{n_embd})", fontsize=11)
    ax2.set_xlabel("Embedding dim")
    ax2.set_ylabel("Token (clustered)")
    
    # PCA or raw data
    ax3 = axes[0, 2]
    if n_embd > 2:
        # Do PCA for dimensions > 2
        _, _, Vt = np.linalg.svd(X_emb, full_matrices=False)
        X2 = X_emb @ Vt[:2].T
        ax3.scatter(X2[:, 0], X2[:, 1], s=0, alpha=0)  # Invisible points for layout
        ax3.set_title(f"Token Embeddings PCA 2D (vocab={vocab_size})", fontsize=11)
        ax3.set_xlabel("PC1")
        ax3.set_ylabel("PC2")
        ax3.grid(True, alpha=0.2)
        if vocab_size <= 80:
            for i in range(vocab_size):
                ax3.text(X2[i, 0], X2[i, 1], itos[i], fontsize=8)
    elif n_embd == 2:
        # For 2D embeddings, show raw data
        ax3.scatter(X_emb[:, 0], X_emb[:, 1], s=0, alpha=0)  # Invisible points for layout
        ax3.set_title(f"Token Embeddings (vocab={vocab_size})", fontsize=11)
        ax3.set_xlabel("Dim 0")
        ax3.set_ylabel("Dim 1")
        ax3.grid(True, alpha=0.2)
        if vocab_size <= 80:
            for i in range(vocab_size):
                ax3.text(X_emb[i, 0], X_emb[i, 1], itos[i], fontsize=8)
    else:
        # For 1D embeddings, just plot the single dimension
        X1 = X_emb[:, 0]
        ax3.scatter(X1, np.zeros_like(X1), s=0, alpha=0)  # Invisible points for layout
        ax3.set_title(f"Token Embeddings 1D (vocab={vocab_size})", fontsize=11)
        ax3.set_xlabel("Embedding value")
        ax3.set_ylabel("")
        ax3.grid(True, alpha=0.2)
        if vocab_size <= 80:
            for i in range(vocab_size):
                ax3.text(X1[i], 0, itos[i], fontsize=8, ha='center')
        ax3.set_yticks([])
    
    # Row 2: Position embeddings (raw, clustered, PCA)
    # Position embeddings heatmap
    ax4 = axes[1, 0]
    pos_y_labels = [f"pos {i}" for i in range(block_size)]
    x_labels = list(range(pos_emb_all.shape[1]))
    sns.heatmap(pos_emb_all, yticklabels=pos_y_labels, xticklabels=x_labels, cmap="RdBu_r", center=0, ax=ax4)
    ax4.set_title(f"Position Embeddings (block_size×embd={block_size}×{n_embd})", fontsize=11)
    ax4.set_xlabel("Embedding dim")
    ax4.set_ylabel("Position")
    
    # Clustered position embeddings
    ax5 = axes[1, 1]
    x_labels = list(range(pos_emb_clustered.shape[1]))
    sns.heatmap(pos_emb_clustered, yticklabels=pos_y_labels_clustered, xticklabels=x_labels, cmap="RdBu_r", center=0, ax=ax5)
    ax5.set_title(f"Position Embeddings Clustered (block_size×embd={block_size}×{n_embd})", fontsize=11)
    ax5.set_xlabel("Embedding dim")
    ax5.set_ylabel("Position (clustered)")
    
    # PCA or raw data for position embeddings
    ax6 = axes[1, 2]
    if n_embd > 2:
        # Do PCA for dimensions > 2
        _, _, Vt_pos = np.linalg.svd(X_pos, full_matrices=False)
        X2_pos = X_pos @ Vt_pos[:2].T
        ax6.scatter(X2_pos[:, 0], X2_pos[:, 1], s=0, alpha=0)  # Invisible points for layout
        ax6.set_title(f"Position Embeddings PCA 2D (block_size={block_size})", fontsize=11)
        ax6.set_xlabel("PC1")
        ax6.set_ylabel("PC2")
        ax6.grid(True, alpha=0.2)
        if block_size <= 80:
            for i in range(block_size):
                ax6.text(X2_pos[i, 0], X2_pos[i, 1], f"p{i}", fontsize=8)
    elif n_embd == 2:
        # For 2D embeddings, show raw data
        ax6.scatter(X_pos[:, 0], X_pos[:, 1], s=0, alpha=0)  # Invisible points for layout
        ax6.set_title(f"Position Embeddings (block_size={block_size})", fontsize=11)
        ax6.set_xlabel("Dim 0")
        ax6.set_ylabel("Dim 1")
        ax6.grid(True, alpha=0.2)
        if block_size <= 80:
            for i in range(block_size):
                ax6.text(X_pos[i, 0], X_pos[i, 1], f"p{i}", fontsize=8)
    else:
        # For 1D embeddings, just plot the single dimension
        X1_pos = X_pos[:, 0]
        ax6.scatter(X1_pos, np.arange(block_size), s=0, alpha=0)  # Invisible points for layout
        ax6.set_title(f"Position Embeddings 1D (block_size={block_size})", fontsize=11)
        ax6.set_xlabel("Embedding value")
        ax6.set_ylabel("Position index")
        ax6.grid(True, alpha=0.2)
        if block_size <= 80:
            for i in range(block_size):
                ax6.text(X1_pos[i], i, f"p{i}", fontsize=8, ha='center')
    
    # Row 3: Token+Position embeddings (heatmaps for each dimension, then PCA)
    # Create all token-position combinations (ALL tokens including special characters)
    max_token_idx = vocab_size  # Show all tokens including special characters
    num_combinations = max_token_idx * block_size
    all_combinations = np.zeros((num_combinations, n_embd))
    
    for token_idx in range(max_token_idx):
        for pos_idx in range(block_size):
            idx = token_idx * block_size + pos_idx
            all_combinations[idx] = embeddings[token_idx] + pos_emb_all[pos_idx]
    
    # Column 1: Heatmap for dimension 0 (rows: tokens, columns: positions)
    ax10 = axes[2, 0]
    dim0_heatmap = np.zeros((max_token_idx, block_size))
    for token_idx in range(max_token_idx):
        for pos_idx in range(block_size):
            idx = token_idx * block_size + pos_idx
            dim0_heatmap[token_idx, pos_idx] = all_combinations[idx, 0]
    
    token_labels = [itos[i] for i in range(max_token_idx)]
    pos_labels = [f"p{i}" for i in range(block_size)]
    sns.heatmap(dim0_heatmap, yticklabels=token_labels, xticklabels=pos_labels, cmap="RdBu_r", center=0, ax=ax10)
    ax10.set_title(f"Token+Position: Dim 0 (tokens×positions)", fontsize=11)
    ax10.set_xlabel("Position")
    ax10.set_ylabel("Token")
    
    # Column 2: Heatmap for dimension 1 (rows: tokens, columns: positions)
    ax11 = axes[2, 1]
    if n_embd >= 2:
        dim1_heatmap = np.zeros((max_token_idx, block_size))
        for token_idx in range(max_token_idx):
            for pos_idx in range(block_size):
                idx = token_idx * block_size + pos_idx
                dim1_heatmap[token_idx, pos_idx] = all_combinations[idx, 1]
        
        sns.heatmap(dim1_heatmap, yticklabels=token_labels, xticklabels=pos_labels, cmap="RdBu_r", center=0, ax=ax11)
        ax11.set_title(f"Token+Position: Dim 1 (tokens×positions)", fontsize=11)
        ax11.set_xlabel("Position")
        ax11.set_ylabel("Token")
    else:
        # For 1D embeddings, just show a placeholder or repeat dim 0
        ax11.text(0.5, 0.5, "N/A (1D embeddings)", ha='center', va='center', transform=ax11.transAxes)
        ax11.set_title("Token+Position: Dim 1 (N/A)", fontsize=11)
    
    # Column 3: PCA or raw data of all token-position combinations
    ax12 = axes[2, 2]
    if n_embd > 2:
        # Do PCA for dimensions > 2
        X_comb = all_combinations.astype(np.float64)
        X_comb = X_comb - X_comb.mean(axis=0, keepdims=True)
        _, _, Vt_comb = np.linalg.svd(X_comb, full_matrices=False)
        X2_comb = X_comb @ Vt_comb[:2].T
        
        # Plot with s=0 so only annotations are visible
        ax12.scatter(X2_comb[:, 0], X2_comb[:, 1], s=0, alpha=0)
        labels_comb = []
        for token_idx in range(max_token_idx):
            token_str = str(itos[token_idx])
            for pos_idx in range(block_size):
                labels_comb.append(f"{token_str}p{pos_idx}")  # No underscore
        
        for i in range(len(labels_comb)):
            ax12.text(X2_comb[i, 0], X2_comb[i, 1], labels_comb[i], fontsize=6, ha='center', va='center')
        
        ax12.set_title(f"Token+Position: PCA (all tokens)", fontsize=11)
        ax12.set_xlabel("PC1")
        ax12.set_ylabel("PC2")
        ax12.grid(True, alpha=0.2)
    elif n_embd == 2:
        # For 2D embeddings, show raw data
        X_comb = all_combinations.astype(np.float64)
        ax12.scatter(X_comb[:, 0], X_comb[:, 1], s=0, alpha=0)
        labels_comb = []
        for token_idx in range(max_token_idx):
            token_str = str(itos[token_idx])
            for pos_idx in range(block_size):
                labels_comb.append(f"{token_str}p{pos_idx}")
        
        for i in range(len(labels_comb)):
            ax12.text(X_comb[i, 0], X_comb[i, 1], labels_comb[i], fontsize=6, ha='center', va='center')
        
        ax12.set_title(f"Token+Position: Raw (all tokens)", fontsize=11)
        ax12.set_xlabel("Dim 0")
        ax12.set_ylabel("Dim 1")
        ax12.grid(True, alpha=0.2)
    else:
        # For 1D embeddings
        X1_comb = all_combinations[:, 0]
        ax12.scatter(X1_comb, np.zeros_like(X1_comb), s=0, alpha=0)
        labels_comb = []
        for token_idx in range(max_token_idx):
            token_str = str(itos[token_idx])
            for pos_idx in range(block_size):
                labels_comb.append(f"{token_str}p{pos_idx}")
        
        for i in range(len(labels_comb)):
            ax12.text(X1_comb[i], 0, labels_comb[i], fontsize=6, ha='center', va='center')
        
        ax12.set_title(f"Token+Position: Raw 1D (all tokens)", fontsize=11)
        ax12.set_xlabel("Embedding value")
        ax12.set_ylabel("")
        ax12.set_yticks([])
        ax12.grid(True, alpha=0.2)
        
        ax12.set_title(f"Token+Position: 1D (all tokens)", fontsize=11)
        ax12.set_xlabel("Embedding value")
        ax12.set_ylabel("")
        ax12.grid(True, alpha=0.2)
        ax12.set_yticks([])
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
    model.train()

@torch.no_grad()
def plot_token_position_embedding_space(model, itos, save_path=None):
    """
    BIG figure showing all token-position combinations in original embedding space and PCA space.
    Shows both RAW (first 2 dims) and PCA scatter plots side by side.
    """
    model.eval()
    
    # Get token and position embeddings
    embeddings = model.token_embedding.weight.detach().cpu().numpy()  # (vocab, N_EMBD)
    vocab_size, n_embd = embeddings.shape
    block_size = model.block_size
    pos_emb_all = model.position_embedding_table.weight.detach().cpu().numpy()  # (block_size, n_embd)
    
    # Create all token-position combinations (ALL tokens including special characters)
    num_combinations = vocab_size * block_size
    all_combinations = np.zeros((num_combinations, n_embd))
    labels = []
    
    for token_idx in range(vocab_size):
        token_str = str(itos[token_idx])
        for pos_idx in range(block_size):
            idx = token_idx * block_size + pos_idx
            all_combinations[idx] = embeddings[token_idx] + pos_emb_all[pos_idx]
            labels.append(f"{token_str}p{pos_idx}")  # No underscore
    
    # Create BIG figure: 1 row, 2 columns (RAW and PCA)
    fig, axes = plt.subplots(1, 2, figsize=(24, 12))
    
    # Left plot: Original embedding space (Dim 0 vs Dim 1)
    ax1 = axes[0]
    if n_embd >= 2:
        X_orig = all_combinations[:, [0, 1]]
        ax1.scatter(X_orig[:, 0], X_orig[:, 1], s=0, alpha=0)  # s=0 for invisible dots
        for i in range(len(labels)):
            ax1.text(X_orig[i, 0], X_orig[i, 1], labels[i], fontsize=6, ha='center', va='center')
        ax1.set_title(f"Original Token+Position Embeddings: Dim 0 vs Dim 1\n(All tokens, {num_combinations} combinations)", fontsize=14)
        ax1.set_xlabel("Embedding Dim 0", fontsize=12)
        ax1.set_ylabel("Embedding Dim 1", fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
    else:
        # 1D case
        X_orig_1d = all_combinations[:, 0]
        ax1.scatter(X_orig_1d, np.zeros_like(X_orig_1d), s=0, alpha=0)
        for i in range(len(labels)):
            ax1.text(X_orig_1d[i], 0, labels[i], fontsize=6, ha='center', va='center', rotation=90)
        ax1.set_title(f"Original Token+Position Embeddings: Dim 0\n(All tokens, {num_combinations} combinations)", fontsize=14)
        ax1.set_xlabel("Embedding Dim 0", fontsize=12)
        ax1.set_ylabel("")
        ax1.grid(True, alpha=0.3)
        ax1.set_yticks([])
    
    # Right plot: PCA space
    ax2 = axes[1]
    if n_embd >= 2:
        X_comb = all_combinations.astype(np.float64)
        X_comb = X_comb - X_comb.mean(axis=0, keepdims=True)
        _, _, Vt_comb = np.linalg.svd(X_comb, full_matrices=False)
        X2_comb = X_comb @ Vt_comb[:2].T
        
        # Plot with s=0 so only annotations are visible
        ax2.scatter(X2_comb[:, 0], X2_comb[:, 1], s=0, alpha=0)
        for i in range(len(labels)):
            ax2.text(X2_comb[i, 0], X2_comb[i, 1], labels[i], fontsize=6, ha='center', va='center')
        
        ax2.set_title(f"Token+Position Embeddings: PCA 2D\n(All tokens, {num_combinations} combinations)", fontsize=14)
        ax2.set_xlabel("PC1", fontsize=12)
        ax2.set_ylabel("PC2", fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
    else:
        # 1D embeddings
        X1_comb = all_combinations[:, 0]
        ax2.scatter(X1_comb, np.zeros_like(X1_comb), s=0, alpha=0)
        for i in range(len(labels)):
            ax2.text(X1_comb[i], 0, labels[i], fontsize=6, ha='center', va='center', rotation=90)
        ax2.set_title(f"Token+Position Embeddings: PCA 1D\n(All tokens, {num_combinations} combinations)", fontsize=14)
        ax2.set_xlabel("PC1", fontsize=12)
        ax2.set_ylabel("")
        ax2.grid(True, alpha=0.3)
        ax2.set_yticks([])
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
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
def plot_qkv_transformations(model, itos, save_path=None):
    """
    Plot QKV weights and their transformations of all token-position combinations.
    Row 1: W_Q, W_K, W_V weights
    Row 2: All token-position combinations transformed by Q, K, V respectively
    """
    model.eval()
    
    # Get token and position embeddings
    embeddings = model.token_embedding.weight.detach().cpu().numpy()  # (vocab, N_EMBD)
    vocab_size, n_embd = embeddings.shape
    block_size = model.block_size
    pos_emb_all = model.position_embedding_table.weight.detach().cpu().numpy()  # (block_size, n_embd)
    
    # Get QKV weights - average across heads
    Wq_all, Wk_all, Wv_all = [], [], []
    for h in model.sa_heads.heads:
        Wq_all.append(h.query.weight.cpu().numpy())  # (hs, C)
        Wk_all.append(h.key.weight.cpu().numpy())    # (hs, C)
        Wv_all.append(h.value.weight.cpu().numpy())  # (hs, C)
    
    # Average weights across heads
    W_Q = np.stack(Wq_all, axis=0).mean(axis=0)  # (hs, C)
    W_K = np.stack(Wk_all, axis=0).mean(axis=0)  # (hs, C)
    W_V = np.stack(Wv_all, axis=0).mean(axis=0)  # (hs, C)
    
    head_size = W_Q.shape[0]
    
    # Create all token-position combinations (ALL tokens including special characters)
    max_token_idx = vocab_size  # Show all tokens including special characters
    num_combinations = max_token_idx * block_size
    all_combinations = np.zeros((num_combinations, n_embd))
    labels = []
    
    for token_idx in range(max_token_idx):
        token_str = str(itos[token_idx])
        for pos_idx in range(block_size):
            idx = token_idx * block_size + pos_idx
            all_combinations[idx] = embeddings[token_idx] + pos_emb_all[pos_idx]
            labels.append(f"{token_str}p{pos_idx}")  # No underscore
    
    # Transform by Q, K, V
    # all_combinations is (N, n_embd), W_Q is (hs, n_embd)
    # Transform: (N, n_embd) @ (n_embd, hs) = (N, hs) -> transpose W_Q first
    Q_transformed = all_combinations @ W_Q.T  # (N, hs)
    K_transformed = all_combinations @ W_K.T  # (N, hs)
    V_transformed = all_combinations @ W_V.T  # (N, hs)
    
    # Create figure: 5 rows, 3 columns
    # Row 1: Original token+position embeddings (spanning all 3 columns)
    # Row 2: QKV weights
    # Row 3: Transformed token-position combinations (scatter)
    # Row 4: Q/K/V Dim 0 heatmaps (tokens × positions)
    # Row 5: Q/K/V Dim 1 heatmaps (tokens × positions)
    fig = plt.figure(figsize=(24, 32))
    gs = GridSpec(5, 3, figure=fig, hspace=0.4, wspace=0.3)
    
    # Row 1: Original token+position embeddings (spanning all 3 columns)
    ax0 = fig.add_subplot(gs[0, :])
    if n_embd >= 2:
        # Plot original embeddings in first 2 dimensions
        X_orig = all_combinations[:, [0, 1]]
        ax0.scatter(X_orig[:, 0], X_orig[:, 1], s=0, alpha=0)
        for i in range(len(labels)):
            ax0.text(X_orig[i, 0], X_orig[i, 1], labels[i], fontsize=6, ha='center', va='center')
        ax0.set_title(f"Original Token+Position Embeddings: Dim 0 vs Dim 1\n(All tokens, {num_combinations} combinations)", fontsize=12)
        ax0.set_xlabel("Embedding Dim 0")
        ax0.set_ylabel("Embedding Dim 1")
        ax0.grid(True, alpha=0.3)
        ax0.axis('equal')
    else:
        # 1D case
        X_orig_1d = all_combinations[:, 0]
        ax0.scatter(X_orig_1d, np.zeros_like(X_orig_1d), s=0, alpha=0)
        for i in range(len(labels)):
            ax0.text(X_orig_1d[i], 0, labels[i], fontsize=6, ha='center', va='center', rotation=90)
        ax0.set_title(f"Original Token+Position Embeddings: Dim 0\n(All tokens, {num_combinations} combinations)", fontsize=12)
        ax0.set_xlabel("Embedding Dim 0")
        ax0.set_ylabel("")
        ax0.grid(True, alpha=0.3)
        ax0.set_yticks([])
    
    # Row 2: QKV weights
    # W_Q
    ax1 = fig.add_subplot(gs[1, 0])
    x_labels = list(range(n_embd))
    y_labels_local = list(range(head_size))
    sns.heatmap(W_Q, cmap="viridis", xticklabels=x_labels, yticklabels=y_labels_local, cbar=True, ax=ax1)
    ax1.set_title(f"W_Q (hs×C={head_size}×{n_embd})", fontsize=12)
    ax1.set_xlabel("C (embedding dim)")
    ax1.set_ylabel("hs (head_size)")
    
    # W_K
    ax2 = fig.add_subplot(gs[1, 1])
    sns.heatmap(W_K, cmap="viridis", xticklabels=x_labels, yticklabels=y_labels_local, cbar=True, ax=ax2)
    ax2.set_title(f"W_K (hs×C={head_size}×{n_embd})", fontsize=12)
    ax2.set_xlabel("C (embedding dim)")
    ax2.set_ylabel("hs (head_size)")
    
    # W_V
    ax3 = fig.add_subplot(gs[1, 2])
    sns.heatmap(W_V, cmap="viridis", xticklabels=x_labels, yticklabels=y_labels_local, cbar=True, ax=ax3)
    ax3.set_title(f"W_V (hs×C={head_size}×{n_embd})", fontsize=12)
    ax3.set_xlabel("C (embedding dim)")
    ax3.set_ylabel("hs (head_size)")
    
    # Row 3: Transformed token-position combinations
    # Q-transformed
    ax4 = fig.add_subplot(gs[2, 0])
    if head_size >= 2:
        Q_2d = Q_transformed[:, [0, 1]]
        ax4.scatter(Q_2d[:, 0], Q_2d[:, 1], s=0, alpha=0)
        for i in range(len(labels)):
            ax4.text(Q_2d[i, 0], Q_2d[i, 1], labels[i], fontsize=7, ha='center', va='center')
        ax4.set_title(f"Q-Transformed: Dim 0 vs Dim 1\n(All tokens, {num_combinations} combinations)", fontsize=12)
        ax4.set_xlabel("Head Dim 0")
        ax4.set_ylabel("Head Dim 1")
        ax4.grid(True, alpha=0.3)
        ax4.axis('equal')
    else:
        # 1D case
        Q_1d = Q_transformed[:, 0]
        ax4.scatter(Q_1d, np.zeros_like(Q_1d), s=0, alpha=0)
        for i in range(len(labels)):
            ax4.text(Q_1d[i], 0, labels[i], fontsize=7, ha='center', va='center', rotation=90)
        ax4.set_title(f"Q-Transformed: Dim 0\n(All tokens, {num_combinations} combinations)", fontsize=12)
        ax4.set_xlabel("Head Dim 0")
        ax4.set_ylabel("")
        ax4.grid(True, alpha=0.3)
        ax4.set_yticks([])
    
    # K-transformed
    ax5 = fig.add_subplot(gs[2, 1])
    if head_size >= 2:
        K_2d = K_transformed[:, [0, 1]]
        ax5.scatter(K_2d[:, 0], K_2d[:, 1], s=0, alpha=0)
        for i in range(len(labels)):
            ax5.text(K_2d[i, 0], K_2d[i, 1], labels[i], fontsize=7, ha='center', va='center')
        ax5.set_title(f"K-Transformed: Dim 0 vs Dim 1\n(All tokens, {num_combinations} combinations)", fontsize=12)
        ax5.set_xlabel("Head Dim 0")
        ax5.set_ylabel("Head Dim 1")
        ax5.grid(True, alpha=0.3)
        ax5.axis('equal')
    else:
        K_1d = K_transformed[:, 0]
        ax5.scatter(K_1d, np.zeros_like(K_1d), s=0, alpha=0)
        for i in range(len(labels)):
            ax5.text(K_1d[i], 0, labels[i], fontsize=7, ha='center', va='center', rotation=90)
        ax5.set_title(f"K-Transformed: Dim 0\n(All tokens, {num_combinations} combinations)", fontsize=12)
        ax5.set_xlabel("Head Dim 0")
        ax5.set_ylabel("")
        ax5.grid(True, alpha=0.3)
        ax5.set_yticks([])
    
    # V-transformed
    ax6 = fig.add_subplot(gs[2, 2])
    if head_size >= 2:
        V_2d = V_transformed[:, [0, 1]]
        ax6.scatter(V_2d[:, 0], V_2d[:, 1], s=0, alpha=0)
        for i in range(len(labels)):
            ax6.text(V_2d[i, 0], V_2d[i, 1], labels[i], fontsize=7, ha='center', va='center')
        ax6.set_title(f"V-Transformed: Dim 0 vs Dim 1\n(All tokens, {num_combinations} combinations)", fontsize=12)
        ax6.set_xlabel("Head Dim 0")
        ax6.set_ylabel("Head Dim 1")
        ax6.grid(True, alpha=0.3)
        ax6.axis('equal')
    else:
        V_1d = V_transformed[:, 0]
        ax6.scatter(V_1d, np.zeros_like(V_1d), s=0, alpha=0)
        for i in range(len(labels)):
            ax6.text(V_1d[i], 0, labels[i], fontsize=7, ha='center', va='center', rotation=90)
        ax6.set_title(f"V-Transformed: Dim 0\n(All tokens, {num_combinations} combinations)", fontsize=12)
        ax6.set_xlabel("Head Dim 0")
        ax6.set_ylabel("")
        ax6.grid(True, alpha=0.3)
        ax6.set_yticks([])
    
    # Row 4: Q/K/V Dim 0 heatmaps (tokens × positions)
    # Create heatmap data: reshape transformed values into (tokens, positions) grid
    token_labels = [str(itos[i]) for i in range(vocab_size)]
    pos_labels = [f"p{i}" for i in range(block_size)]
    
    # Q Dim 0 heatmap
    ax7 = fig.add_subplot(gs[3, 0])
    Q_dim0_heatmap = Q_transformed[:, 0].reshape(vocab_size, block_size)
    sns.heatmap(Q_dim0_heatmap, cmap="RdBu_r", center=0, xticklabels=pos_labels, yticklabels=token_labels, cbar=True, ax=ax7)
    ax7.set_title(f"Q-Transformed: Dim 0 (tokens×positions)", fontsize=12)
    ax7.set_xlabel("Position")
    ax7.set_ylabel("Token")
    
    # K Dim 0 heatmap
    ax8 = fig.add_subplot(gs[3, 1])
    K_dim0_heatmap = K_transformed[:, 0].reshape(vocab_size, block_size)
    sns.heatmap(K_dim0_heatmap, cmap="RdBu_r", center=0, xticklabels=pos_labels, yticklabels=token_labels, cbar=True, ax=ax8)
    ax8.set_title(f"K-Transformed: Dim 0 (tokens×positions)", fontsize=12)
    ax8.set_xlabel("Position")
    ax8.set_ylabel("Token")
    
    # V Dim 0 heatmap
    ax9 = fig.add_subplot(gs[3, 2])
    V_dim0_heatmap = V_transformed[:, 0].reshape(vocab_size, block_size)
    sns.heatmap(V_dim0_heatmap, cmap="RdBu_r", center=0, xticklabels=pos_labels, yticklabels=token_labels, cbar=True, ax=ax9)
    ax9.set_title(f"V-Transformed: Dim 0 (tokens×positions)", fontsize=12)
    ax9.set_xlabel("Position")
    ax9.set_ylabel("Token")
    
    # Row 5: Q/K/V Dim 1 heatmaps (tokens × positions) - only if head_size >= 2
    if head_size >= 2:
        # Q Dim 1 heatmap
        ax10 = fig.add_subplot(gs[4, 0])
        Q_dim1_heatmap = Q_transformed[:, 1].reshape(vocab_size, block_size)
        sns.heatmap(Q_dim1_heatmap, cmap="RdBu_r", center=0, xticklabels=pos_labels, yticklabels=token_labels, cbar=True, ax=ax10)
        ax10.set_title(f"Q-Transformed: Dim 1 (tokens×positions)", fontsize=12)
        ax10.set_xlabel("Position")
        ax10.set_ylabel("Token")
        
        # K Dim 1 heatmap
        ax11 = fig.add_subplot(gs[4, 1])
        K_dim1_heatmap = K_transformed[:, 1].reshape(vocab_size, block_size)
        sns.heatmap(K_dim1_heatmap, cmap="RdBu_r", center=0, xticklabels=pos_labels, yticklabels=token_labels, cbar=True, ax=ax11)
        ax11.set_title(f"K-Transformed: Dim 1 (tokens×positions)", fontsize=12)
        ax11.set_xlabel("Position")
        ax11.set_ylabel("Token")
        
        # V Dim 1 heatmap
        ax12 = fig.add_subplot(gs[4, 2])
        V_dim1_heatmap = V_transformed[:, 1].reshape(vocab_size, block_size)
        sns.heatmap(V_dim1_heatmap, cmap="RdBu_r", center=0, xticklabels=pos_labels, yticklabels=token_labels, cbar=True, ax=ax12)
        ax12.set_title(f"V-Transformed: Dim 1 (tokens×positions)", fontsize=12)
        ax12.set_xlabel("Position")
        ax12.set_ylabel("Token")
    else:
        # For 1D head_size, show placeholder
        for col in range(3):
            ax = fig.add_subplot(gs[4, col])
            ax.text(0.5, 0.5, "N/A (1D head_size)", ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f"{'QKV'[col]}-Transformed: Dim 1 (N/A)", fontsize=12)
            ax.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()
    model.train()

@torch.no_grad()
def plot_weights_qkv_two_sequences(model, X_list, itos, save_path=None, num_sequences=3):
    """
    Plot QKV for multiple sequences, split into two separate plots.
    Plot 1: Q, K, masked QK^T, Attention, scatter(Q vs K)
    Plot 2: Attention, V, Final Output, scatter(V vs Final Output with lines)
    
    Args:
        model: The model
        X_list: List of input sequences, or single sequence (will be converted to list)
        itos: Index to string mapping
        save_path: Base path to save figures (will save as save_path_part1.png and save_path_part2.png)
        num_sequences: Number of sequences to show (if X_list is single sequence, will use it multiple times)
    """
    model.eval()
    
    # Handle single sequence input
    if not isinstance(X_list, list):
        X_list = [X_list]
    
    # Use provided sequences up to num_sequences
    sequences_to_plot = X_list[:num_sequences]
    num_sequences = len(sequences_to_plot)
    
    # Helper function to compute PCA for 2D visualization
    def pca_2d(data):
        """Reduce data to 2D using PCA only when dimension > 2"""
        if data.shape[1] <= 2:
            # Use raw data directly when dimension is 2 or less
            if data.shape[1] == 2:
                return data
            elif data.shape[1] == 1:
                # Pad with zeros if only 1 dimension
                result = np.zeros((data.shape[0], 2))
                result[:, 0] = data[:, 0]
                return result
            else:
                return data[:, :2]
        # Center the data
        data_centered = data - data.mean(axis=0, keepdims=True)
        # SVD for PCA
        U, s, Vt = np.linalg.svd(data_centered, full_matrices=False)
        # Project to first 2 principal components
        return data_centered @ Vt[:2].T
    
    # Prepare data for all sequences
    all_data = []
    for seq_idx, X in enumerate(sequences_to_plot):
        tokens = [itos[i.item()] for i in X[0]]
        seq_str = " ".join(tokens)
        B, T = X.shape
        
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
        
        # Compute masked QK^T (apply causal mask before softmax)
        QK_T_torch = torch.from_numpy(QK_T).float()
        # Create causal mask (lower triangular)
        tril_mask = torch.tril(torch.ones(T, T))
        Masked_QK_T = QK_T_torch.masked_fill(tril_mask == 0, float("-inf")).numpy()  # (T, T)
        
        # Get attention weights (softmax(QK^T) after masking) from all heads
        wei_all = []
        for h in model.sa_heads.heads:
            _, wei = h(x)  # wei: (B, T, T)
            wei_all.append(wei[0].cpu().numpy())  # (T, T)
        
        # Average attention weights across heads
        Attention = np.stack(wei_all, axis=0).mean(axis=0)  # (T, T)
        
        # Compute final output: Attention @ V (this is what actually gets used)
        Final_Output = Attention @ V  # (T, T) @ (T, hs) = (T, hs)
        
        all_data.append({
            'tokens': tokens,
            'seq_str': seq_str,
            'seq_idx': seq_idx,
            'Q': Q,
            'K': K,
            'V': V,
            'Masked_QK_T': Masked_QK_T,
            'Attention': Attention,
            'Final_Output': Final_Output,
            'T': T
        })
    
    # ========== PLOT 1: Q, K, masked QK^T, Attention, scatter(Q vs K) ==========
    num_cols_plot1 = 5  # Q, K, masked QK^T, Attention, scatter
    fig1 = plt.figure(figsize=(6 * num_cols_plot1, 4 * num_sequences))
    gs1 = GridSpec(num_sequences, num_cols_plot1, figure=fig1, hspace=0.4, wspace=0.3)
    
    for data_dict in all_data:
        seq_idx = data_dict['seq_idx']
        tokens = data_dict['tokens']
        seq_str = data_dict['seq_str']
        Q = data_dict['Q']
        K = data_dict['K']
        Masked_QK_T = data_dict['Masked_QK_T']
        Attention = data_dict['Attention']
        T = data_dict['T']
        
        # Column 0: Q
        ax = fig1.add_subplot(gs1[seq_idx, 0])
        dim_str = f"(T×hs={Q.shape[0]}×{Q.shape[1]})"
        sns.heatmap(Q, cmap="viridis", xticklabels=list(range(Q.shape[1])), 
                   yticklabels=tokens, cbar=True, ax=ax)
        ax.set_xlabel("hs", fontsize=10)
        ax.set_ylabel(f"Seq {seq_idx+1}\n{seq_str}\n", fontsize=9)
        ax.set_title(f"Q {dim_str}", fontsize=11)
        
        # Column 1: K
        ax = fig1.add_subplot(gs1[seq_idx, 1])
        dim_str = f"(T×hs={K.shape[0]}×{K.shape[1]})"
        sns.heatmap(K, cmap="viridis", xticklabels=list(range(K.shape[1])), 
                   yticklabels=tokens, cbar=True, ax=ax)
        ax.set_xlabel("hs", fontsize=10)
        ax.set_ylabel("T", fontsize=10)
        ax.set_title(f"K {dim_str}", fontsize=11)
        
        # Column 2: Scatter plot Q vs K
        ax = fig1.add_subplot(gs1[seq_idx, 2])
        Q_2d = pca_2d(Q)
        K_2d = pca_2d(K)
        
        # Plot points with smaller markers
        ax.scatter(Q_2d[:, 0], Q_2d[:, 1], label='Q', alpha=0.7, s=20)
        ax.scatter(K_2d[:, 0], K_2d[:, 1], label='K', alpha=0.7, s=20, marker='^')
        
        # Annotate all points with token and position
        for i, (token, pos) in enumerate(zip(tokens, range(len(tokens)))):
            # Annotate Q points
            ax.annotate(f'{token}p{pos}', (Q_2d[i, 0], Q_2d[i, 1]), 
                       fontsize=7, alpha=0.8, xytext=(3, 3), textcoords='offset points')
            # Annotate K points
            ax.annotate(f'{token}p{pos}', (K_2d[i, 0], K_2d[i, 1]), 
                       fontsize=7, alpha=0.8, xytext=(3, 3), textcoords='offset points')
        
        # Update axis labels based on whether PCA was used
        if Q.shape[1] > 2:
            ax.set_xlabel("PC1", fontsize=10)
            ax.set_ylabel("PC2", fontsize=10)
            title_suffix = " (PCA)"
        else:
            ax.set_xlabel("Dim 1", fontsize=10)
            ax.set_ylabel("Dim 2", fontsize=10)
            title_suffix = " (raw)"
        ax.set_title(f"Q vs K{title_suffix}", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Column 3: masked QK^T
        ax = fig1.add_subplot(gs1[seq_idx, 3])
        dim_str = f"(T×T={T}×{T})"
        sns.heatmap(Masked_QK_T, cmap="viridis", xticklabels=tokens, 
                   yticklabels=tokens, cbar=True, ax=ax)
        ax.set_xlabel("T", fontsize=10)
        ax.set_ylabel("T", fontsize=10)
        ax.set_title(f"masked QK^T {dim_str}", fontsize=11)
        
        # Column 4: Attention
        ax = fig1.add_subplot(gs1[seq_idx, 4])
        dim_str = f"(T×T={T}×{T})"
        sns.heatmap(Attention, cmap="magma", vmin=0.0, vmax=1.0, 
                   xticklabels=tokens, yticklabels=tokens, cbar=True, ax=ax)
        ax.set_xlabel("T", fontsize=10)
        ax.set_ylabel("T", fontsize=10)
        ax.set_title(f"Attention {dim_str}", fontsize=11)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    if save_path:
        # Save with meaningful name: query_key_attention (contains Q, K, masked QK^T, Attention)
        save_path_part1 = save_path.replace('qkv.png', 'qkv_query_key_attention.png') if 'qkv.png' in save_path else save_path.replace('.png', '_query_key_attention.png') if save_path.endswith('.png') else save_path + '_query_key_attention.png'
        plt.savefig(save_path_part1, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()
    
    # ========== PLOT 2: Attention, V, Final Output, scatter(V vs Final Output with lines) ==========
    num_cols_plot2 = 4  # Attention, V, Final Output, scatter
    fig2 = plt.figure(figsize=(6 * num_cols_plot2, 4 * num_sequences))
    gs2 = GridSpec(num_sequences, num_cols_plot2, figure=fig2, hspace=0.4, wspace=0.3)
    
    for data_dict in all_data:
        seq_idx = data_dict['seq_idx']
        tokens = data_dict['tokens']
        seq_str = data_dict['seq_str']
        V = data_dict['V']
        Attention = data_dict['Attention']
        Final_Output = data_dict['Final_Output']
        T = data_dict['T']
        
        # Column 0: Attention (same as plot 1)
        ax = fig2.add_subplot(gs2[seq_idx, 0])
        dim_str = f"(T×T={T}×{T})"
        sns.heatmap(Attention, cmap="magma", vmin=0.0, vmax=1.0, 
                   xticklabels=tokens, yticklabels=tokens, cbar=True, ax=ax)
        ax.set_xlabel("T", fontsize=10)
        ax.set_ylabel(f"Seq {seq_idx+1}\n{seq_str}\n", fontsize=9)
        ax.set_title(f"Attention {dim_str}", fontsize=11)
        
        # Column 1: V
        ax = fig2.add_subplot(gs2[seq_idx, 1])
        dim_str = f"(T×hs={V.shape[0]}×{V.shape[1]})"
        sns.heatmap(V, cmap="viridis", xticklabels=list(range(V.shape[1])), 
                   yticklabels=tokens, cbar=True, ax=ax)
        ax.set_xlabel("hs", fontsize=10)
        ax.set_ylabel("T", fontsize=10)
        ax.set_title(f"V {dim_str}", fontsize=11)
        
        # Column 2: Final Output
        ax = fig2.add_subplot(gs2[seq_idx, 2])
        dim_str = f"(T×hs={Final_Output.shape[0]}×{Final_Output.shape[1]})"
        sns.heatmap(Final_Output, cmap="viridis", xticklabels=list(range(Final_Output.shape[1])), 
                   yticklabels=tokens, cbar=True, ax=ax)
        ax.set_xlabel("hs", fontsize=10)
        ax.set_ylabel("T", fontsize=10)
        ax.set_title(f"Final Output {dim_str}", fontsize=11)
        
        # Column 3: Scatter plot V vs Final Output with arrows
        ax = fig2.add_subplot(gs2[seq_idx, 3])
        V_2d = pca_2d(V)
        Final_Output_2d = pca_2d(Final_Output)
        
        # Plot points with smaller markers
        ax.scatter(V_2d[:, 0], V_2d[:, 1], label='V', alpha=0.7, s=20, color='blue')
        ax.scatter(Final_Output_2d[:, 0], Final_Output_2d[:, 1], label='Final Output', alpha=0.7, s=20, color='red', marker='^')
        
        # Draw arrows pointing from V to Final Output
        for i in range(len(V_2d)):
            ax.annotate('', xy=(Final_Output_2d[i, 0], Final_Output_2d[i, 1]),
                       xytext=(V_2d[i, 0], V_2d[i, 1]),
                       arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5, lw=1.5))
        
        # Annotate all points with token and position
        for i, (token, pos) in enumerate(zip(tokens, range(len(tokens)))):
            # Annotate V points
            ax.annotate(f'{token}p{pos}', (V_2d[i, 0], V_2d[i, 1]), 
                       fontsize=7, alpha=0.8, xytext=(3, 3), textcoords='offset points', color='blue')
            # Annotate Final Output points
            ax.annotate(f'{token}p{pos}', (Final_Output_2d[i, 0], Final_Output_2d[i, 1]), 
                       fontsize=7, alpha=0.8, xytext=(3, 3), textcoords='offset points', color='red')
        
        # Update axis labels based on whether PCA was used
        if V.shape[1] > 2:
            ax.set_xlabel("PC1", fontsize=10)
            ax.set_ylabel("PC2", fontsize=10)
            title_suffix = " (PCA)"
        else:
            ax.set_xlabel("Dim 1", fontsize=10)
            ax.set_ylabel("Dim 2", fontsize=10)
            title_suffix = " (raw)"
        ax.set_title(f"V vs Final Output{title_suffix}", fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    if save_path:
        # Save with meaningful name: value_output (contains Attention, V, Final Output)
        save_path_part2 = save_path.replace('qkv.png', 'qkv_value_output.png') if 'qkv.png' in save_path else save_path.replace('.png', '_value_output.png') if save_path.endswith('.png') else save_path + '_value_output.png'
        plt.savefig(save_path_part2, bbox_inches='tight', dpi=150)
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
    
    # Compute masked QK^T (apply causal mask before softmax)
    QK_T_torch = torch.from_numpy(QK_T).float()
    # Create causal mask (lower triangular)
    tril_mask = torch.tril(torch.ones(T, T))
    Masked_QK_T = QK_T_torch.masked_fill(tril_mask == 0, float("-inf")).numpy()  # (T, T)
    
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
    
    # Compute final output: Attention @ V (this is what actually gets used)
    Final_Output = Attention @ V  # (T, T) @ (T, hs) = (T, hs)
    
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
    
    # Row 2: Q, K, masked QK^T, Attention, V, Final Output
    row2_titles = ["Q", "K", "masked QK^T", "Attention", "V", "Final Output"]
    row2_data = [Q, K, Masked_QK_T, Attention, V, Final_Output]
    
    for j, (data, title) in enumerate(zip(row2_data, row2_titles)):
        ax = fig.add_subplot(gs[row_offset + 1, j])
        if title in ["masked QK^T", "Attention"]:
            # masked QK^T and Attention are (T, T) matrices - use tokens for both axes
            x_labels = tokens
            y_labels_local = tokens
            dim_str = f"(T×T={data.shape[0]}×{data.shape[1]})"
            
            cmap = "magma" if title == "Attention" else "viridis"
            vmin, vmax = (0.0, 1.0) if title == "Attention" else (None, None)
            
            sns.heatmap(data, cmap=cmap, vmin=vmin, vmax=vmax,
                        xticklabels=x_labels, yticklabels=y_labels_local, cbar=True, ax=ax)
            ax.set_title(f"{title} {dim_str}", fontsize=11)
            if j == 0:
                ax.set_ylabel("T", fontsize=10)
            ax.set_xlabel("T", fontsize=10)
        elif title == "Final Output":
            # Final Output is (T, head_size) matrix - use tokens for y-axis
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

@torch.no_grad()
def estimate_rule_error(model, generator, decode, block_size, num_samples=20, seq_length=30):
    """
    Generate sequences and check rule error.
    Returns the fraction of CONSTRAINED positions that violate the rule.
    For operator-based generators, only counts positions immediately after operators.
    """
    model.eval()
    
    total_constrained = 0
    incorrect_constrained = 0
    
    vocab_size = model.token_embedding.weight.shape[0]
    
    for _ in range(num_samples):
        # Generate a sequence
        start_token = random.randint(0, vocab_size - 1)
        start = torch.tensor([[start_token]], dtype=torch.long)
        sample = model.generate(start, max_new_tokens=seq_length - 1)[0].tolist()
        generated_integers = decode(sample)
        
        # Verify the sequence
        correctness, _ = generator.verify_sequence(generated_integers)
        
        # For operator-based generators, only count positions after operators
        if isinstance(generator, OperatorBasedGenerator):
            for i, token in enumerate(generated_integers[:-1]):
                if generator.is_operator(token):
                    # Position i+1 is constrained by the rule
                    if i + 1 < len(correctness):
                        total_constrained += 1
                        if correctness[i + 1] == 0:
                            incorrect_constrained += 1
        else:
            # For non-operator generators, count all positions except first
            total_constrained += len(correctness) - 1
            incorrect_constrained += sum(1 for c in correctness[1:] if c == 0)
    
    model.train()
    return incorrect_constrained / total_constrained if total_constrained > 0 else 0.0


def plot_generated_sequences_heatmap(generated_sequences, generator, save_path=None, num_sequences=5, max_length=30):
    """
    Plot generated sequences as an annotated heatmap showing correctness.
    Red = incorrect, White/Green = correct.
    """
    sequences_to_show = generated_sequences[:num_sequences]
    
    # Truncate to max_length and pad to same length
    max_len = min(max_length, max(len(seq) for seq in sequences_to_show))
    
    # Create data matrix and correctness matrix
    data_matrix = []
    correctness_matrix = []
    
    for seq in sequences_to_show:
        seq_truncated = seq[:max_len]
        correctness, _ = generator.verify_sequence(seq_truncated)
        
        # Pad if necessary
        while len(seq_truncated) < max_len:
            seq_truncated.append(-1)  # Padding value
            correctness.append(-1)  # Padding marker
        
        data_matrix.append(seq_truncated)
        correctness_matrix.append(correctness)
    
    data_matrix = np.array(data_matrix)
    correctness_matrix = np.array(correctness_matrix)
    
    # Create figure - scale appropriately for number of sequences
    fig_height = max(6, min(20, num_sequences * 0.6 + 2))
    fig_width = min(24, max(12, max_len * 0.5))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Create color matrix: 1 (correct) = light green, 0 (incorrect) = red, -1 (padding) = gray
    from matplotlib.colors import ListedColormap
    colors = ['#ff6b6b', '#90EE90', '#d3d3d3']  # red, light green, gray
    cmap = ListedColormap(colors)
    
    # Map correctness to color indices: 0->0 (red), 1->1 (green), -1->2 (gray)
    color_indices = np.where(correctness_matrix == -1, 2, correctness_matrix)
    
    # Plot the heatmap
    im = ax.imshow(color_indices, cmap=cmap, aspect='auto', vmin=0, vmax=2)
    
    # Add text annotations (the actual numbers)
    # Adjust font size based on number of sequences
    fontsize = max(7, min(10, 12 - num_sequences // 5))
    for i in range(len(sequences_to_show)):
        for j in range(max_len):
            val = data_matrix[i, j]
            if val != -1:  # Not padding
                text_color = 'black'
                ax.text(j, i, str(val), ha='center', va='center', fontsize=fontsize, color=text_color, fontweight='bold')
    
    # Set labels
    ax.set_xlabel("Position in Sequence", fontsize=11)
    ax.set_ylabel("Sequence #", fontsize=11)
    ax.set_title("Generated Sequences with Rule Correctness\n(Green = Correct, Red = Incorrect)", fontsize=12)
    
    # Set ticks
    ax.set_xticks(range(0, max_len, max(1, max_len // 15)))
    ax.set_yticks(range(num_sequences))
    ax.set_yticklabels([f"Seq {i+1}" for i in range(num_sequences)])
    
    # Add colorbar/legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#90EE90', label='Correct'),
        Patch(facecolor='#ff6b6b', label='Incorrect'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()
    
    # Return summary statistics
    total_correct = np.sum(correctness_matrix == 1)
    total_incorrect = np.sum(correctness_matrix == 0)
    total_valid = total_correct + total_incorrect
    accuracy = total_correct / total_valid if total_valid > 0 else 0
    return accuracy, total_correct, total_incorrect

def plot_training_data_heatmap(training_sequences, generator, save_path=None, num_sequences=50, max_length=50):
    """
    Plot training data sequences as an annotated heatmap showing correctness.
    Red = incorrect, Green = correct.
    Same style as generated sequences heatmap.
    """
    sequences_to_show = training_sequences[:num_sequences]
    
    # Truncate to max_length and pad to same length
    max_len = min(max_length, max(len(seq) for seq in sequences_to_show))
    
    # Create data matrix and correctness matrix
    data_matrix = []
    correctness_matrix = []
    
    for seq in sequences_to_show:
        seq_truncated = seq[:max_len]
        correctness, _ = generator.verify_sequence(seq_truncated)
        
        # Pad if necessary
        while len(seq_truncated) < max_len:
            seq_truncated.append(-1)  # Padding value
            correctness.append(-1)  # Padding marker
        
        data_matrix.append(seq_truncated)
        correctness_matrix.append(correctness)
    
    data_matrix = np.array(data_matrix)
    correctness_matrix = np.array(correctness_matrix)
    
    # Create figure - scale appropriately for number of sequences
    fig_height = max(8, min(30, num_sequences * 0.4 + 2))
    fig_width = min(24, max(12, max_len * 0.4))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Create color matrix: 1 (correct) = light green, 0 (incorrect) = red, -1 (padding) = gray
    from matplotlib.colors import ListedColormap
    colors = ['#ff6b6b', '#90EE90', '#d3d3d3']  # red, light green, gray
    cmap = ListedColormap(colors)
    
    # Map correctness to color indices: 0->0 (red), 1->1 (green), -1->2 (gray)
    color_indices = np.where(correctness_matrix == -1, 2, correctness_matrix)
    
    # Plot the heatmap
    im = ax.imshow(color_indices, cmap=cmap, aspect='auto', vmin=0, vmax=2)
    
    # Add text annotations (the actual numbers)
    # Adjust font size based on number of sequences
    fontsize = max(6, min(9, 11 - num_sequences // 10))
    for i in range(len(sequences_to_show)):
        for j in range(max_len):
            val = data_matrix[i, j]
            if val != -1:  # Not padding
                text_color = 'black'
                ax.text(j, i, str(val), ha='center', va='center', fontsize=fontsize, color=text_color, fontweight='bold')
    
    # Set labels
    ax.set_xlabel("Position in Sequence", fontsize=11)
    ax.set_ylabel("Sequence #", fontsize=11)
    ax.set_title("Training Data Sequences with Rule Correctness\n(Green = Correct, Red = Incorrect)", fontsize=12)
    
    # Set ticks
    ax.set_xticks(range(0, max_len, max(1, max_len // 15)))
    ax.set_yticks(range(0, num_sequences, max(1, num_sequences // 20)))
    ax.set_yticklabels([f"Seq {i+1}" for i in range(0, num_sequences, max(1, num_sequences // 20))])
    
    # Add colorbar/legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#90EE90', label='Correct'),
        Patch(facecolor='#ff6b6b', label='Incorrect'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()
    
    # Return summary statistics
    total_correct = np.sum(correctness_matrix == 1)
    total_incorrect = np.sum(correctness_matrix == 0)
    total_valid = total_correct + total_incorrect
    accuracy = total_correct / total_valid if total_valid > 0 else 0
    return accuracy, total_correct, total_incorrect

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

    def __init__(self, n_embd, ffwd_mult=4):
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
        # Project attention output back to n_embd for residual connection
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        # Feedforward layer for computation after attention
        self.ffwd = FeedForward(n_embd, ffwd_mult=16)
        # lm_head maps n_embd -> vocab
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.block_size = block_size

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None, return_wei: bool = False):
        B, T = idx.shape

        token_emb = self.token_embedding(idx)  # (B,T,C)
        positions = torch.arange(T, device=idx.device) % self.block_size
        pos_emb = self.position_embedding_table(positions)  # (T,C)

        x = token_emb + pos_emb  # (B,T,n_embd)

        attn_out, wei = self.sa_heads(x)  # (B,T,num_heads*head_size)
        attn_out = self.proj(attn_out)    # (B,T,n_embd)
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

# -----------------------------
# Main training script
# -----------------------------
def main(config_name: str = "copy_modulo"):
    """
    Main training function.
    
    Args:
        config_name: Name of the config file (without .yaml extension) in the configs folder
    """
    print(f"Starting training with config: {config_name}")
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
    
    # 2) Build encoder/decoder for integers (or integers + operators)
    min_value = data_config['min_value']
    max_value = data_config['max_value']
    
    # Check if generator uses operators
    if isinstance(generator, OperatorBasedGenerator):
        operators = generator.operators
        encode, decode, vocab_size, itos, stoi = build_encoder_with_operators(
            min_value=min_value, max_value=max_value, operators=operators
        )
        print("Vocabulary size:", vocab_size)
        print("Vocabulary (integers + operators):", [itos[i] for i in range(vocab_size)])
    else:
        encode, decode, vocab_size, itos, stoi = build_encoder_for_integers(min_value=min_value, max_value=max_value)
    print("Vocabulary size:", vocab_size)
    print("Vocabulary (integers):", [itos[i] for i in range(vocab_size)])

    # 3) Encode sequences (integer values -> token indices)
    encoded_sequences = [encode(seq) for seq in sequences]
    
    # 4) Split sequences into train/val
    train_sequences, val_sequences = split_train_val_sequences(encoded_sequences, train_ratio=0.9)
    print(f"Train: {len(train_sequences)} sequences, Val: {len(val_sequences)} sequences")
    
    # 4.5) Save some training data to a text file
    num_samples_to_save = min(50, len(sequences))  # Save up to 50 sample sequences
    with open(os.path.join(plots_dir, "training_data_samples.txt"), "w", encoding="utf-8") as f:
        f.write(f"# Training Data Samples for: {config_name_actual}\n")
        f.write(f"# {num_samples_to_save} sample sequences (original integer values before encoding)\n")
        f.write(f"# Format: space-separated integers, one sequence per line\n\n")
        for i, seq in enumerate(sequences[:num_samples_to_save]):
            f.write(" ".join(str(val) for val in seq) + "\n")
    print(f"Saved {num_samples_to_save} training data samples to {os.path.join(plots_dir, 'training_data_samples.txt')}")
    
    # 4.6) Visualize training data with rule correctness
    train_heatmap_accuracy, train_correct_count, train_incorrect_count = plot_training_data_heatmap(
        sequences, generator,  # Use original sequences (before encoding)
        save_path=os.path.join(plots_dir, "training_data_heatmap.png"),
        num_sequences=min(50, len(sequences)),  # Show up to 50 training sequences
        max_length=50
    )
    print(f"Training data heatmap: {train_correct_count} correct, {train_incorrect_count} incorrect positions ({train_heatmap_accuracy:.1%} accuracy)")

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
    rule_error_history = []

    batch_size = training_config['batch_size']
    max_steps = training_config['max_steps']
    eval_interval = training_config['eval_interval']
    eval_iterations = training_config['eval_iterations']
    
    X_fixed, _ = get_batch_from_sequences(train_sequences, block_size, batch_size)

    for step in range(max_steps):
        # Evaluate occasionally
        if step % eval_interval == 0:
            losses = estimate_loss(model, train_sequences, val_sequences, block_size, batch_size, eval_iterations)
            rule_err = estimate_rule_error(model, generator, decode, block_size, num_samples=20, seq_length=30)

            steps_for_plot.append(step)
            train_loss_history.append(losses["train"])
            val_loss_history.append(losses["validation"])
            rule_error_history.append(rule_err)

            print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['validation']:.4f}, rule err {rule_err:.4f}", flush=True)

        # One batch
        X, Y = get_batch_from_sequences(train_sequences, block_size, batch_size)

        # Forward + backward + update
        _, loss = model(X, Y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Beep to signal training is complete
    print('\a', end='', flush=True)

    # 4) Show results
    print("Final loss:", loss.item(), flush=True)
    print(f"Final rule error: {rule_error_history[-1]:.4f}" if rule_error_history else "", flush=True)

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
    # Combined learning curve with loss and rule accuracy
    plot_learning_curve(steps_for_plot, train_loss_history, val_loss_history, 
                       rule_error_history=rule_error_history,
                       save_path=os.path.join(plots_dir, "learning_curve.png"), 
                       eval_interval=eval_interval)
    
    # Plot annotated heatmap of generated sequences showing correctness (all sequences on one figure)
    heatmap_accuracy, correct_count, incorrect_count = plot_generated_sequences_heatmap(
        generated_sequences, generator,
        save_path=os.path.join(plots_dir, "generated_sequences_heatmap.png"),
        num_sequences=len(generated_sequences),  # Show all generated sequences
        max_length=50  # Show more positions
    )
    print(f"Generated sequences heatmap: {correct_count} correct, {incorrect_count} incorrect positions ({heatmap_accuracy:.1%} accuracy)")
    
    # Create multiple example sequences for plotting
    X1, _ = get_batch_from_sequences(train_sequences, block_size, 1)
    X2, _ = get_batch_from_sequences(train_sequences, block_size, 1)
    X3, _ = get_batch_from_sequences(train_sequences, block_size, 1)
    X_list = [X1, X2, X3]
    
    # Plot QKV for multiple sequences (shown as rows)
    plot_weights_qkv_two_sequences(model, X_list, itos, save_path=os.path.join(plots_dir, "qkv.png"), num_sequences=3)
    plot_embeddings_pca(model, itos, save_path=os.path.join(plots_dir, "embeddings.png"))
    plot_qkv_transformations(model, itos, save_path=os.path.join(plots_dir, "qkv_transformations.png"))
    plot_token_position_embedding_space(model, itos, save_path=os.path.join(plots_dir, "token_position_embedding_space.png"))
    plot_attention_matrix(model, X_list, itos, save_path=os.path.join(plots_dir, "attention_matrix.png"), num_sequences=3)



if __name__ == "__main__":
    # Get config name from command line argument, default to "default"
    config_name = sys.argv[1] if len(sys.argv) > 1 else "plus_last_even"
    main(config_name=config_name)
