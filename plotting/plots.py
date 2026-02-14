"""Plotting functions for embeddings, attention, heatmaps, architecture."""
import os
import numpy as np
import torch
from torch.nn import functional as F
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import matplotlib.patheffects as pe
import seaborn as sns

from data import get_batch_from_sequences

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

# Unicode subscript digits ₀₁₂₃₄₅₆₇₈₉ for position labels (e.g. "8₃" = token 8 at position 3)
_SUBSCRIPT_DIGITS = "₀₁₂₃₄₅₆₇₈₉"
def _pos_subscript(n):
    """Format integer n as subscript (e.g. 3 -> ₃, 12 -> ₁₂)."""
    return "".join(_SUBSCRIPT_DIGITS[int(d)] for d in str(n))
def _token_pos_label(token_str, pos_idx):
    """Label for token at position: token with position as subscript (e.g. '8₃')."""
    return f"{token_str}{_pos_subscript(pos_idx)}"
def _pos_only_label(pos_idx):
    """Label for position-only: p with subscript (e.g. 'p₀')."""
    return f"p{_pos_subscript(pos_idx)}"

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
    plt.title(f"Causal Mask (tril) {t_size}×{t_size} —allowed=1 blocked=0")
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

    # 4) Attention weights —readable version
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
    plt.title(f"Attention Weights {t_size}×{t_size} —readable")
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
    import matplotlib.colors as mcolors
    
    # Dynamic font sizing based on number of items
    def get_fontsize(num_items):
        # Subscript labels (e.g. 8₃) are more compact than "8p3", so we can use larger fonts
        if num_items <= 12:
            return 22
        elif num_items <= 20:
            return 18
        elif num_items <= 40:
            return 14
        elif num_items <= 80:
            return 12
        elif num_items <= 150:
            return 10
        else:
            return 8
    
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
    
    # Create color maps
    # Token colors: warm spectrum (reds/oranges/yellows)
    token_cmap = plt.cm.get_cmap('YlOrRd')
    token_colors = [token_cmap(0.3 + 0.6 * i / max(vocab_size - 1, 1)) for i in range(vocab_size)]
    
    # Position colors: cool spectrum (blues/purples)
    pos_cmap = plt.cm.get_cmap('cool')
    pos_colors = [pos_cmap(0.2 + 0.7 * i / max(block_size - 1, 1)) for i in range(block_size)]
    
    # Helper to blend colors
    def blend_colors(token_color, pos_color, token_weight=0.6):
        tc = np.array(mcolors.to_rgb(token_color))
        pc = np.array(mcolors.to_rgb(pos_color))
        return tuple(token_weight * tc + (1 - token_weight) * pc)
    
    # Create figure: 2 rows, 3 columns
    # Row 0: First column as a row — Token raw, Position raw, Token+Position dim 0
    # Row 1: Third column as a row — Token PCA, Position PCA, Token+Position PCA
    # Middle column (clustered) is deleted
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Row 0, Col 0: Token embeddings raw (was column 0, row 0)
    ax1 = axes[0, 0]
    x_labels = list(range(embeddings.shape[1]))
    sns.heatmap(embeddings, yticklabels=y_labels, xticklabels=x_labels, cmap="RdBu_r", center=0, ax=ax1)
    ax1.set_title(f"Token Embeddings (vocab×embd={vocab_size}×{n_embd})", fontsize=11)
    ax1.set_xlabel("Embedding dim")
    ax1.set_ylabel("Token")
    
    # Row 1, Col 0: Token embeddings PCA (was column 2, row 0)
    ax3 = axes[1, 0]
    if n_embd > 2:
        # Do PCA for dimensions > 2
        _, _, Vt = np.linalg.svd(X_emb, full_matrices=False)
        X2 = X_emb @ Vt[:2].T
        margin = 0.15 * max(X2[:, 0].max() - X2[:, 0].min(), X2[:, 1].max() - X2[:, 1].min())
        ax3.set_xlim(X2[:, 0].min() - margin, X2[:, 0].max() + margin)
        ax3.set_ylim(X2[:, 1].min() - margin, X2[:, 1].max() + margin)
        ax3.set_title(f"Token Embeddings PCA 2D (vocab={vocab_size})", fontsize=11, fontweight='bold')
        ax3.set_xlabel("PC1")
        ax3.set_ylabel("PC2")
        ax3.grid(True, alpha=0.3)
        token_fontsize = get_fontsize(vocab_size)
        if vocab_size <= 80:
            for i in range(vocab_size):
                ax3.text(X2[i, 0], X2[i, 1], itos[i], fontsize=token_fontsize, fontweight='bold',
                        ha='center', va='center', color=token_colors[i])
    elif n_embd == 2:
        # For 2D embeddings, show raw data
        margin = 0.15 * max(X_emb[:, 0].max() - X_emb[:, 0].min(), X_emb[:, 1].max() - X_emb[:, 1].min())
        ax3.set_xlim(X_emb[:, 0].min() - margin, X_emb[:, 0].max() + margin)
        ax3.set_ylim(X_emb[:, 1].min() - margin, X_emb[:, 1].max() + margin)
        ax3.set_title(f"Token Embeddings (vocab={vocab_size})", fontsize=11, fontweight='bold')
        ax3.set_xlabel("Dim 0")
        ax3.set_ylabel("Dim 1")
        ax3.grid(True, alpha=0.3)
        token_fontsize = get_fontsize(vocab_size)
        if vocab_size <= 80:
            for i in range(vocab_size):
                ax3.text(X_emb[i, 0], X_emb[i, 1], itos[i], fontsize=token_fontsize, fontweight='bold',
                        ha='center', va='center', color=token_colors[i])
    else:
        # For 1D embeddings, just plot the single dimension
        X1 = X_emb[:, 0]
        margin = 0.15 * (X1.max() - X1.min())
        ax3.set_xlim(X1.min() - margin, X1.max() + margin)
        ax3.set_ylim(-0.5, 0.5)
        ax3.set_title(f"Token Embeddings 1D (vocab={vocab_size})", fontsize=11, fontweight='bold')
        ax3.set_xlabel("Embedding value")
        ax3.set_ylabel("")
        ax3.grid(True, alpha=0.3)
        token_fontsize = get_fontsize(vocab_size)
        if vocab_size <= 80:
            for i in range(vocab_size):
                ax3.text(X1[i], 0, itos[i], fontsize=token_fontsize, fontweight='bold',
                        ha='center', va='center', color=token_colors[i])
        ax3.set_yticks([])
    
    # Row 0, Col 1: Position embeddings raw (was column 0, row 1)
    ax4 = axes[0, 1]
    pos_y_labels = [f"pos {i}" for i in range(block_size)]
    x_labels = list(range(pos_emb_all.shape[1]))
    sns.heatmap(pos_emb_all, yticklabels=pos_y_labels, xticklabels=x_labels, cmap="RdBu_r", center=0, ax=ax4)
    ax4.set_title(f"Position Embeddings (block_size×embd={block_size}×{n_embd})", fontsize=11)
    ax4.set_xlabel("Embedding dim")
    ax4.set_ylabel("Position")
    
    # Row 1, Col 1: Position embeddings PCA (was column 2, row 1)
    ax6 = axes[1, 1]
    if n_embd > 2:
        # Do PCA for dimensions > 2
        _, _, Vt_pos = np.linalg.svd(X_pos, full_matrices=False)
        X2_pos = X_pos @ Vt_pos[:2].T
        margin = 0.15 * max(X2_pos[:, 0].max() - X2_pos[:, 0].min(), X2_pos[:, 1].max() - X2_pos[:, 1].min())
        ax6.set_xlim(X2_pos[:, 0].min() - margin, X2_pos[:, 0].max() + margin)
        ax6.set_ylim(X2_pos[:, 1].min() - margin, X2_pos[:, 1].max() + margin)
        ax6.set_title(f"Position Embeddings PCA 2D (block_size={block_size})", fontsize=11, fontweight='bold')
        ax6.set_xlabel("PC1")
        ax6.set_ylabel("PC2")
        ax6.grid(True, alpha=0.3)
        pos_fontsize = get_fontsize(block_size)
        if block_size <= 80:
            for i in range(block_size):
                ax6.text(X2_pos[i, 0], X2_pos[i, 1], _pos_only_label(i), fontsize=pos_fontsize, fontweight='bold',
                        ha='center', va='center', color=pos_colors[i])
    elif n_embd == 2:
        # For 2D embeddings, show raw data
        margin = 0.15 * max(X_pos[:, 0].max() - X_pos[:, 0].min(), X_pos[:, 1].max() - X_pos[:, 1].min())
        ax6.set_xlim(X_pos[:, 0].min() - margin, X_pos[:, 0].max() + margin)
        ax6.set_ylim(X_pos[:, 1].min() - margin, X_pos[:, 1].max() + margin)
        ax6.set_title(f"Position Embeddings (block_size={block_size})", fontsize=11, fontweight='bold')
        ax6.set_xlabel("Dim 0")
        ax6.set_ylabel("Dim 1")
        ax6.grid(True, alpha=0.3)
        pos_fontsize = get_fontsize(block_size)
        if block_size <= 80:
            for i in range(block_size):
                ax6.text(X_pos[i, 0], X_pos[i, 1], _pos_only_label(i), fontsize=pos_fontsize, fontweight='bold',
                        ha='center', va='center', color=pos_colors[i])
    else:
        # For 1D embeddings, just plot the single dimension
        X1_pos = X_pos[:, 0]
        margin = 0.15 * (X1_pos.max() - X1_pos.min())
        ax6.set_xlim(X1_pos.min() - margin, X1_pos.max() + margin)
        ax6.set_ylim(-0.5, block_size - 0.5)
        ax6.set_title(f"Position Embeddings 1D (block_size={block_size})", fontsize=11, fontweight='bold')
        ax6.set_xlabel("Embedding value")
        ax6.set_ylabel("Position index")
        ax6.grid(True, alpha=0.3)
        pos_fontsize = get_fontsize(block_size)
        if block_size <= 80:
            for i in range(block_size):
                ax6.text(X1_pos[i], i, _pos_only_label(i), fontsize=pos_fontsize, fontweight='bold',
                        ha='center', va='center', color=pos_colors[i])
    
    # Row 2: Token+Position dim 0 heatmap (was column 0, row 2)
    # Create all token-position combinations (ALL tokens including special characters)
    max_token_idx = vocab_size  # Show all tokens including special characters
    num_combinations = max_token_idx * block_size
    all_combinations = np.zeros((num_combinations, n_embd))
    
    for token_idx in range(max_token_idx):
        for pos_idx in range(block_size):
            idx = token_idx * block_size + pos_idx
            all_combinations[idx] = embeddings[token_idx] + pos_emb_all[pos_idx]
    
    # Row 0, Col 2: Token+Position dim 0 heatmap (was column 0, row 2)
    ax10 = axes[0, 2]
    dim0_heatmap = np.zeros((max_token_idx, block_size))
    for token_idx in range(max_token_idx):
        for pos_idx in range(block_size):
            idx = token_idx * block_size + pos_idx
            dim0_heatmap[token_idx, pos_idx] = all_combinations[idx, 0]
    
    token_labels = [itos[i] for i in range(max_token_idx)]
    pos_labels = [_pos_only_label(i) for i in range(block_size)]
    sns.heatmap(dim0_heatmap, yticklabels=token_labels, xticklabels=pos_labels, cmap="RdBu_r", center=0, ax=ax10)
    ax10.set_title(f"Token+Position: Dim 0 (tokens×positions)", fontsize=11)
    ax10.set_xlabel("Position")
    ax10.set_ylabel("Token")
    
    # Row 1, Col 2: Token+Position PCA (was column 2, row 2)
    ax12 = axes[1, 2]
    # Dynamic font size for token+position (usually more items)
    combo_fontsize = get_fontsize(num_combinations)
    
    if n_embd > 2:
        # Do PCA for dimensions > 2
        X_comb = all_combinations.astype(np.float64)
        X_comb = X_comb - X_comb.mean(axis=0, keepdims=True)
        _, _, Vt_comb = np.linalg.svd(X_comb, full_matrices=False)
        X2_comb = X_comb @ Vt_comb[:2].T
        
        margin = 0.15 * max(X2_comb[:, 0].max() - X2_comb[:, 0].min(), X2_comb[:, 1].max() - X2_comb[:, 1].min())
        ax12.set_xlim(X2_comb[:, 0].min() - margin, X2_comb[:, 0].max() + margin)
        ax12.set_ylim(X2_comb[:, 1].min() - margin, X2_comb[:, 1].max() + margin)
        
        for token_idx in range(max_token_idx):
            token_str = str(itos[token_idx])
            for pos_idx in range(block_size):
                idx = token_idx * block_size + pos_idx
                label = _token_pos_label(token_str, pos_idx)
                color = blend_colors(token_colors[token_idx], pos_colors[pos_idx])
                ax12.text(X2_comb[idx, 0], X2_comb[idx, 1], label, fontsize=combo_fontsize, fontweight='bold',
                         ha='center', va='center', color=color)
        
        ax12.set_title(f"Token+Position: PCA (all tokens)", fontsize=11, fontweight='bold')
        ax12.set_xlabel("PC1")
        ax12.set_ylabel("PC2")
        ax12.grid(True, alpha=0.3)
    elif n_embd == 2:
        # For 2D embeddings, show raw data
        X_comb = all_combinations.astype(np.float64)
        
        margin = 0.15 * max(X_comb[:, 0].max() - X_comb[:, 0].min(), X_comb[:, 1].max() - X_comb[:, 1].min())
        ax12.set_xlim(X_comb[:, 0].min() - margin, X_comb[:, 0].max() + margin)
        ax12.set_ylim(X_comb[:, 1].min() - margin, X_comb[:, 1].max() + margin)
        
        for token_idx in range(max_token_idx):
            token_str = str(itos[token_idx])
            for pos_idx in range(block_size):
                idx = token_idx * block_size + pos_idx
                label = _token_pos_label(token_str, pos_idx)
                color = blend_colors(token_colors[token_idx], pos_colors[pos_idx])
                ax12.text(X_comb[idx, 0], X_comb[idx, 1], label, fontsize=combo_fontsize, fontweight='bold',
                         ha='center', va='center', color=color)
        
        ax12.set_title(f"Token+Position: Raw (all tokens)", fontsize=11, fontweight='bold')
        ax12.set_xlabel("Dim 0")
        ax12.set_ylabel("Dim 1")
        ax12.grid(True, alpha=0.3)
    else:
        # For 1D embeddings
        X1_comb = all_combinations[:, 0]
        
        margin = 0.15 * (X1_comb.max() - X1_comb.min())
        ax12.set_xlim(X1_comb.min() - margin, X1_comb.max() + margin)
        ax12.set_ylim(-0.5, 0.5)
        
        for token_idx in range(max_token_idx):
            token_str = str(itos[token_idx])
            for pos_idx in range(block_size):
                idx = token_idx * block_size + pos_idx
                label = _token_pos_label(token_str, pos_idx)
                color = blend_colors(token_colors[token_idx], pos_colors[pos_idx])
                ax12.text(X1_comb[idx], 0, label, fontsize=combo_fontsize, fontweight='bold',
                         ha='center', va='center', color=color)
        
        ax12.set_title(f"Token+Position: 1D (all tokens)", fontsize=11, fontweight='bold')
        ax12.set_xlabel("Embedding value")
        ax12.set_ylabel("")
        ax12.grid(True, alpha=0.3)
        ax12.set_yticks([])
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
    model.train()


@torch.no_grad()
def plot_embeddings_scatterplots_only(model, itos, save_path=None, fixed_limits=None, step_label=None):
    """
    Create a separate figure with just the 3 scatterplots from plot_embeddings_pca:
    1. Token Embeddings scatterplot (warm colors: reds/oranges)
    2. Position Embeddings scatterplot (cool colors: blues/purples)
    3. Token+Position Embeddings scatterplot (merged colors)
    
    Args:
        model: The model to visualize
        itos: Index-to-string mapping
        save_path: Path to save the figure
        fixed_limits: Optional dict with keys 'token', 'position', 'combined' each containing (xlim, ylim) tuples
        step_label: Optional label to add to the figure title (e.g., "Step: 1000")
    """
    import matplotlib.colors as mcolors
    
    # Dynamic font sizing based on number of items
    def get_fontsize(num_items):
        # Subscript labels are more compact, so we can use larger fonts
        if num_items <= 12:
            return 24
        elif num_items <= 20:
            return 20
        elif num_items <= 40:
            return 16
        elif num_items <= 80:
            return 13
        elif num_items <= 150:
            return 11
        else:
            return 9
    
    model.eval()
    
    # Get token embeddings
    embeddings = model.token_embedding.weight.detach().cpu().numpy()  # (vocab, N_EMBD)
    vocab_size, n_embd = embeddings.shape
    
    # Get position embeddings for all positions
    block_size = model.block_size
    pos_emb_all = model.position_embedding_table.weight.detach().cpu().numpy()  # (block_size, n_embd)
    
    # Create color maps
    # Token colors: warm spectrum (reds/oranges/yellows)
    token_cmap = plt.cm.get_cmap('YlOrRd')
    token_colors = [token_cmap(0.3 + 0.6 * i / max(vocab_size - 1, 1)) for i in range(vocab_size)]
    
    # Position colors: cool spectrum (blues/purples)
    pos_cmap = plt.cm.get_cmap('cool')
    pos_colors = [pos_cmap(0.2 + 0.7 * i / max(block_size - 1, 1)) for i in range(block_size)]
    
    # Create figure: 1 row, 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    
    # Add step label to figure title if provided
    if step_label is not None:
        fig.suptitle(f"Step: {step_label}", fontsize=16, fontweight='bold', y=0.98)
    
    # Column 1: Token Embeddings scatterplot
    ax1 = axes[0]
    X_emb = embeddings.astype(np.float64)
    X_emb = X_emb - X_emb.mean(axis=0, keepdims=True)
    
    if n_embd > 2:
        _, _, Vt = np.linalg.svd(X_emb, full_matrices=False)
        X2 = X_emb @ Vt[:2].T
        # Set axis limits with margin (or use fixed limits)
        if fixed_limits and 'token' in fixed_limits:
            ax1.set_xlim(fixed_limits['token'][0])
            ax1.set_ylim(fixed_limits['token'][1])
        else:
            margin = 0.15 * max(X2[:, 0].max() - X2[:, 0].min(), X2[:, 1].max() - X2[:, 1].min())
            ax1.set_xlim(X2[:, 0].min() - margin, X2[:, 0].max() + margin)
            ax1.set_ylim(X2[:, 1].min() - margin, X2[:, 1].max() + margin)
        ax1.set_title(f"Token Embeddings PCA 2D (vocab={vocab_size})", fontsize=14, fontweight='bold')
        ax1.set_xlabel("PC1", fontsize=12)
        ax1.set_ylabel("PC2", fontsize=12)
        token_fontsize = get_fontsize(vocab_size)
        if vocab_size <= 80:
            for i in range(vocab_size):
                ax1.text(X2[i, 0], X2[i, 1], itos[i], fontsize=token_fontsize, fontweight='bold',
                        ha='center', va='center', color=token_colors[i])
    elif n_embd == 2:
        X2 = X_emb
        # Set axis limits with margin (or use fixed limits)
        if fixed_limits and 'token' in fixed_limits:
            ax1.set_xlim(fixed_limits['token'][0])
            ax1.set_ylim(fixed_limits['token'][1])
        else:
            margin = 0.15 * max(X2[:, 0].max() - X2[:, 0].min(), X2[:, 1].max() - X2[:, 1].min())
            ax1.set_xlim(X2[:, 0].min() - margin, X2[:, 0].max() + margin)
            ax1.set_ylim(X2[:, 1].min() - margin, X2[:, 1].max() + margin)
        ax1.set_title(f"Token Embeddings (vocab={vocab_size})", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Dim 0", fontsize=12)
        ax1.set_ylabel("Dim 1", fontsize=12)
        token_fontsize = get_fontsize(vocab_size)
        if vocab_size <= 80:
            for i in range(vocab_size):
                ax1.text(X2[i, 0], X2[i, 1], itos[i], fontsize=token_fontsize, fontweight='bold',
                        ha='center', va='center', color=token_colors[i])
    else:
        X1 = X_emb[:, 0]
        if fixed_limits and 'token' in fixed_limits:
            ax1.set_xlim(fixed_limits['token'][0])
            ax1.set_ylim(fixed_limits['token'][1])
        else:
            margin = 0.15 * (X1.max() - X1.min())
            ax1.set_xlim(X1.min() - margin, X1.max() + margin)
            ax1.set_ylim(-0.5, 0.5)
        ax1.set_title(f"Token Embeddings 1D (vocab={vocab_size})", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Embedding value", fontsize=12)
        ax1.set_ylabel("")
        token_fontsize = get_fontsize(vocab_size)
        if vocab_size <= 80:
            for i in range(vocab_size):
                ax1.text(X1[i], 0, itos[i], fontsize=token_fontsize, fontweight='bold',
                        ha='center', va='center', color=token_colors[i])
        ax1.set_yticks([])
    ax1.grid(True, alpha=0.3)
    
    # Column 2: Position Embeddings scatterplot
    ax2 = axes[1]
    X_pos = pos_emb_all.astype(np.float64)
    X_pos = X_pos - X_pos.mean(axis=0, keepdims=True)
    
    if n_embd > 2:
        _, _, Vt_pos = np.linalg.svd(X_pos, full_matrices=False)
        X2_pos = X_pos @ Vt_pos[:2].T
        if fixed_limits and 'position' in fixed_limits:
            ax2.set_xlim(fixed_limits['position'][0])
            ax2.set_ylim(fixed_limits['position'][1])
        else:
            margin = 0.15 * max(X2_pos[:, 0].max() - X2_pos[:, 0].min(), X2_pos[:, 1].max() - X2_pos[:, 1].min())
            ax2.set_xlim(X2_pos[:, 0].min() - margin, X2_pos[:, 0].max() + margin)
            ax2.set_ylim(X2_pos[:, 1].min() - margin, X2_pos[:, 1].max() + margin)
        ax2.set_title(f"Position Embeddings PCA 2D (block_size={block_size})", fontsize=14, fontweight='bold')
        ax2.set_xlabel("PC1", fontsize=12)
        ax2.set_ylabel("PC2", fontsize=12)
        pos_fontsize = get_fontsize(block_size)
        if block_size <= 80:
            for i in range(block_size):
                ax2.text(X2_pos[i, 0], X2_pos[i, 1], _pos_only_label(i), fontsize=pos_fontsize, fontweight='bold',
                        ha='center', va='center', color=pos_colors[i])
    elif n_embd == 2:
        X2_pos = X_pos
        if fixed_limits and 'position' in fixed_limits:
            ax2.set_xlim(fixed_limits['position'][0])
            ax2.set_ylim(fixed_limits['position'][1])
        else:
            margin = 0.15 * max(X2_pos[:, 0].max() - X2_pos[:, 0].min(), X2_pos[:, 1].max() - X2_pos[:, 1].min())
            ax2.set_xlim(X2_pos[:, 0].min() - margin, X2_pos[:, 0].max() + margin)
            ax2.set_ylim(X2_pos[:, 1].min() - margin, X2_pos[:, 1].max() + margin)
        ax2.set_title(f"Position Embeddings (block_size={block_size})", fontsize=14, fontweight='bold')
        ax2.set_xlabel("Dim 0", fontsize=12)
        ax2.set_ylabel("Dim 1", fontsize=12)
        pos_fontsize = get_fontsize(block_size)
        if block_size <= 80:
            for i in range(block_size):
                ax2.text(X2_pos[i, 0], X2_pos[i, 1], _pos_only_label(i), fontsize=pos_fontsize, fontweight='bold',
                        ha='center', va='center', color=pos_colors[i])
    else:
        X1_pos = X_pos[:, 0]
        if fixed_limits and 'position' in fixed_limits:
            ax2.set_xlim(fixed_limits['position'][0])
            ax2.set_ylim(fixed_limits['position'][1])
        else:
            margin = 0.15 * (X1_pos.max() - X1_pos.min())
            ax2.set_xlim(X1_pos.min() - margin, X1_pos.max() + margin)
            ax2.set_ylim(-0.5, block_size - 0.5)
        ax2.set_title(f"Position Embeddings 1D (block_size={block_size})", fontsize=14, fontweight='bold')
        ax2.set_xlabel("Embedding value", fontsize=12)
        ax2.set_ylabel("Position index", fontsize=12)
        pos_fontsize = get_fontsize(block_size)
        if block_size <= 80:
            for i in range(block_size):
                ax2.text(X1_pos[i], i, _pos_only_label(i), fontsize=pos_fontsize, fontweight='bold',
                        ha='center', va='center', color=pos_colors[i])
    ax2.grid(True, alpha=0.3)
    
    # Column 3: Token+Position Embeddings scatterplot
    ax3 = axes[2]
    max_token_idx = vocab_size
    num_combinations = max_token_idx * block_size
    all_combinations = np.zeros((num_combinations, n_embd))
    
    for token_idx in range(max_token_idx):
        for pos_idx in range(block_size):
            idx = token_idx * block_size + pos_idx
            all_combinations[idx] = embeddings[token_idx] + pos_emb_all[pos_idx]
    
    # Create merged colors: blend token hue with position lightness
    def blend_colors(token_color, pos_color, token_weight=0.6):
        """Blend token and position colors."""
        tc = np.array(mcolors.to_rgb(token_color))
        pc = np.array(mcolors.to_rgb(pos_color))
        blended = token_weight * tc + (1 - token_weight) * pc
        return tuple(blended)
    
    # Dynamic font size for token+position (usually more items)
    combo_fontsize = get_fontsize(num_combinations)
    
    if n_embd > 2:
        X_comb = all_combinations.astype(np.float64)
        X_comb = X_comb - X_comb.mean(axis=0, keepdims=True)
        _, _, Vt_comb = np.linalg.svd(X_comb, full_matrices=False)
        X2_comb = X_comb @ Vt_comb[:2].T
        
        if fixed_limits and 'combined' in fixed_limits:
            ax3.set_xlim(fixed_limits['combined'][0])
            ax3.set_ylim(fixed_limits['combined'][1])
        else:
            margin = 0.15 * max(X2_comb[:, 0].max() - X2_comb[:, 0].min(), X2_comb[:, 1].max() - X2_comb[:, 1].min())
            ax3.set_xlim(X2_comb[:, 0].min() - margin, X2_comb[:, 0].max() + margin)
            ax3.set_ylim(X2_comb[:, 1].min() - margin, X2_comb[:, 1].max() + margin)
        
        for token_idx in range(max_token_idx):
            token_str = str(itos[token_idx])
            for pos_idx in range(block_size):
                idx = token_idx * block_size + pos_idx
                label = _token_pos_label(token_str, pos_idx)
                color = blend_colors(token_colors[token_idx], pos_colors[pos_idx])
                ax3.text(X2_comb[idx, 0], X2_comb[idx, 1], label, fontsize=combo_fontsize, fontweight='bold',
                        ha='center', va='center', color=color)
        
        ax3.set_title(f"Token+Position: PCA (all tokens)", fontsize=14, fontweight='bold')
        ax3.set_xlabel("PC1", fontsize=12)
        ax3.set_ylabel("PC2", fontsize=12)
    elif n_embd == 2:
        X_comb = all_combinations.astype(np.float64)
        
        if fixed_limits and 'combined' in fixed_limits:
            ax3.set_xlim(fixed_limits['combined'][0])
            ax3.set_ylim(fixed_limits['combined'][1])
        else:
            margin = 0.15 * max(X_comb[:, 0].max() - X_comb[:, 0].min(), X_comb[:, 1].max() - X_comb[:, 1].min())
            ax3.set_xlim(X_comb[:, 0].min() - margin, X_comb[:, 0].max() + margin)
            ax3.set_ylim(X_comb[:, 1].min() - margin, X_comb[:, 1].max() + margin)
        
        for token_idx in range(max_token_idx):
            token_str = str(itos[token_idx])
            for pos_idx in range(block_size):
                idx = token_idx * block_size + pos_idx
                label = _token_pos_label(token_str, pos_idx)
                color = blend_colors(token_colors[token_idx], pos_colors[pos_idx])
                ax3.text(X_comb[idx, 0], X_comb[idx, 1], label, fontsize=combo_fontsize, fontweight='bold',
                        ha='center', va='center', color=color)
        
        ax3.set_title(f"Token+Position: Raw (all tokens)", fontsize=14, fontweight='bold')
        ax3.set_xlabel("Dim 0", fontsize=12)
        ax3.set_ylabel("Dim 1", fontsize=12)
    else:
        X1_comb = all_combinations[:, 0]
        
        if fixed_limits and 'combined' in fixed_limits:
            ax3.set_xlim(fixed_limits['combined'][0])
            ax3.set_ylim(fixed_limits['combined'][1])
        else:
            margin = 0.15 * (X1_comb.max() - X1_comb.min())
            ax3.set_xlim(X1_comb.min() - margin, X1_comb.max() + margin)
            ax3.set_ylim(-0.5, 0.5)
        
        for token_idx in range(max_token_idx):
            token_str = str(itos[token_idx])
            for pos_idx in range(block_size):
                idx = token_idx * block_size + pos_idx
                label = _token_pos_label(token_str, pos_idx)
                color = blend_colors(token_colors[token_idx], pos_colors[pos_idx])
                ax3.text(X1_comb[idx], 0, label, fontsize=combo_fontsize, fontweight='bold',
                        ha='center', va='center', color=color)
        
        ax3.set_title(f"Token+Position: 1D (all tokens)", fontsize=14, fontweight='bold')
        ax3.set_xlabel("Embedding value", fontsize=12)
        ax3.set_ylabel("")
        ax3.set_yticks([])
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()
    model.train()


@torch.no_grad()
def plot_embedding_qkv_comprehensive(model, itos, save_path=None, fixed_limits=None, step_label=None):
    """
    Comprehensive figure showing all key embedding spaces:
    Row 1: Token embeddings, Position embeddings, Token+Position (sum)
    Row 2: Q-transformed, K-transformed, V-transformed
    Row 3: Q and K together (red/blue), Attention-weighted output
    
    Args:
        model: The model to visualize
        itos: Index-to-string mapping
        save_path: Path to save the figure
        fixed_limits: Optional dict with limits for consistent animation
        step_label: Optional step number for title
    """
    import matplotlib.colors as mcolors
    
    model.eval()
    
    # Dynamic font sizing
    def get_fontsize(num_items):
        # Subscript labels are more compact, so we can use larger fonts
        if num_items <= 12:
            return 20
        elif num_items <= 20:
            return 16
        elif num_items <= 40:
            return 13
        elif num_items <= 80:
            return 11
        elif num_items <= 150:
            return 9
        else:
            return 7
    
    # Get embeddings
    embeddings = model.token_embedding.weight.detach().cpu().numpy()
    vocab_size, n_embd = embeddings.shape
    block_size = model.block_size
    pos_emb_all = model.position_embedding_table.weight.detach().cpu().numpy()
    
    # Color maps
    token_cmap = plt.cm.get_cmap('YlOrRd')
    token_colors = [token_cmap(0.3 + 0.6 * i / max(vocab_size - 1, 1)) for i in range(vocab_size)]
    pos_cmap = plt.cm.get_cmap('cool')
    pos_colors = [pos_cmap(0.2 + 0.7 * i / max(block_size - 1, 1)) for i in range(block_size)]
    
    def blend_colors(tc, pc, w=0.6):
        tc_rgb = np.array(mcolors.to_rgb(tc))
        pc_rgb = np.array(mcolors.to_rgb(pc))
        return tuple(w * tc_rgb + (1 - w) * pc_rgb)
    
    # Create token+position combinations
    num_combinations = vocab_size * block_size
    all_combinations = np.zeros((num_combinations, n_embd))
    combo_labels = []
    combo_colors = []
    
    for token_idx in range(vocab_size):
        for pos_idx in range(block_size):
            idx = token_idx * block_size + pos_idx
            all_combinations[idx] = embeddings[token_idx] + pos_emb_all[pos_idx]
            combo_labels.append(_token_pos_label(itos[token_idx], pos_idx))
            combo_colors.append(blend_colors(token_colors[token_idx], pos_colors[pos_idx]))
    
    # Get QKV weights from first head
    head = model.sa_heads.heads[0]
    W_Q = head.query.weight.detach().cpu().numpy()
    W_K = head.key.weight.detach().cpu().numpy()
    W_V = head.value.weight.detach().cpu().numpy()
    head_size = W_Q.shape[0]
    
    # Transform by Q, K, V
    Q_transformed = all_combinations @ W_Q.T
    K_transformed = all_combinations @ W_K.T
    V_transformed = all_combinations @ W_V.T
    
    # Masked attention output only: causal position masking (pos p attends only to 0..p)
    attn_outputs_masked = []
    for query_idx in range(num_combinations):
        pos_p = query_idx % block_size
        q = Q_transformed[query_idx:query_idx+1]  # (1, hs)
        scores = (q @ K_transformed.T) / np.sqrt(head_size)  # (1, N)
        # Mask: attend only to keys at position <= pos_p
        mask = np.ones(num_combinations, dtype=bool)
        for key_idx in range(num_combinations):
            pos_q = key_idx % block_size
            if pos_q > pos_p:
                scores[0, key_idx] = -1e9
                mask[key_idx] = False
        row_max = scores[0, mask].max() if mask.any() else 0.0
        scores_exp = np.exp(scores - row_max)
        scores_exp[0, ~mask] = 0.0
        total = scores_exp.sum()
        attn_weights = scores_exp / total if total > 0 else scores_exp
        out = attn_weights @ V_transformed  # (1, hs)
        attn_outputs_masked.append(out[0])
    attn_outputs_masked = np.array(attn_outputs_masked)  # (N, hs)
    
    # Create figure: 3 rows, 3 columns
    # Smaller size for video frames to avoid ffmpeg encoder issues
    fig_size = (16, 13) if step_label is not None else (24, 20)
    fig = plt.figure(figsize=fig_size)
    
    if step_label is not None:
        fig.suptitle(f"Step: {step_label}", fontsize=20, fontweight='bold', y=0.98)
    
    # Helper to get 2D coords (raw or first 2 dims)
    def get_2d(data):
        if data.shape[1] >= 2:
            return data[:, 0], data[:, 1]
        else:
            return data[:, 0], np.zeros(len(data))
    
    def set_limits(ax, key, x, y):
        if fixed_limits and key in fixed_limits:
            ax.set_xlim(fixed_limits[key][0])
            ax.set_ylim(fixed_limits[key][1])
        else:
            margin_x = 0.15 * (x.max() - x.min()) if x.max() != x.min() else 1
            margin_y = 0.15 * (y.max() - y.min()) if y.max() != y.min() else 1
            ax.set_xlim(x.min() - margin_x, x.max() + margin_x)
            ax.set_ylim(y.min() - margin_y, y.max() + margin_y)
    
    # Row 1: Token, Position, Token+Position
    # Token embeddings
    ax1 = fig.add_subplot(3, 3, 1)
    X_emb = embeddings - embeddings.mean(axis=0)
    x_tok, y_tok = get_2d(X_emb)
    set_limits(ax1, 'token', x_tok, y_tok)
    fs_tok = get_fontsize(vocab_size)
    for i in range(vocab_size):
        ax1.text(x_tok[i], y_tok[i], itos[i], fontsize=fs_tok, fontweight='bold',
                ha='center', va='center', color=token_colors[i])
    ax1.set_title("Token Embeddings", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Dim 0")
    ax1.set_ylabel("Dim 1")
    ax1.grid(True, alpha=0.3)
    
    # Position embeddings
    ax2 = fig.add_subplot(3, 3, 2)
    X_pos = pos_emb_all - pos_emb_all.mean(axis=0)
    x_pos, y_pos = get_2d(X_pos)
    set_limits(ax2, 'position', x_pos, y_pos)
    fs_pos = get_fontsize(block_size)
    for i in range(block_size):
        ax2.text(x_pos[i], y_pos[i], _pos_only_label(i), fontsize=fs_pos, fontweight='bold',
                ha='center', va='center', color=pos_colors[i])
    ax2.set_title("Position Embeddings", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Dim 0")
    ax2.set_ylabel("Dim 1")
    ax2.grid(True, alpha=0.3)
    
    # Token+Position
    ax3 = fig.add_subplot(3, 3, 3)
    x_comb, y_comb = get_2d(all_combinations)
    set_limits(ax3, 'combined', x_comb, y_comb)
    fs_comb = get_fontsize(num_combinations)
    for i in range(num_combinations):
        ax3.text(x_comb[i], y_comb[i], combo_labels[i], fontsize=fs_comb, fontweight='bold',
                ha='center', va='center', color=combo_colors[i])
    ax3.set_title("Token+Position (Sum)", fontsize=14, fontweight='bold')
    ax3.set_xlabel("Dim 0")
    ax3.set_ylabel("Dim 1")
    ax3.grid(True, alpha=0.3)
    
    # Row 2: Q, K, V transformed
    # Q-transformed
    ax4 = fig.add_subplot(3, 3, 4)
    x_q, y_q = get_2d(Q_transformed)
    set_limits(ax4, 'Q', x_q, y_q)
    for i in range(num_combinations):
        ax4.text(x_q[i], y_q[i], combo_labels[i], fontsize=fs_comb, fontweight='bold',
                ha='center', va='center', color='blue', alpha=0.7)
    ax4.set_title("Q-Transformed", fontsize=14, fontweight='bold')
    ax4.set_xlabel("Head Dim 0")
    ax4.set_ylabel("Head Dim 1")
    ax4.grid(True, alpha=0.3)
    
    # K-transformed
    ax5 = fig.add_subplot(3, 3, 5)
    x_k, y_k = get_2d(K_transformed)
    set_limits(ax5, 'K', x_k, y_k)
    for i in range(num_combinations):
        ax5.text(x_k[i], y_k[i], combo_labels[i], fontsize=fs_comb, fontweight='bold',
                ha='center', va='center', color='red', alpha=0.7)
    ax5.set_title("K-Transformed", fontsize=14, fontweight='bold')
    ax5.set_xlabel("Head Dim 0")
    ax5.set_ylabel("Head Dim 1")
    ax5.grid(True, alpha=0.3)
    
    # V-transformed
    ax6 = fig.add_subplot(3, 3, 6)
    x_v, y_v = get_2d(V_transformed)
    set_limits(ax6, 'V', x_v, y_v)
    v_cmap = plt.cm.get_cmap('Greens')
    v_colors = [v_cmap(0.3 + 0.6 * i / max(num_combinations - 1, 1)) for i in range(num_combinations)]
    for i in range(num_combinations):
        ax6.text(x_v[i], y_v[i], combo_labels[i], fontsize=fs_comb, fontweight='bold',
                ha='center', va='center', color=v_colors[i])
    ax6.set_title("V-Transformed", fontsize=14, fontweight='bold')
    ax6.set_xlabel("Head Dim 0")
    ax6.set_ylabel("Head Dim 1")
    ax6.grid(True, alpha=0.3)
    
    # Row 3: Q+K combined, Attention output
    # Q and K together
    ax7 = fig.add_subplot(3, 3, 7)
    # Combine Q and K limits
    all_x = np.concatenate([x_q, x_k])
    all_y = np.concatenate([y_q, y_k])
    set_limits(ax7, 'QK', all_x, all_y)
    # Plot Q in blue
    for i in range(num_combinations):
        ax7.text(x_q[i], y_q[i], combo_labels[i], fontsize=fs_comb, fontweight='bold',
                ha='center', va='center', color='blue', alpha=0.7)
    # Plot K in red
    for i in range(num_combinations):
        ax7.text(x_k[i], y_k[i], combo_labels[i], fontsize=fs_comb, fontweight='bold',
                ha='center', va='center', color='red', alpha=0.7)
    ax7.set_title("Q (blue) and K (red) Together", fontsize=14, fontweight='bold')
    ax7.set_xlabel("Head Dim 0")
    ax7.set_ylabel("Head Dim 1")
    ax7.grid(True, alpha=0.3)
    # Add legend
    ax7.plot([], [], 'o', color='blue', label='Q', markersize=8)
    ax7.plot([], [], 'o', color='red', label='K', markersize=8)
    ax7.legend(loc='upper right', fontsize=10)
    
    # Masked attention output only (causal: pos p attends only to 0..p) — what we actually use
    ax8 = fig.add_subplot(3, 3, 8)
    x_attn_m, y_attn_m = get_2d(attn_outputs_masked)
    set_limits(ax8, 'attn_masked', x_attn_m, y_attn_m)
    purple_cmap = plt.cm.get_cmap('Purples')
    attn_m_colors = [purple_cmap(0.3 + 0.6 * i / max(num_combinations - 1, 1)) for i in range(num_combinations)]
    for i in range(num_combinations):
        ax8.text(x_attn_m[i], y_attn_m[i], combo_labels[i], fontsize=fs_comb, fontweight='bold',
                ha='center', va='center', color=attn_m_colors[i])
    ax8.set_title("Attention Output (causal mask: p→0..p)", fontsize=14, fontweight='bold')
    ax8.set_xlabel("Head Dim 0")
    ax8.set_ylabel("Head Dim 1")
    ax8.grid(True, alpha=0.3)
    
    # Hide unused subplot
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96] if step_label else [0, 0, 1, 1])
    
    if save_path:
        # For static figures, keep tight layout
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()


def plot_tokenpos_qkv_simple(model, itos, save_path=None, fixed_limits=None, step_label: int | None = None):
    """
    Simpler 2x2 figure:
    - Token+position (sum) embedding space
    - Q-transformed space
    - K-transformed space
    - V-transformed space

    This matches the requested \"embeddings, Q, K, V\" learning-dynamics view.
    """
    model.eval()

    # Get embeddings
    embeddings = model.token_embedding.weight.detach().cpu().numpy()
    vocab_size, n_embd = embeddings.shape
    block_size = model.block_size
    pos_emb_all = model.position_embedding_table.weight.detach().cpu().numpy()

    # Create token+position combinations
    num_combinations = vocab_size * block_size
    all_combinations = np.zeros((num_combinations, n_embd))
    combo_labels: list[str] = []
    for token_idx in range(vocab_size):
        for pos_idx in range(block_size):
            idx = token_idx * block_size + pos_idx
            all_combinations[idx] = embeddings[token_idx] + pos_emb_all[pos_idx]
            combo_labels.append(_token_pos_label(itos[token_idx], pos_idx))

    # First head Q/K/V
    head = model.sa_heads.heads[0]
    W_Q = head.query.weight.detach().cpu().numpy()
    W_K = head.key.weight.detach().cpu().numpy()
    W_V = head.value.weight.detach().cpu().numpy()

    Q_transformed = all_combinations @ W_Q.T
    K_transformed = all_combinations @ W_K.T
    V_transformed = all_combinations @ W_V.T

    def get_2d(data):
        if data.shape[1] >= 2:
            return data[:, 0], data[:, 1]
        else:
            return data[:, 0], np.zeros(len(data))

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    if step_label is not None:
        fig.suptitle(f"Step: {step_label}", fontsize=18, fontweight="bold", y=0.98)

    # Panel 1: token+position embeddings (black)
    ax_emb = axes[0, 0]
    x_e, y_e = get_2d(all_combinations)
    ax_emb.scatter(x_e, y_e, s=0, alpha=0)
    for i, label in enumerate(combo_labels):
        ax_emb.text(x_e[i], y_e[i], label, fontsize=9, ha="center", va="center", color="black")
    ax_emb.set_title("Token+Position embeddings", fontsize=12, fontweight="bold")
    ax_emb.set_xlabel("Dim 0", fontsize=10)
    ax_emb.set_ylabel("Dim 1", fontsize=10)
    ax_emb.grid(True, alpha=0.3)

    # Panel 2: Q (blue)
    ax_q = axes[0, 1]
    x_q, y_q = get_2d(Q_transformed)
    ax_q.scatter(x_q, y_q, s=0, alpha=0)
    for i, label in enumerate(combo_labels):
        ax_q.text(x_q[i], y_q[i], label, fontsize=9, ha="center", va="center", color="blue")
    ax_q.set_title("Q-transformed", fontsize=12, fontweight="bold")
    ax_q.set_xlabel("Head dim 0", fontsize=10)
    ax_q.set_ylabel("Head dim 1", fontsize=10)
    ax_q.grid(True, alpha=0.3)

    # Panel 3: K (red)
    ax_k = axes[1, 0]
    x_k, y_k = get_2d(K_transformed)
    ax_k.scatter(x_k, y_k, s=0, alpha=0)
    for i, label in enumerate(combo_labels):
        ax_k.text(x_k[i], y_k[i], label, fontsize=9, ha="center", va="center", color="red")
    ax_k.set_title("K-transformed", fontsize=12, fontweight="bold")
    ax_k.set_xlabel("Head dim 0", fontsize=10)
    ax_k.set_ylabel("Head dim 1", fontsize=10)
    ax_k.grid(True, alpha=0.3)

    # Panel 4: V (green)
    ax_v = axes[1, 1]
    x_v, y_v = get_2d(V_transformed)
    ax_v.scatter(x_v, y_v, s=0, alpha=0)
    for i, label in enumerate(combo_labels):
        ax_v.text(x_v[i], y_v[i], label, fontsize=9, ha="center", va="center", color="green")
    ax_v.set_title("V-transformed", fontsize=12, fontweight="bold")
    ax_v.set_xlabel("Head dim 0", fontsize=10)
    ax_v.set_ylabel("Head dim 1", fontsize=10)
    ax_v.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96] if step_label is not None else None)

    if save_path:
        # For video frames (step_label not None), avoid bbox_inches='tight' so all frames share the same size.
        if step_label is not None:
            plt.savefig(save_path, dpi=150, facecolor="white")
        else:
            plt.savefig(save_path, bbox_inches="tight", dpi=150, facecolor="white")
        plt.close(fig)
    else:
        plt.show()
    
    model.train()


@torch.no_grad()
def plot_sequence_embeddings(model, X, itos, save_path=None):
    """
    Plot embeddings for a specific sequence showing token embeddings, position embeddings,
    and their combined embeddings in 2D space.
    
    Args:
        model: The model
        X: Input sequence tensor of shape (B, T) - will use first sequence if batch
        itos: Index to string mapping
        save_path: Path to save figure
    """
    model.eval()
    
    # Extract first sequence if batch
    if X.dim() == 2 and X.shape[0] > 1:
        X = X[0:1]  # Take first sequence
    elif X.dim() == 1:
        X = X.unsqueeze(0)  # Add batch dimension
    
    B, T = X.shape
    tokens = [itos[i.item()] for i in X[0]]
    seq_str = " ".join(tokens)
    
    # Get embeddings
    token_emb = model.token_embedding(X)  # (B, T, n_embd)
    positions = torch.arange(T, device=X.device) % model.block_size
    pos_emb = model.position_embedding_table(positions)  # (T, n_embd)
    combined_emb = token_emb + pos_emb  # (B, T, n_embd)
    
    # Convert to numpy
    token_emb_np = token_emb[0].cpu().numpy()  # (T, n_embd)
    pos_emb_np = pos_emb.cpu().numpy()  # (T, n_embd)
    combined_emb_np = combined_emb[0].cpu().numpy()  # (T, n_embd)
    
    n_embd = token_emb_np.shape[1]
    
    # Get all possible embeddings for overlay
    vocab_size = model.token_embedding.weight.shape[0]
    block_size = model.block_size
    all_token_emb = model.token_embedding.weight.detach().cpu().numpy()  # (vocab_size, n_embd)
    all_pos_emb = model.position_embedding_table.weight.detach().cpu().numpy()  # (block_size, n_embd)
    
    # Create all token-position combinations
    all_combined_emb = np.zeros((vocab_size * block_size, n_embd))
    for token_idx in range(vocab_size):
        for pos_idx in range(block_size):
            idx = token_idx * block_size + pos_idx
            all_combined_emb[idx] = all_token_emb[token_idx] + all_pos_emb[pos_idx]
    
    # Create figure: 2 rows, 3 columns with equal row heights
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3, height_ratios=[1, 1])
    
    if n_embd >= 2:
        # Row 1: Heatmaps
        # Column 1: Token embeddings heatmap
        ax1 = fig.add_subplot(gs[0, 0])
        sns.heatmap(token_emb_np, cmap="viridis", xticklabels=list(range(n_embd)),
                    yticklabels=tokens, cbar=True, ax=ax1)
        ax1.set_title("Token Embeddings", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Embedding Dim", fontsize=12)
        ax1.set_ylabel("Token", fontsize=12)
        
        # Column 2: Position embeddings heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        pos_labels = [f"p{i}" for i in range(T)]
        sns.heatmap(pos_emb_np, cmap="viridis", xticklabels=list(range(n_embd)),
                    yticklabels=pos_labels, cbar=True, ax=ax2)
        ax2.set_title("Position Embeddings", fontsize=14, fontweight='bold')
        ax2.set_xlabel("Embedding Dim", fontsize=12)
        ax2.set_ylabel("Position", fontsize=12)
        
        # Column 3: Combined embeddings heatmap
        ax3 = fig.add_subplot(gs[0, 2])
        labels = [_token_pos_label(tokens[i], i) for i in range(T)]
        sns.heatmap(combined_emb_np, cmap="viridis", xticklabels=list(range(n_embd)),
                    yticklabels=labels, cbar=True, ax=ax3)
        ax3.set_title("Token+Position Embeddings", fontsize=14, fontweight='bold')
        ax3.set_xlabel("Embedding Dim", fontsize=12)
        ax3.set_ylabel("Token+Position", fontsize=12)
        
        # Row 2: Scatter plots
        # Calculate consistent axis limits across all three plots for uniform sizing.
        # IMPORTANT: include ALL embeddings (background + sequence) so nothing gets clipped.
        all_x = np.concatenate(
            [
                all_token_emb[:, 0],
                all_pos_emb[:, 0],
                all_combined_emb[:, 0],
                token_emb_np[:, 0],
                pos_emb_np[:, 0],
                combined_emb_np[:, 0],
            ]
        )
        all_y = np.concatenate(
            [
                all_token_emb[:, 1],
                all_pos_emb[:, 1],
                all_combined_emb[:, 1],
                token_emb_np[:, 1],
                pos_emb_np[:, 1],
                combined_emb_np[:, 1],
            ]
        )
        
        x_min, x_max = all_x.min(), all_x.max()
        y_min, y_max = all_y.min(), all_y.max()
        
        # Add padding
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_pad = x_range * 0.1 if x_range > 0 else 0.1
        y_pad = y_range * 0.1 if y_range > 0 else 0.1
        
        x_lim = (x_min - x_pad, x_max + x_pad)
        y_lim = (y_min - y_pad, y_max + y_pad)
        
        # Column 1: Token embeddings scatter
        ax4 = fig.add_subplot(gs[1, 0])
        # Background: ALL token embeddings (annotated, shaded out)
        for token_idx in range(vocab_size):
            token_str = str(itos[token_idx])
            ax4.text(
                all_token_emb[token_idx, 0],
                all_token_emb[token_idx, 1],
                token_str,
                fontsize=14,
                alpha=0.5,
                ha="center",
                va="center",
                color="dimgray",
                zorder=1,
            )
        # Foreground: tokens from THIS sequence (unchanged labels)
        ax4.scatter(token_emb_np[:, 0], token_emb_np[:, 1], s=0, alpha=0, zorder=2)
        for i, token in enumerate(tokens):
            ax4.text(
                token_emb_np[i, 0],
                token_emb_np[i, 1],
                token,
                fontsize=14,
                fontweight="bold",
                ha="center",
                va="center",
                color="blue",
                zorder=3,
            )
        ax4.set_title("Token Embeddings (2D)", fontsize=14, fontweight='bold')
        ax4.set_xlabel("Dim 0", fontsize=12)
        ax4.set_ylabel("Dim 1", fontsize=12)
        ax4.set_xlim(x_lim)
        ax4.set_ylim(y_lim)
        ax4.grid(True, alpha=0.3)
        # Remove aspect='equal' to allow plots to fill width like heatmaps above
        
        # Column 2: Position embeddings scatter
        ax5 = fig.add_subplot(gs[1, 1])
        # Background: ALL position embeddings (annotated, shaded out)
        for pos_idx in range(block_size):
            ax5.text(
                all_pos_emb[pos_idx, 0],
                all_pos_emb[pos_idx, 1],
                _pos_only_label(pos_idx),
                fontsize=14,
                alpha=0.55,
                ha="center",
                va="center",
                color="dimgray",
                zorder=1,
            )
        # Foreground: positions used in THIS sequence
        ax5.scatter(pos_emb_np[:, 0], pos_emb_np[:, 1], s=0, alpha=0, zorder=2)
        for i in range(T):
            ax5.text(
                pos_emb_np[i, 0],
                pos_emb_np[i, 1],
                f"p{i}",
                fontsize=14,
                fontweight="bold",
                ha="center",
                va="center",
                color="green",
                zorder=3,
            )
        ax5.set_title("Position Embeddings (2D)", fontsize=14, fontweight='bold')
        ax5.set_xlabel("Dim 0", fontsize=12)
        ax5.set_ylabel("Dim 1", fontsize=12)
        ax5.set_xlim(x_lim)
        ax5.set_ylim(y_lim)
        ax5.grid(True, alpha=0.3)
        # Remove aspect='equal' to allow plots to fill width like heatmaps above
        
        # Column 3: Combined embeddings scatter
        ax6 = fig.add_subplot(gs[1, 2])
        # Background: ALL token+position combinations (annotated, shaded out)
        for token_idx in range(vocab_size):
            token_str = str(itos[token_idx])
            for pos_idx in range(block_size):
                idx = token_idx * block_size + pos_idx
                label = _token_pos_label(token_str, pos_idx)
                ax6.text(
                    all_combined_emb[idx, 0],
                    all_combined_emb[idx, 1],
                    label,
                    fontsize=14,
                    alpha=0.4,
                    ha="center",
                    va="center",
                    color="dimgray",
                    zorder=1,
                )
        # Foreground: (token, position) pairs from THIS sequence
        ax6.scatter(combined_emb_np[:, 0], combined_emb_np[:, 1], s=0, alpha=0, zorder=2)
        for i in range(T):
            ax6.text(
                combined_emb_np[i, 0],
                combined_emb_np[i, 1],
                labels[i],
                fontsize=14,
                fontweight="bold",
                ha="center",
                va="center",
                color="purple",
                zorder=3,
            )
        ax6.set_title("Token+Position Embeddings (2D)", fontsize=14, fontweight='bold')
        ax6.set_xlabel("Dim 0", fontsize=12)
        ax6.set_ylabel("Dim 1", fontsize=12)
        ax6.set_xlim(x_lim)
        ax6.set_ylim(y_lim)
        ax6.grid(True, alpha=0.3)
        # Remove aspect='equal' to allow plots to fill width like heatmaps above
        
        # Add sequence as supertitle
        fig.suptitle(f"Sequence Embeddings: {seq_str}", fontsize=16, fontweight='bold', y=0.98)
    else:
        # 1D case - simpler visualization
        fig.suptitle(f"Sequence Embeddings (1D): {seq_str}", fontsize=16, fontweight='bold')
        ax = fig.add_subplot(gs[:, :])
        ax.scatter(combined_emb_np[:, 0], np.zeros(T), s=100)
        for i in range(T):
            ax.text(combined_emb_np[i, 0], 0, labels[i], fontsize=12, ha='center', va='center')
        ax.set_xlabel("Embedding Dim 0", fontsize=12)
        ax.set_yticks([])
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()
        print(f"Sequence embeddings plot saved to {save_path}")
    else:
        plt.show()
    model.train()

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
            labels.append(_token_pos_label(token_str, pos_idx))
    
    # Create BIG figure: 1 row, 2 columns (RAW and PCA)
    fig, axes = plt.subplots(1, 2, figsize=(24, 12))
    
    # Left plot: Original embedding space (Dim 0 vs Dim 1)
    ax1 = axes[0]
    if n_embd >= 2:
        X_orig = all_combinations[:, [0, 1]]
        ax1.scatter(X_orig[:, 0], X_orig[:, 1], s=0, alpha=0)  # s=0 for invisible dots
        for i in range(len(labels)):
            ax1.text(X_orig[i, 0], X_orig[i, 1], labels[i], fontsize=9, ha='center', va='center')
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
            ax1.text(X_orig_1d[i], 0, labels[i], fontsize=9, ha='center', va='center', rotation=90)
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
            ax2.text(X2_comb[i, 0], X2_comb[i, 1], labels[i], fontsize=9, ha='center', va='center')
        
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
            ax2.text(X1_comb[i], 0, labels[i], fontsize=9, ha='center', va='center', rotation=90)
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
    
    # Compute QK^T with sqrt(k) normalization (attention formula: Q @ K^T / sqrt(d_k))
    head_size = Q.shape[1]
    QK_T = (Q @ K.T) / np.sqrt(head_size)  # (T, hs) @ (hs, T) = (T, T), scaled by 1/sqrt(d_k)
    
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
            labels.append(_token_pos_label(token_str, pos_idx))
    
    # Transform by Q, K, V
    # all_combinations is (N, n_embd), W_Q is (hs, n_embd)
    # Transform: (N, n_embd) @ (n_embd, hs) = (N, hs) -> transpose W_Q first
    Q_transformed = all_combinations @ W_Q.T  # (N, hs)
    K_transformed = all_combinations @ W_K.T  # (N, hs)
    V_transformed = all_combinations @ W_V.T  # (N, hs)
    
    # Create figure: 2 rows, 4 columns
    # Row 1: Original token+position embeddings + W_Q, W_K, W_V
    # Row 2: Q, K, V transformed embeddings (aligned under W_Q/W_K/W_V)
    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.3)
    
    # Row 1, Col 1: Original token+position embeddings
    ax0 = fig.add_subplot(gs[0, 0])
    if n_embd >= 2:
        # Plot original embeddings in first 2 dimensions
        X_orig = all_combinations[:, [0, 1]]
        ax0.scatter(X_orig[:, 0], X_orig[:, 1], s=0, alpha=0)
        for i in range(len(labels)):
            ax0.text(X_orig[i, 0], X_orig[i, 1], labels[i], fontsize=9, ha='center', va='center')
        ax0.set_title(f"Original Token+Position Embeddings: Dim 0 vs Dim 1\n(All tokens, {num_combinations} combinations)", fontsize=12)
        ax0.set_xlabel("Embedding Dim 0")
        ax0.set_ylabel("Embedding Dim 1")
        ax0.grid(True, alpha=0.3)
        ax0.set_aspect('equal', adjustable='box')
        ax0.set_xlim(left=-5)
    else:
        # 1D case
        X_orig_1d = all_combinations[:, 0]
        ax0.scatter(X_orig_1d, np.zeros_like(X_orig_1d), s=0, alpha=0)
        for i in range(len(labels)):
            ax0.text(X_orig_1d[i], 0, labels[i], fontsize=9, ha='center', va='center', rotation=90)
        ax0.set_title(f"Original Token+Position Embeddings: Dim 0\n(All tokens, {num_combinations} combinations)", fontsize=12)
        ax0.set_xlabel("Embedding Dim 0")
        ax0.set_ylabel("")
        ax0.grid(True, alpha=0.3)
        ax0.set_yticks([])
    
    # Row 1: QKV weights
    # W_Q
    ax1 = fig.add_subplot(gs[0, 1])
    x_labels = list(range(n_embd))
    y_labels_local = list(range(head_size))
    sns.heatmap(W_Q, cmap="viridis", xticklabels=x_labels, yticklabels=y_labels_local, cbar=True, ax=ax1)
    ax1.set_title(f"W_Q (hs×C={head_size}×{n_embd})", fontsize=12)
    ax1.set_xlabel("C (embedding dim)")
    ax1.set_ylabel("hs (head_size)")
    
    # W_K
    ax2 = fig.add_subplot(gs[0, 2])
    sns.heatmap(W_K, cmap="viridis", xticklabels=x_labels, yticklabels=y_labels_local, cbar=True, ax=ax2)
    ax2.set_title(f"W_K (hs×C={head_size}×{n_embd})", fontsize=12)
    ax2.set_xlabel("C (embedding dim)")
    ax2.set_ylabel("hs (head_size)")
    
    # W_V
    ax3 = fig.add_subplot(gs[0, 3])
    sns.heatmap(W_V, cmap="viridis", xticklabels=x_labels, yticklabels=y_labels_local, cbar=True, ax=ax3)
    ax3.set_title(f"W_V (hs×C={head_size}×{n_embd})", fontsize=12)
    ax3.set_xlabel("C (embedding dim)")
    ax3.set_ylabel("hs (head_size)")
    
    # Row 2: Transformed token-position combinations
    ax_empty = fig.add_subplot(gs[1, 0])
    ax_empty.axis("off")

    # Q-transformed
    ax4 = fig.add_subplot(gs[1, 1])
    if head_size >= 2:
        Q_2d = Q_transformed[:, [0, 1]]
        ax4.scatter(Q_2d[:, 0], Q_2d[:, 1], s=0, alpha=0)
        for i in range(len(labels)):
            ax4.text(Q_2d[i, 0], Q_2d[i, 1], labels[i], fontsize=10, ha='center', va='center', color='blue')
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
            ax4.text(Q_1d[i], 0, labels[i], fontsize=10, ha='center', va='center', rotation=90, color='blue')
        ax4.set_title(f"Q-Transformed: Dim 0\n(All tokens, {num_combinations} combinations)", fontsize=12)
        ax4.set_xlabel("Head Dim 0")
        ax4.set_ylabel("")
        ax4.grid(True, alpha=0.3)
        ax4.set_yticks([])
    
    # K-transformed
    ax5 = fig.add_subplot(gs[1, 2])
    if head_size >= 2:
        K_2d = K_transformed[:, [0, 1]]
        ax5.scatter(K_2d[:, 0], K_2d[:, 1], s=0, alpha=0)
        for i in range(len(labels)):
            ax5.text(K_2d[i, 0], K_2d[i, 1], labels[i], fontsize=10, ha='center', va='center', color='red')
        ax5.set_title(f"K-Transformed: Dim 0 vs Dim 1\n(All tokens, {num_combinations} combinations)", fontsize=12)
        ax5.set_xlabel("Head Dim 0")
        ax5.set_ylabel("Head Dim 1")
        ax5.grid(True, alpha=0.3)
        ax5.axis('equal')
    else:
        K_1d = K_transformed[:, 0]
        ax5.scatter(K_1d, np.zeros_like(K_1d), s=0, alpha=0)
        for i in range(len(labels)):
            ax5.text(K_1d[i], 0, labels[i], fontsize=10, ha='center', va='center', rotation=90, color='red')
        ax5.set_title(f"K-Transformed: Dim 0\n(All tokens, {num_combinations} combinations)", fontsize=12)
        ax5.set_xlabel("Head Dim 0")
        ax5.set_ylabel("")
        ax5.grid(True, alpha=0.3)
        ax5.set_yticks([])
    
    # V-transformed
    ax6 = fig.add_subplot(gs[1, 3])
    # V colors: green spectrum
    v_cmap = plt.cm.get_cmap('Greens')
    v_colors = [v_cmap(0.3 + 0.6 * i / max(num_combinations - 1, 1)) for i in range(num_combinations)]
    
    if head_size >= 2:
        V_2d = V_transformed[:, [0, 1]]
        margin = 0.15 * max(V_2d[:, 0].max() - V_2d[:, 0].min(), V_2d[:, 1].max() - V_2d[:, 1].min())
        ax6.set_xlim(V_2d[:, 0].min() - margin, V_2d[:, 0].max() + margin)
        ax6.set_ylim(V_2d[:, 1].min() - margin, V_2d[:, 1].max() + margin)
        for i in range(len(labels)):
            ax6.text(V_2d[i, 0], V_2d[i, 1], labels[i], fontsize=14, fontweight='bold', 
                    ha='center', va='center', color=v_colors[i])
        ax6.set_title(f"V-Transformed: Dim 0 vs Dim 1\n(All tokens, {num_combinations} combinations)", fontsize=12, fontweight='bold')
        ax6.set_xlabel("Head Dim 0")
        ax6.set_ylabel("Head Dim 1")
        ax6.grid(True, alpha=0.3)
    else:
        V_1d = V_transformed[:, 0]
        margin = 0.15 * (V_1d.max() - V_1d.min())
        ax6.set_xlim(V_1d.min() - margin, V_1d.max() + margin)
        ax6.set_ylim(-0.5, 0.5)
        for i in range(len(labels)):
            ax6.text(V_1d[i], 0, labels[i], fontsize=14, fontweight='bold', 
                    ha='center', va='center', color=v_colors[i])
        ax6.set_title(f"V-Transformed: Dim 0\n(All tokens, {num_combinations} combinations)", fontsize=12, fontweight='bold')
        ax6.set_xlabel("Head Dim 0")
        ax6.set_ylabel("")
        ax6.grid(True, alpha=0.3)
        ax6.set_yticks([])
    
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
        
        # Compute QK^T with sqrt(k) normalization (attention formula: Q @ K^T / sqrt(d_k))
        head_size = Q.shape[1]
        QK_T = (Q @ K.T) / np.sqrt(head_size)  # (T, hs) @ (hs, T) = (T, T), scaled by 1/sqrt(d_k)
        
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
    
    # Get all possible Q/K combinations for overlay
    vocab_size = model.token_embedding.weight.shape[0]
    block_size = model.block_size
    head = model.sa_heads.heads[0]  # Use first head for overlay
    W_Q = head.query.weight.detach().cpu().numpy()  # (head_size, n_embd)
    W_K = head.key.weight.detach().cpu().numpy()  # (head_size, n_embd)
    all_token_emb = model.token_embedding.weight.detach().cpu().numpy()  # (vocab_size, n_embd)
    all_pos_emb = model.position_embedding_table.weight.detach().cpu().numpy()  # (block_size, n_embd)
    
    # Create all Q and K combinations
    all_Q_combinations = []
    all_K_combinations = []
    all_QK_labels = []
    for token_idx in range(vocab_size):
        token_str = str(itos[token_idx])
        for pos_idx in range(block_size):
            combined_emb = all_token_emb[token_idx] + all_pos_emb[pos_idx]
            q_vec = W_Q @ combined_emb  # (head_size,)
            k_vec = W_K @ combined_emb  # (head_size,)
            all_Q_combinations.append(q_vec)
            all_K_combinations.append(k_vec)
            all_QK_labels.append(_token_pos_label(token_str, pos_idx))
    
    all_Q_combinations = np.array(all_Q_combinations)  # (vocab_size * block_size, head_size)
    all_K_combinations = np.array(all_K_combinations)  # (vocab_size * block_size, head_size)
    
    # ========== PLOT 1: Q, K, masked QK^T, Attention, scatter(Q vs K) ==========
    use_two_rows = num_sequences == 1
    if use_two_rows:
        n_rows_1, n_cols_plot1 = 2, 3  # Row0: Q, K, Q vs K; Row1: masked QK^T, Attention
        fig1 = plt.figure(figsize=(6 * n_cols_plot1, 5 * n_rows_1))
        gs1 = GridSpec(n_rows_1, n_cols_plot1, figure=fig1, hspace=0.35, wspace=0.3)
    else:
        num_cols_plot1 = 5
        fig1 = plt.figure(figsize=(6 * num_cols_plot1, 4 * num_sequences))
        gs1 = GridSpec(num_sequences, num_cols_plot1, figure=fig1, hspace=0.4, wspace=0.3)
    
    # Collect all Q and K from all sequences to compute consistent PCA
    all_Q_data = []
    all_K_data = []
    for data_dict in all_data:
        all_Q_data.append(data_dict['Q'])
        all_K_data.append(data_dict['K'])
    all_Q_data = np.vstack(all_Q_data)  # (total_T, head_size)
    all_K_data = np.vstack(all_K_data)  # (total_T, head_size)
    
    # Compute PCA transformation from ALL sequence data (to apply consistently)
    combined_QK_for_pca = np.vstack([all_Q_data, all_K_data, all_Q_combinations, all_K_combinations])
    pca_transform = None
    if combined_QK_for_pca.shape[1] > 2:
        # Compute PCA transformation from all data
        data_centered = combined_QK_for_pca - combined_QK_for_pca.mean(axis=0, keepdims=True)
        U, s, Vt = np.linalg.svd(data_centered, full_matrices=False)
        pca_transform = Vt[:2].T  # (head_size, 2)
        # Apply PCA to all combinations
        all_Q_centered = all_Q_combinations - combined_QK_for_pca.mean(axis=0, keepdims=True)
        all_K_centered = all_K_combinations - combined_QK_for_pca.mean(axis=0, keepdims=True)
        all_Q_2d_overlay = all_Q_centered @ pca_transform
        all_K_2d_overlay = all_K_centered @ pca_transform
    else:
        # Use raw dimensions
        all_Q_2d_overlay = all_Q_combinations[:, :2] if all_Q_combinations.shape[1] >= 2 else pca_2d(all_Q_combinations)
        all_K_2d_overlay = all_K_combinations[:, :2] if all_K_combinations.shape[1] >= 2 else pca_2d(all_K_combinations)
    
    for data_dict in all_data:
        seq_idx = data_dict['seq_idx']
        tokens = data_dict['tokens']
        seq_str = data_dict['seq_str']
        Q = data_dict['Q']
        K = data_dict['K']
        Masked_QK_T = data_dict['Masked_QK_T']
        Attention = data_dict['Attention']
        T = data_dict['T']
        if use_two_rows:
            r0, r1, c_q, c_k, c_scat, c_masked, c_att = 0, 1, 0, 1, 2, 0, 1
        else:
            r0 = r1 = seq_idx
            c_q, c_k, c_scat, c_masked, c_att = 0, 1, 2, 3, 4

        # Q
        ax = fig1.add_subplot(gs1[r0, c_q])
        dim_str = f"(T×hs={Q.shape[0]}×{Q.shape[1]})"
        sns.heatmap(Q, cmap="viridis", xticklabels=list(range(Q.shape[1])), 
                   yticklabels=tokens, cbar=True, ax=ax)
        ax.set_xlabel("hs", fontsize=10)
        ax.set_ylabel(f"Seq {seq_idx+1}\n{seq_str}\n" if not use_two_rows else f"{seq_str}", fontsize=9)
        ax.set_title(f"Q {dim_str}", fontsize=11)
        
        # K
        ax = fig1.add_subplot(gs1[r0, c_k])
        dim_str = f"(T×hs={K.shape[0]}×{K.shape[1]})"
        sns.heatmap(K, cmap="viridis", xticklabels=list(range(K.shape[1])), 
                   yticklabels=tokens, cbar=True, ax=ax)
        ax.set_xlabel("hs", fontsize=10)
        ax.set_ylabel("T", fontsize=10)
        ax.set_title(f"K {dim_str}", fontsize=11)
        
        # Scatter plot Q vs K
        ax = fig1.add_subplot(gs1[r0, c_scat])
        
        # Apply consistent PCA transformation to sequence-specific Q/K
        if pca_transform is not None:
            Q_centered = Q - combined_QK_for_pca.mean(axis=0, keepdims=True)
            K_centered = K - combined_QK_for_pca.mean(axis=0, keepdims=True)
            Q_2d = Q_centered @ pca_transform
            K_2d = K_centered @ pca_transform
        else:
            Q_2d = pca_2d(Q)
            K_2d = pca_2d(K)
        
        # Combine Q and K data (including overlay) to set axis limits properly
        all_data_2d = np.vstack([Q_2d, K_2d, all_Q_2d_overlay, all_K_2d_overlay])
        x_min, x_max = all_data_2d[:, 0].min(), all_data_2d[:, 0].max()
        y_min, y_max = all_data_2d[:, 1].min(), all_data_2d[:, 1].max()
        x_margin = (x_max - x_min) * 0.1
        y_margin = (y_max - y_min) * 0.1
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        
        # Background overlay: ALL Q/K combinations (annotated, lighter grey)
        # Draw overlay FIRST so it's underneath
        for i, label in enumerate(all_QK_labels):
            x_q, y_q = all_Q_2d_overlay[i, 0], all_Q_2d_overlay[i, 1]
            x_k, y_k = all_K_2d_overlay[i, 0], all_K_2d_overlay[i, 1]
            # Check if points are within axis limits before drawing
            if x_min - x_margin <= x_q <= x_max + x_margin and y_min - y_margin <= y_q <= y_max + y_margin:
                ax.text(x_q, y_q, label,
                       fontsize=11, alpha=0.7, ha='center', va='center', 
                       color='#808080', zorder=1)  # Lighter grey
            if x_min - x_margin <= x_k <= x_max + x_margin and y_min - y_margin <= y_k <= y_max + y_margin:
                ax.text(x_k, y_k, label,
                       fontsize=11, alpha=0.7, ha='center', va='center', 
                       color='#808080', zorder=1)  # Lighter grey
        
        # Foreground: Sequence-specific Q/K points
        for i, (token, pos) in enumerate(zip(tokens, range(len(tokens)))):
            # Annotate Q points (blue for Query)
            ax.text(Q_2d[i, 0], Q_2d[i, 1], _token_pos_label(token, pos), 
                   fontsize=14, fontweight='bold', ha='center', va='center', color='blue', zorder=3)
            # Annotate K points (red for Key)
            ax.text(K_2d[i, 0], K_2d[i, 1], _token_pos_label(token, pos), 
                   fontsize=14, fontweight='bold', ha='center', va='center', color='red', zorder=3)
        
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
        ax.grid(True, alpha=0.3)
        
        # masked QK^T
        ax = fig1.add_subplot(gs1[r1, c_masked])
        dim_str = f"(T×T={T}×{T})"
        sns.heatmap(Masked_QK_T, cmap="viridis", xticklabels=tokens, 
                   yticklabels=tokens, cbar=True, ax=ax)
        ax.set_xlabel("T", fontsize=10)
        ax.set_ylabel("T", fontsize=10)
        ax.set_title(f"masked QK^T {dim_str}", fontsize=11)
        
        # Attention
        ax = fig1.add_subplot(gs1[r1, c_att])
        dim_str = f"(T×T={T}×{T})"
        sns.heatmap(Attention, cmap="magma", vmin=0.0, vmax=1.0, 
                   xticklabels=tokens, yticklabels=tokens, cbar=True, ax=ax)
        ax.set_xlabel("T", fontsize=10)
        ax.set_ylabel("T", fontsize=10)
        ax.set_title(f"Attention {dim_str}", fontsize=11)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    if save_path:
        # Save directly to the numbered filename (13_qk_attention.png)
        # Extract directory and base name, then construct numbered filename
        import os
        save_dir = os.path.dirname(save_path)
        save_path_part1 = os.path.join(save_dir, "13_qk_attention.png")
        plt.savefig(save_path_part1, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()
    
    # Get all possible V combinations for overlay
    W_V = head.value.weight.detach().cpu().numpy()  # (head_size, n_embd)
    all_V_combinations = []
    all_V_labels = []
    for token_idx in range(vocab_size):
        token_str = str(itos[token_idx])
        for pos_idx in range(block_size):
            combined_emb = all_token_emb[token_idx] + all_pos_emb[pos_idx]
            v_vec = W_V @ combined_emb  # (head_size,)
            all_V_combinations.append(v_vec)
            all_V_labels.append(_token_pos_label(token_str, pos_idx))
    
    all_V_combinations = np.array(all_V_combinations)  # (vocab_size * block_size, head_size)
    
    # ========== PLOT 2: Attention, V, Final Output, scatter(V), scatter(Final Output) ==========
    # Final Output = Attention @ V: weighted sum of value vectors where attention weights determine
    # how much each position contributes. For position i: Final_Output[i] = sum_j(Attention[i,j] * V[j])
    use_two_rows_2 = num_sequences == 1
    if use_two_rows_2:
        n_rows_2, n_cols_plot2 = 2, 3  # Row0: Attention, V, Final Output; Row1: V scatter, Final scatter
        fig2 = plt.figure(figsize=(6 * n_cols_plot2, 5 * n_rows_2))
        gs2 = GridSpec(n_rows_2, n_cols_plot2, figure=fig2, hspace=0.35, wspace=0.3)
    else:
        num_cols_plot2 = 5
        fig2 = plt.figure(figsize=(6 * num_cols_plot2, 4 * num_sequences))
        gs2 = GridSpec(num_sequences, num_cols_plot2, figure=fig2, hspace=0.4, wspace=0.3)
    
    # First pass: collect all V data to compute consistent PCA
    all_V_data = []
    for data_dict in all_data:
        all_V_data.append(data_dict['V'])
    all_V_data = np.vstack(all_V_data)  # (total_T, head_size)
    
    # Compute PCA transformation from ALL V data (to apply consistently)
    combined_V_for_pca = np.vstack([all_V_data, all_V_combinations])
    pca_transform_V = None
    if combined_V_for_pca.shape[1] > 2:
        # Compute PCA transformation from all data
        data_centered = combined_V_for_pca - combined_V_for_pca.mean(axis=0, keepdims=True)
        U, s, Vt = np.linalg.svd(data_centered, full_matrices=False)
        pca_transform_V = Vt[:2].T
        # Apply PCA to all combinations
        all_V_centered = all_V_combinations - combined_V_for_pca.mean(axis=0, keepdims=True)
        all_V_2d_overlay = all_V_centered @ pca_transform_V
    else:
        all_V_2d_overlay = all_V_combinations[:, :2] if all_V_combinations.shape[1] >= 2 else pca_2d(all_V_combinations)
    
    # Collect all scatter data to calculate shared axis limits
    all_V_2d = []
    all_Final_Output_2d = []
    for data_dict in all_data:
        V = data_dict['V']
        Final_Output = data_dict['Final_Output']
        # Apply consistent PCA transformation
        if pca_transform_V is not None:
            V_centered = V - combined_V_for_pca.mean(axis=0, keepdims=True)
            V_2d = V_centered @ pca_transform_V
        else:
            V_2d = pca_2d(V)
        Final_Output_2d = pca_2d(Final_Output)
        all_V_2d.append(V_2d)
        all_Final_Output_2d.append(Final_Output_2d)
    
    # Calculate shared axis limits from all scatter data (including overlay)
    all_scatter_data = np.vstack(all_V_2d + all_Final_Output_2d + [all_V_2d_overlay])
    x_min_shared, x_max_shared = all_scatter_data[:, 0].min(), all_scatter_data[:, 0].max()
    y_min_shared, y_max_shared = all_scatter_data[:, 1].min(), all_scatter_data[:, 1].max()
    x_range_shared = x_max_shared - x_min_shared if x_max_shared != x_min_shared else 1.0
    y_range_shared = y_max_shared - y_min_shared if y_max_shared != y_min_shared else 1.0
    padding = 0.1
    xlim_shared = (x_min_shared - padding * x_range_shared, x_max_shared + padding * x_range_shared)
    ylim_shared = (y_min_shared - padding * y_range_shared, y_max_shared + padding * y_range_shared)
    
    # Second pass: create plots with shared axis limits
    for idx, data_dict in enumerate(all_data):
        seq_idx = data_dict['seq_idx']
        tokens = data_dict['tokens']
        seq_str = data_dict['seq_str']
        V = data_dict['V']
        Attention = data_dict['Attention']
        Final_Output = data_dict['Final_Output']
        T = data_dict['T']
        if use_two_rows_2:
            r0_2, r1_2 = 0, 1
            c_att2, c_v2, c_final2, c_vscat2, c_finalscat2 = 0, 1, 2, 0, 1
        else:
            r0_2 = r1_2 = seq_idx
            c_att2, c_v2, c_final2, c_vscat2, c_finalscat2 = 0, 1, 2, 3, 4
        
        # Get pre-computed 2D projections
        V_2d = all_V_2d[idx]
        Final_Output_2d = all_Final_Output_2d[idx]
        
        # Attention
        ax = fig2.add_subplot(gs2[r0_2, c_att2])
        dim_str = f"(T×T={T}×{T})"
        sns.heatmap(Attention, cmap="magma", vmin=0.0, vmax=1.0, 
                   xticklabels=tokens, yticklabels=tokens, cbar=True, ax=ax)
        ax.set_xlabel("T", fontsize=10)
        ax.set_ylabel(f"Seq {seq_idx+1}\n{seq_str}\n" if not use_two_rows_2 else seq_str, fontsize=9)
        ax.set_title(f"Attention {dim_str}", fontsize=11)
        
        # V
        ax = fig2.add_subplot(gs2[r0_2, c_v2])
        dim_str = f"(T×hs={V.shape[0]}×{V.shape[1]})"
        sns.heatmap(V, cmap="viridis", xticklabels=list(range(V.shape[1])), 
                   yticklabels=tokens, cbar=True, ax=ax)
        ax.set_xlabel("hs", fontsize=10)
        ax.set_ylabel("T", fontsize=10)
        ax.set_title(f"V {dim_str}", fontsize=11)
        
        # Final Output (Attention @ V)
        ax = fig2.add_subplot(gs2[r0_2, c_final2])
        dim_str = f"(T×hs={Final_Output.shape[0]}×{Final_Output.shape[1]})"
        sns.heatmap(Final_Output, cmap="viridis", xticklabels=list(range(Final_Output.shape[1])), 
                   yticklabels=tokens, cbar=True, ax=ax)
        ax.set_xlabel("hs", fontsize=10)
        ax.set_ylabel("T", fontsize=10)
        ax.set_title(f"Final Output (Attention@V) {dim_str}", fontsize=11)
        
        # Scatter plot for V
        ax = fig2.add_subplot(gs2[r1_2, c_vscat2])
        
        # Use shared axis limits
        ax.set_xlim(xlim_shared)
        ax.set_ylim(ylim_shared)
        
        # Background overlay: ALL V combinations (annotated, lighter grey)
        # Draw overlay FIRST so it's underneath
        for i, label in enumerate(all_V_labels):
            # Check if point is within axis limits before drawing
            x, y = all_V_2d_overlay[i, 0], all_V_2d_overlay[i, 1]
            if xlim_shared[0] <= x <= xlim_shared[1] and ylim_shared[0] <= y <= ylim_shared[1]:
                ax.text(x, y, label,
                       fontsize=7, alpha=0.7, ha='center', va='center', 
                       color='#808080', zorder=1)  # Lighter grey
        
        # Foreground: Sequence-specific V points
        for i, (token, pos) in enumerate(zip(tokens, range(len(tokens)))):
            ax.text(V_2d[i, 0], V_2d[i, 1], _token_pos_label(token, pos), 
                   fontsize=9, fontweight='bold', ha='center', va='center', color='blue', zorder=3)
        
        # Update axis labels based on whether PCA was used
        if V.shape[1] > 2:
            ax.set_xlabel("PC1", fontsize=10)
            ax.set_ylabel("PC2", fontsize=10)
            title_suffix = " (PCA)"
        else:
            ax.set_xlabel("Dim 1", fontsize=10)
            ax.set_ylabel("Dim 2", fontsize=10)
            title_suffix = " (raw)"
        ax.set_title(f"V{title_suffix}", fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Scatter plot for Final Output
        ax = fig2.add_subplot(gs2[r1_2, c_finalscat2])
        
        # Use shared axis limits
        ax.set_xlim(xlim_shared)
        ax.set_ylim(ylim_shared)
        
        # Annotate Final Output points with token and position (no scatter points)
        for i, (token, pos) in enumerate(zip(tokens, range(len(tokens)))):
            ax.text(Final_Output_2d[i, 0], Final_Output_2d[i, 1], _token_pos_label(token, pos), 
                   fontsize=9, fontweight='bold', ha='center', va='center', color='red')
        
        # Update axis labels based on whether PCA was used
        if Final_Output.shape[1] > 2:
            ax.set_xlabel("PC1", fontsize=10)
            ax.set_ylabel("PC2", fontsize=10)
            title_suffix = " (PCA)"
        else:
            ax.set_xlabel("Dim 1", fontsize=10)
            ax.set_ylabel("Dim 2", fontsize=10)
            title_suffix = " (raw)"
        ax.set_title(f"Final Output (Attention@V){title_suffix}", fontsize=11)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    if save_path:
        # Save directly to the numbered filename (14_value_output.png)
        # Extract directory and base name, then construct numbered filename
        import os
        save_dir = os.path.dirname(save_path)
        save_path_part2 = os.path.join(save_dir, "14_value_output.png")
        plt.savefig(save_path_part2, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()
    
    model.train()

@torch.no_grad()
def plot_residuals(model, X_list, itos, save_path=None, num_sequences=3):
    """
    Plot residuals: V transformed (after attention), Embeddings (token+position), and Sum (residual connection).
    Shows heatmaps and scatter plots for all three, plus embeddings as arrows from origin.
    
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
    
    # Helper function to compute PCA for 2D visualization
    def pca_2d(data):
        """Reduce data to 2D using PCA only when dimension > 2"""
        if data.shape[1] <= 2:
            if data.shape[1] == 2:
                return data
            elif data.shape[1] == 1:
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
        
        # Get embeddings (token + position)
        token_emb = model.token_embedding(X)  # (B, T, n_embd)
        pos = torch.arange(T, device=X.device) % model.block_size
        pos_emb = model.position_embedding_table(pos)  # (T, n_embd)
        embeddings = (token_emb + pos_emb)[0].cpu().numpy()  # (T, n_embd)
        
        # Get actual attention output from model (this is what gets added in residual connection)
        x = token_emb + pos_emb
        attn_out, _ = model.sa_heads(x)  # (B, T, n_embd) - concatenated across heads
        V_transformed = attn_out[0].cpu().numpy()  # (T, n_embd)
        
        # Also get V and Attention for visualization (averaged across heads)
        v_all = []
        wei_all = []
        for h in model.sa_heads.heads:
            v_all.append(h.value(x)[0].cpu().numpy())  # (T, hs)
            _, wei = h(x)  # wei: (B, T, T)
            wei_all.append(wei[0].cpu().numpy())  # (T, T)
        
        # Get Sum (residual connection: embeddings + V_transformed)
        # V_transformed is already (T, n_embd) from the model output
        Sum = embeddings + V_transformed  # (T, n_embd)
        
        all_data.append({
            'tokens': tokens,
            'seq_str': seq_str,
            'seq_idx': seq_idx,
            'embeddings': embeddings,
            'V_transformed': V_transformed,
            'Sum': Sum,
            'T': T
        })
    
    # Create figure layout.
    # When single sequence: 2 rows x 4 cols.
    #   Col 0: Embed hm (r0) + scatter (r1). Col 1: V_trans hm (r0) + scatter (r1).
    #   Col 2: Final hm (r0) + Embed→Final arrows (r1).
    #   Col 3: empty (r0), Final scatter (r1) — normal 1x1 next to arrows.
    use_two_rows_r = num_sequences == 1
    if use_two_rows_r:
        n_rows_r, num_cols = 2, 4
        fig = plt.figure(figsize=(4 * num_cols, 5 * n_rows_r))
        gs = GridSpec(n_rows_r, num_cols, figure=fig, hspace=0.4, wspace=0.3)
    else:
        num_cols = 7
        fig = plt.figure(figsize=(6 * num_cols, 4 * num_sequences))
        gs = GridSpec(num_sequences, num_cols, figure=fig, hspace=0.4, wspace=0.3)
    
    # Create consistent color mapping for each token+position combination
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    # Get all unique token+position combinations across all sequences
    all_token_pos = set()
    for data_dict in all_data:
        for token, pos in zip(data_dict['tokens'], range(len(data_dict['tokens']))):
            all_token_pos.add((token, pos))
    # Create color map - use tab20 for more distinct colors, cycle if needed
    num_unique = len(all_token_pos)
    cmap = cm.get_cmap('tab20')
    if num_unique > 20:
        # Use a larger colormap if needed
        cmap = cm.get_cmap('tab20')
        colors_list = [cmap(i % 20) for i in range(num_unique)]
    else:
        colors_list = [cmap(i) for i in range(num_unique)]
    
    # Create mapping: (token, pos) -> color
    token_pos_to_color = {}
    for idx, (token, pos) in enumerate(sorted(all_token_pos)):
        token_pos_to_color[(token, pos)] = colors_list[idx]
    
    # First pass: collect all scatter data to calculate shared axis limits
    all_scatter_data = []
    for data_dict in all_data:
        V_transformed_2d = pca_2d(data_dict['V_transformed'])
        embeddings_2d = pca_2d(data_dict['embeddings'])  # Needed for arrows column
        Sum_2d = pca_2d(data_dict['Sum'])
        all_scatter_data.append(V_transformed_2d)
        all_scatter_data.append(embeddings_2d)
        all_scatter_data.append(Sum_2d)
    
    # Calculate shared axis limits
    all_scatter_stacked = np.vstack(all_scatter_data)
    x_min_shared, x_max_shared = all_scatter_stacked[:, 0].min(), all_scatter_stacked[:, 0].max()
    y_min_shared, y_max_shared = all_scatter_stacked[:, 1].min(), all_scatter_stacked[:, 1].max()
    x_range_shared = x_max_shared - x_min_shared if x_max_shared != x_min_shared else 1.0
    y_range_shared = y_max_shared - y_min_shared if y_max_shared != y_min_shared else 1.0
    padding = 0.1
    xlim_shared = (x_min_shared - padding * x_range_shared, x_max_shared + padding * x_range_shared)
    ylim_shared = (y_min_shared - padding * y_range_shared, y_max_shared + padding * y_range_shared)
    
    # Second pass: create plots
    for data_dict in all_data:
        seq_idx = data_dict['seq_idx']
        tokens = data_dict['tokens']
        seq_str = data_dict['seq_str']
        embeddings = data_dict['embeddings']
        V_transformed = data_dict['V_transformed']
        Sum = data_dict['Sum']
        T = data_dict['T']
        if use_two_rows_r:
            # Col 0: Embed (hm+sc). Col 1: V_trans (hm+sc). Col 2: Final hm + arrows. Col 3: empty r0, Final sc r1.
            r0_r, r1_r = 0, 1
            c_emb_hm, c_v_hm, c_emb_sc, c_v_sc = 0, 1, 0, 1
            c_final_hm = 2       # Final heatmap in col 2, row 0
            c_final_arrow = 2    # Embed→Final arrows in col 2, row 1
            c_final_sc = 3      # Final scatter in col 3, row 1 (next to arrows; r0 col 3 left empty)
        else:
            r0_r = r1_r = r2_r = seq_idx
            c_emb_hm, c_v_hm, c_sum_hm, c_emb_sc, c_v_sc, c_arrow, c_fin_sc = 0, 1, 2, 3, 4, 5, 6
            c_final_span = 2  # unused in multi-seq
        
        # Embeddings heatmap (row 0)
        ax = fig.add_subplot(gs[r0_r, c_emb_hm])
        dim_str = f"(T×d={T}×{embeddings.shape[1]})"
        sns.heatmap(embeddings, cmap="RdBu_r", center=0,
                   xticklabels=False, yticklabels=tokens, cbar=True, ax=ax)
        ax.set_xlabel("Dim", fontsize=10)
        ax.set_ylabel(seq_str if use_two_rows_r else f"Seq {seq_idx+1}\n{seq_str}\n", fontsize=9)
        ax.set_title(f"Embeddings (Token+Pos) {dim_str}", fontsize=11)
        
        # V Transformed heatmap
        ax = fig.add_subplot(gs[r0_r, c_v_hm])
        dim_str = f"(T×d={T}×{V_transformed.shape[1]})"
        sns.heatmap(V_transformed, cmap="RdBu_r", center=0, 
                   xticklabels=False, yticklabels=tokens, cbar=True, ax=ax)
        ax.set_xlabel("Dim", fontsize=10)
        ax.set_ylabel(seq_str if use_two_rows_r else f"Seq {seq_idx+1}\n{seq_str}\n", fontsize=9)
        ax.set_title(f"V Transformed (Attention@V) {dim_str}", fontsize=11)
        
        # Final heatmap (col 2, row 0 when single seq)
        ax = fig.add_subplot(gs[r0_r, c_final_hm] if use_two_rows_r else gs[r0_r, c_sum_hm])
        dim_str = f"(T×d={T}×{Sum.shape[1]})"
        sns.heatmap(Sum, cmap="RdBu_r", center=0,
                   xticklabels=False, yticklabels=tokens, cbar=True, ax=ax)
        ax.set_xlabel("Dim", fontsize=10)
        ax.set_ylabel(seq_str if use_two_rows_r else f"Seq {seq_idx+1}\n{seq_str}\n", fontsize=9)
        ax.set_title(f"Final (Embed+V_transformed) {dim_str}", fontsize=11)
        
        # Embeddings scatter (row 1, under Embed heatmap)
        ax = fig.add_subplot(gs[r1_r, c_emb_sc])
        embeddings_2d = pca_2d(embeddings)
        # Mark origin with faint dashed lines
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=5)
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=5)
        for i, (token, pos) in enumerate(zip(tokens, range(len(tokens)))):
            color = token_pos_to_color[(token, pos)]
            ax.text(embeddings_2d[i, 0], embeddings_2d[i, 1], _token_pos_label(token, pos),
                   fontsize=9, fontweight='bold', ha='center', va='center', color=color)
        ax.set_xlim(xlim_shared)
        ax.set_ylim(ylim_shared)
        used_pca = embeddings.shape[1] > 2
        if used_pca:
            ax.set_xlabel("PC1", fontsize=10)
            ax.set_ylabel("PC2", fontsize=10)
            title_suffix = " (PCA)"
        else:
            ax.set_xlabel("Dim 1", fontsize=10)
            ax.set_ylabel("Dim 2", fontsize=10)
            title_suffix = ""
        ax.set_title(f"Embed{title_suffix}", fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # V Transformed scatter (with arrows from origin to each point)
        ax = fig.add_subplot(gs[r1_r, c_v_sc])
        V_transformed_2d = pca_2d(V_transformed)
        # Mark origin with faint dashed lines
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=5)
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=5)
        for i, (token, pos) in enumerate(zip(tokens, range(len(tokens)))):
            color = token_pos_to_color[(token, pos)]
            x, y = V_transformed_2d[i, 0], V_transformed_2d[i, 1]
            # Arrow from origin to this point
            arr_len = np.sqrt(x**2 + y**2)
            head = max(0.12, min(0.25, arr_len * 0.12))
            ax.arrow(0, 0, x, y, head_width=head, head_length=head, fc=color, ec=color, alpha=0.8, length_includes_head=True, width=0.015, zorder=2)
            ax.text(x, y, _token_pos_label(token, pos),
                   fontsize=9, fontweight='bold', ha='center', va='center', color=color, zorder=3)
        ax.set_xlim(xlim_shared)
        ax.set_ylim(ylim_shared)
        used_pca = V_transformed.shape[1] > 2
        if used_pca:
            ax.set_xlabel("PC1", fontsize=10)
            ax.set_ylabel("PC2", fontsize=10)
            title_suffix = " (PCA)"
        else:
            ax.set_xlabel("Dim 1", fontsize=10)
            ax.set_ylabel("Dim 2", fontsize=10)
            title_suffix = ""
        ax.set_title(f"V Transformed{title_suffix}", fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Embeddings → Final arrows (col 2, row 1 when single seq)
        ax = fig.add_subplot(gs[r1_r, c_final_arrow] if use_two_rows_r else gs[r1_r, c_arrow])
        embeddings_2d = pca_2d(embeddings)
        Final_2d = pca_2d(Sum)
        # Mark origin with faint dashed lines
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=5)
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=5)
        # Draw arrows from embeddings to Final points
        for i, (token, pos) in enumerate(zip(tokens, range(len(tokens)))):
            color = token_pos_to_color[(token, pos)]
            # Calculate arrow length for head size scaling
            dx = Final_2d[i, 0] - embeddings_2d[i, 0]
            dy = Final_2d[i, 1] - embeddings_2d[i, 1]
            arrow_length = np.sqrt(dx**2 + dy**2)
            head_size = max(0.15, min(0.3, arrow_length * 0.15))  # Scale head size with arrow length
            # Draw arrow from embeddings to Final
            ax.arrow(embeddings_2d[i, 0], embeddings_2d[i, 1],
                    dx, dy,
                    head_width=head_size, head_length=head_size, fc=color, ec=color, alpha=0.8, length_includes_head=True, width=0.02)
            # Annotate only at beginning (embeddings point)
            ax.text(embeddings_2d[i, 0], embeddings_2d[i, 1], _token_pos_label(token, pos),
                   fontsize=9, fontweight='bold', ha='center', va='center', color=color)
        ax.set_xlim(xlim_shared)
        ax.set_ylim(ylim_shared)
        used_pca = embeddings.shape[1] > 2
        if used_pca:
            ax.set_xlabel("PC1", fontsize=10)
            ax.set_ylabel("PC2", fontsize=10)
            title_suffix = " (PCA)"
        else:
            ax.set_xlabel("Dim 1", fontsize=10)
            ax.set_ylabel("Dim 2", fontsize=10)
            title_suffix = ""
        ax.set_title(f"Embed → Final (Modified by Attention){title_suffix}", fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Final scatter (col 3, row 1 when single seq — next to arrows; space above left empty)
        ax = fig.add_subplot(gs[r1_r, c_final_sc] if use_two_rows_r else gs[r2_r, c_fin_sc])
        Final_2d = pca_2d(Sum)
        # Mark origin with faint dashed lines
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=5)
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=5)
        for i, (token, pos) in enumerate(zip(tokens, range(len(tokens)))):
            color = token_pos_to_color[(token, pos)]
            ax.text(Final_2d[i, 0], Final_2d[i, 1], _token_pos_label(token, pos),
                   fontsize=9, fontweight='bold', ha='center', va='center', color=color)
        ax.set_xlim(xlim_shared)
        ax.set_ylim(ylim_shared)
        used_pca = Sum.shape[1] > 2
        if used_pca:
            ax.set_xlabel("PC1", fontsize=10)
            ax.set_ylabel("PC2", fontsize=10)
            title_suffix = " (PCA)"
        else:
            ax.set_xlabel("Dim 1", fontsize=10)
            ax.set_ylabel("Dim 2", fontsize=10)
            title_suffix = ""
        ax.set_title(f"Final{title_suffix}", fontsize=11)
        ax.grid(True, alpha=0.3)
    
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
    
    # Compute QK^T with sqrt(k) normalization (attention formula: Q @ K^T / sqrt(d_k))
    head_size = Q.shape[1]
    QK_T = (Q @ K.T) / np.sqrt(head_size)  # (T, hs) @ (hs, T) = (T, T), scaled by 1/sqrt(d_k)
    
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
    
    # Compute QK^T with sqrt(k) normalization (attention formula: Q @ K^T / sqrt(d_k))
    head_size = Q.shape[1]
    QK_T = (Q @ K.T) / np.sqrt(head_size)  # (T, hs) @ (hs, T) = (T, T), scaled by 1/sqrt(d_k)
    
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

        # Get attention weights and outputs from all heads
        wei_all = []
        out_all = []
        for h in model.sa_heads.heads:
            out, wei = h(x)  # out: (B, T, head_size), wei: (B, T, T)
            wei_all.append(wei[0].cpu().numpy())  # (T, T)
            out_all.append(out[0].cpu().numpy())  # (T, head_size)
        
        # Average attention across heads
        attention_matrix = np.stack(wei_all, axis=0).mean(axis=0)  # (T, T)
        attention_matrices.append(attention_matrix)
        
        # Concatenate head outputs
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
    Uses the generator's `valence_mask()` to determine which positions are constrained.
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
        valence = generator.valence_mask(generated_integers)
        for i, is_constrained in enumerate(valence):
            if i < len(correctness) and is_constrained:
                total_constrained += 1
                if correctness[i] == 0:
                    incorrect_constrained += 1
    
    model.train()
    return incorrect_constrained / total_constrained if total_constrained > 0 else 0.0


def plot_generated_sequences_heatmap(generated_sequences, generator, save_path=None, num_sequences=5, max_length=30, ax=None, title=None, show_legend=True):
    """
    Plot generated sequences as an annotated heatmap showing correctness.
    Red = incorrect, White/Green = correct.
    """
    sequences_to_show = generated_sequences[:num_sequences]
    
    # Truncate to max_length and pad to same length
    max_len = min(max_length, max(len(seq) for seq in sequences_to_show))
    
    # Create data matrix, correctness matrix, and constrained/valence matrix
    data_matrix = []
    correctness_matrix = []
    constrained_matrix = []
    
    for seq in sequences_to_show:
        seq_truncated = seq[:max_len]
        correctness, _ = generator.verify_sequence(seq_truncated)
        valence = generator.valence_mask(seq_truncated)
        
        # Pad if necessary
        while len(seq_truncated) < max_len:
            seq_truncated.append(None)  # Padding value (masked)
            correctness.append(np.nan)  # Padding marker (masked)
            valence.append(np.nan)
        
        data_matrix.append(seq_truncated)
        correctness_matrix.append(correctness)
        constrained_matrix.append(valence)
    
    data_matrix = np.array(data_matrix, dtype=object)
    correctness_matrix = np.array(correctness_matrix, dtype=float)
    constrained_matrix = np.array(constrained_matrix, dtype=float)
    
    created_fig = False
    if ax is None:
        # Fewer sequences, larger panels and fonts
        fig_height = max(6, num_sequences * 1.4 + 2)
        fig_width = max(14, max_len * 0.65)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        created_fig = True
    
    # Create color matrix: 1 (correct) = light green, 0 (incorrect) = red, 2 (neutral/no valence) = gray
    from matplotlib.colors import ListedColormap
    colors = ['#ff6b6b', '#90EE90', '#d3d3d3']  # red, light green, gray
    cmap = ListedColormap(colors)
    cmap.set_bad(color=(1, 1, 1, 0))  # transparent for padding
    
    # Mask padding so it doesn't render at all, and show neutral positions in gray
    color_indices = np.where(
        np.isnan(correctness_matrix),
        np.nan,
        np.where(constrained_matrix == 0.0, 2.0, correctness_matrix)  # 2=neutral, else 0/1
    )
    masked_colors = np.ma.masked_invalid(color_indices)
    
    # Plot the heatmap
    im = ax.imshow(masked_colors, cmap=cmap, aspect='auto', vmin=0, vmax=2)
    
    # Add text annotations (the actual numbers) — larger fixed font
    fontsize = 12
    for i in range(len(sequences_to_show)):
        for j in range(max_len):
            val = data_matrix[i, j]
            if val is not None:  # Not padding
                text_color = 'black'
                ax.text(j, i, str(val), ha='center', va='center', fontsize=fontsize, color=text_color, fontweight='bold')
    
    # Set labels — larger fonts
    ax.set_xlabel("Position in Sequence", fontsize=14)
    ax.set_ylabel("Sequence #", fontsize=14)
    ax.set_title(title or "Generated Sequences with Rule Correctness\n(Green = Correct, Red = Incorrect)", fontsize=15)
    ax.tick_params(axis='both', labelsize=12)
    
    # Set ticks
    ax.set_xticks(range(0, max_len, max(1, max_len // 15)))
    ax.set_yticks(range(num_sequences))
    ax.set_yticklabels([f"Seq {i+1}" for i in range(num_sequences)])
    
    # Add colorbar/legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#90EE90', label='Correct'),
        Patch(facecolor='#ff6b6b', label='Incorrect'),
        Patch(facecolor='#d3d3d3', label='Neutral'),
    ]
    if show_legend:
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    if created_fig:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
        else:
            plt.show()
    
    # Return summary statistics
    # Only count constrained (valenced) positions for accuracy stats
    total_correct = np.sum((correctness_matrix == 1) & (constrained_matrix == 1))
    total_incorrect = np.sum((correctness_matrix == 0) & (constrained_matrix == 1))
    total_valid = total_correct + total_incorrect
    accuracy = total_correct / total_valid if total_valid > 0 else 0
    return accuracy, total_correct, total_incorrect

def plot_generated_sequences_heatmap_before_after(generated_sequences_e0, generated_sequences_final, generator, save_path=None, num_sequences=3, max_length=50):
    """Plot before/after generated sequences heatmaps in one image."""
    if not generated_sequences_e0:
        return plot_generated_sequences_heatmap(
            generated_sequences_final, generator,
            save_path=save_path, num_sequences=num_sequences, max_length=max_length
        )
    
    # Fewer sequences, larger panels
    fig_height = max(10, num_sequences * 1.6 + 5)
    fig_width = max(16, max_length * 0.65)
    fig, axes = plt.subplots(2, 1, figsize=(fig_width, fig_height))
    
    acc0, c0, i0 = plot_generated_sequences_heatmap(
        generated_sequences_e0, generator,
        save_path=None, num_sequences=num_sequences, max_length=max_length,
        ax=axes[0], title=f"E0 Generated Sequences (n={num_sequences})", show_legend=False
    )
    accf, cf, inf = plot_generated_sequences_heatmap(
        generated_sequences_final, generator,
        save_path=None, num_sequences=num_sequences, max_length=max_length,
        ax=axes[1], title=f"Final Generated Sequences (n={num_sequences})", show_legend=True
    )
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()
    
    return (acc0, c0, i0), (accf, cf, inf)

def plot_training_data_heatmap(training_sequences, generator, save_path=None, num_sequences=4, max_length=50):
    """
    Plot training data sequences as an annotated heatmap showing correctness.
    Red = incorrect, Green = correct.
    Same style as generated sequences heatmap. Shows fewer sequences, larger cells and fonts.
    """
    sequences_to_show = training_sequences[:num_sequences]
    
    # Truncate to max_length and pad to same length
    max_len = min(max_length, max(len(seq) for seq in sequences_to_show))
    
    # Create data matrix, correctness matrix, and constrained/valence matrix
    data_matrix = []
    correctness_matrix = []
    constrained_matrix = []
    
    for seq in sequences_to_show:
        seq_truncated = seq[:max_len]
        correctness, _ = generator.verify_sequence(seq_truncated)
        valence = generator.valence_mask(seq_truncated)
        
        # Pad if necessary
        while len(seq_truncated) < max_len:
            seq_truncated.append(None)  # Padding value (masked)
            correctness.append(np.nan)  # Padding marker (masked)
            valence.append(np.nan)

        data_matrix.append(seq_truncated)
        correctness_matrix.append(correctness)
        constrained_matrix.append(valence)
    
    data_matrix = np.array(data_matrix, dtype=object)
    correctness_matrix = np.array(correctness_matrix, dtype=float)
    constrained_matrix = np.array(constrained_matrix, dtype=float)
    
    # Fewer sequences, larger panels and fonts
    fig_height = max(8, num_sequences * 1.2 + 2)
    fig_width = max(14, max_len * 0.6)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Create color matrix: 1 (correct) = light green, 0 (incorrect) = red, 2 (neutral/no valence) = gray
    from matplotlib.colors import ListedColormap
    colors = ['#ff6b6b', '#90EE90', '#d3d3d3']  # red, light green, gray
    cmap = ListedColormap(colors)
    cmap.set_bad(color=(1, 1, 1, 0))  # transparent for padding
    
    # Mask padding so it doesn't render at all, and show neutral positions in gray
    color_indices = np.where(
        np.isnan(correctness_matrix),
        np.nan,
        np.where(constrained_matrix == 0.0, 2.0, correctness_matrix)  # 2=neutral, else 0/1
    )
    masked_colors = np.ma.masked_invalid(color_indices)
    
    # Plot the heatmap
    im = ax.imshow(masked_colors, cmap=cmap, aspect='auto', vmin=0, vmax=2)
    
    # Add text annotations — larger fixed font
    fontsize = 12
    for i in range(len(sequences_to_show)):
        for j in range(max_len):
            val = data_matrix[i, j]
            if val is not None:  # Not padding
                text_color = 'black'
                ax.text(j, i, str(val), ha='center', va='center', fontsize=fontsize, color=text_color, fontweight='bold')
    
    # Set labels — larger fonts
    ax.set_xlabel("Position in Sequence", fontsize=14)
    ax.set_ylabel("Sequence #", fontsize=14)
    ax.set_title("Training Data Sequences with Rule Correctness\n(Green = Correct, Red = Incorrect)", fontsize=15)
    ax.tick_params(axis='both', labelsize=12)
    
    # Set ticks — show all sequence labels when we have few sequences
    ax.set_xticks(range(0, max_len, max(1, max_len // 15)))
    ax.set_yticks(range(num_sequences))
    ax.set_yticklabels([f"Seq {i+1}" for i in range(num_sequences)])
    
    # Add colorbar/legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#90EE90', label='Correct'),
        Patch(facecolor='#ff6b6b', label='Incorrect'),
        Patch(facecolor='#d3d3d3', label='Neutral'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()
    
    # Return summary statistics
    # Only count constrained (valenced) positions for accuracy stats
    total_correct = np.sum((correctness_matrix == 1) & (constrained_matrix == 1))
    total_incorrect = np.sum((correctness_matrix == 0) & (constrained_matrix == 1))
    total_valid = total_correct + total_incorrect
    accuracy = total_correct / total_valid if total_valid > 0 else 0
    return accuracy, total_correct, total_incorrect
def plot_architecture_diagram(config: dict, save_path: str = None, model=None, vocab_size=None, batch_size=None):
    """Generate a professional architecture diagram using matplotlib.

    Produces a clean flow-chart with rounded boxes (FancyBboxPatch),
    consistent spacing, a modern colour palette, and proper arrowheads.
    Saves both PNG and SVG side-by-side.
    """
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
    import matplotlib.patheffects as pe

    # ── extract dimensions ──────────────────────────────────────────────
    has_ln1 = has_ln2 = False
    has_proj = False

    if model is not None:
        has_ln1 = hasattr(model, 'ln1') and 'ln1' in model._modules
        has_ln2 = hasattr(model, 'ln2') and 'ln2' in model._modules
        vocab_size_model = model.token_embedding.num_embeddings
        n_embd = model.token_embedding.embedding_dim
        block_size = model.block_size
        num_heads = len(model.sa_heads.heads)
        head_size = model.sa_heads.heads[0].head_size
        has_proj = hasattr(model, 'proj') and 'proj' in model._modules
        ffwd_net = model.ffwd.net
        ffwd_hidden_dim = ffwd_net[0].out_features
        if vocab_size is None:
            vocab_size = vocab_size_model
    else:
        mc = config['model']
        dc = config['data']
        if vocab_size is None:
            vocab_size = dc['max_value'] - dc['min_value'] + 1
            if dc.get('generator_type') in ['PlusLastEvenRule']:
                vocab_size += 1
        n_embd = mc['n_embd']; block_size = mc['block_size']
        num_heads = mc['num_heads']; head_size = mc['head_size']
        ffwd_hidden_dim = n_embd * 16

    if batch_size is None:
        batch_size = config.get('training', {}).get('batch_size', 4)

    # ── colours ─────────────────────────────────────────────────────────
    C_INPUT   = '#E8F4FD'; C_EMBED = '#DAEAF6'
    C_ATTN    = '#FFF8E1'; C_ATTN_BG = '#FFFDF5'; C_ATTN_BD = '#F9A825'
    C_LINEAR  = '#FADBD8'; C_OUTPUT = '#D5F5E3'
    C_STROKE  = '#34495E'; C_RESID = '#27AE60'
    C_SUB     = '#666666'

    # Check if residuals are used
    use_residual = True
    if model is not None:
        use_residual = getattr(model, 'use_residual', True)

    # ── figure setup (defer xlim until we know total width) ───────────
    H_px = 500
    fig, ax = plt.subplots(figsize=(20, 6), dpi=160)
    ax.set_ylim(H_px, 0)        # y increases downward (like SVG)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    # ── drawing helpers ─────────────────────────────────────────────────
    def draw_box(x, y, w, h, color, label, sub=None, fs=10.5, sub_fs=8.5):
        """Rounded-rectangle box centred at (x+w/2, y+h/2)."""
        fancy = FancyBboxPatch((x, y), w, h,
                               boxstyle="round,pad=0,rounding_size=7",
                               facecolor=color, edgecolor=C_STROKE,
                               linewidth=1.4, zorder=3)
        ax.add_patch(fancy)
        cx, cy_box = x + w / 2, y + h / 2
        lines = label.split('\n')
        n = len(lines)
        lh = fs * 1.4
        if sub:
            sub_lines = sub.split('\n')
            n_sub = len(sub_lines)
            label_block = lh * (n - 1)
            sub_block = sub_fs * 1.3 * (n_sub - 1)
            gap_between = 16
            total = label_block + gap_between + sub_block
            label_top = cy_box - total / 2
            for i, ln in enumerate(lines):
                ax.text(cx, label_top + i * lh, ln, ha='center', va='center',
                        fontsize=fs, fontweight='bold', color=C_STROKE, zorder=4,
                        fontfamily='sans-serif')
            sub_top = label_top + label_block + gap_between
            for j, sl in enumerate(sub_lines):
                ax.text(cx, sub_top + j * (sub_fs * 1.3), sl,
                        ha='center', va='center',
                        fontsize=sub_fs, color=C_SUB, zorder=4,
                        fontfamily='sans-serif')
        else:
            base_y = cy_box - lh * (n - 1) / 2
            for i, ln in enumerate(lines):
                ax.text(cx, base_y + i * lh, ln, ha='center', va='center',
                        fontsize=fs, fontweight='bold', color=C_STROKE, zorder=4,
                        fontfamily='sans-serif')

    def draw_arrow(x1, y1, x2, y2, color=C_STROKE, lw=1.3):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                    shrinkA=1, shrinkB=1), zorder=5)

    def draw_circle(cx, cy, r, label='+'):
        circ = Circle((cx, cy), r, facecolor='white', edgecolor=C_STROKE,
                       linewidth=1.4, zorder=3)
        ax.add_patch(circ)
        ax.text(cx, cy, label, ha='center', va='center',
                fontsize=15, fontweight='bold', color=C_STROKE, zorder=4,
                fontfamily='sans-serif')

    def draw_residual(x1, y_start, x2, y_end, label='residual', below=True):
        """Right-angle residual: go down, across, then up to target."""
        drop = 55 if below else -55
        y_mid = y_start + drop
        # down from source
        ax.plot([x1, x1], [y_start, y_mid], color=C_RESID, lw=1.3,
                linestyle='--', zorder=2, clip_on=False)
        # across
        ax.plot([x1, x2], [y_mid, y_mid], color=C_RESID, lw=1.3,
                linestyle='--', zorder=2, clip_on=False)
        # up to target (with arrow)
        draw_arrow(x2, y_mid, x2, y_end, color=C_RESID, lw=1.3)
        # label
        ax.text((x1 + x2) / 2, y_mid + (12 if below else -12), label,
                ha='center', va='center', fontsize=8, color=C_RESID,
                fontfamily='sans-serif', fontstyle='italic', zorder=4)

    # ── layout constants ────────────────────────────────────────────────
    cy = 210           # centre-line y
    bh = 80            # standard box height
    gap = 24           # horizontal gap between elements
    r_plus = 18        # radius of + circles

    # ── build diagram left to right ─────────────────────────────────────
    x = 30

    # Input Tokens
    inp_w = 88
    draw_box(x, cy - bh/2, inp_w, bh, C_INPUT, 'Input\nTokens', f'({batch_size}, {block_size})')
    x_inp_r = x + inp_w
    x += inp_w + gap

    # Token + Position Embeddings (stacked)
    emb_w, emb_h = 115, 64
    emb_gap = 14
    emb_top_y = cy - emb_h - emb_gap / 2
    emb_bot_y = cy + emb_gap / 2
    draw_box(x, emb_top_y, emb_w, emb_h, C_EMBED, 'Token Emb', f'({vocab_size}, {n_embd})')
    draw_box(x, emb_bot_y, emb_w, emb_h, C_EMBED, 'Position Emb', f'({block_size}, {n_embd})')
    draw_arrow(x_inp_r, cy - 12, x, emb_top_y + emb_h / 2)
    draw_arrow(x_inp_r, cy + 12, x, emb_bot_y + emb_h / 2)
    x_emb_r = x + emb_w
    x += emb_w + gap

    # Add embeddings (+)
    plus0_cx = x + r_plus
    draw_circle(plus0_cx, cy, r_plus)
    draw_arrow(x_emb_r, emb_top_y + emb_h / 2, plus0_cx - r_plus, cy - 6)
    draw_arrow(x_emb_r, emb_bot_y + emb_h / 2, plus0_cx - r_plus, cy + 6)
    # Shape label above
    ax.text(plus0_cx, emb_top_y - 14, f'x : (B,T,C) = ({batch_size},{block_size},{n_embd})',
            ha='center', va='center', fontsize=8, color='#777', fontfamily='sans-serif')
    x_add0_r = plus0_cx + r_plus
    x = x_add0_r + gap

    # ── Self-Attention block ────────────────────────────────────────────
    attn_x0 = x - 8

    # W_Q, W_K, W_V (stacked)
    qkv_w, qkv_h, qkv_gap = 66, 44, 8
    total_qkv = 3 * qkv_h + 2 * qkv_gap
    qkv_top = cy - total_qkv / 2
    qkv_cy_list = []
    for i, lbl in enumerate(['W_Q', 'W_K', 'W_V']):
        by = qkv_top + i * (qkv_h + qkv_gap)
        draw_box(x, by, qkv_w, qkv_h, C_ATTN, lbl, f'{n_embd}\u2192{head_size}', fs=10, sub_fs=8)
        qkv_cy_list.append(by + qkv_h / 2)
    # Fan-out arrows from + to each W_Q/W_K/W_V — use a short horizontal
    # trunk then individual branches so arrows don't cross through boxes
    fan_x = x_add0_r + (x - x_add0_r) * 0.5  # midpoint of gap
    # Trunk: horizontal line from + to the fan point
    ax.plot([x_add0_r, fan_x], [cy, cy],
            color=C_STROKE, lw=1.3, zorder=5, solid_capstyle='round')
    # Branches: from fan point to each QKV box
    for yc in qkv_cy_list:
        ax.plot([fan_x, fan_x], [cy, yc],
                color=C_STROKE, lw=1.3, zorder=5, solid_capstyle='round')
        draw_arrow(fan_x, yc, x, yc, lw=1.3)
    x_qkv_r = x + qkv_w
    x += qkv_w + gap

    # QK^T / sqrt(dk)
    dot_w = 80
    x_dot_l = x
    draw_box(x, cy - bh/2, dot_w, bh, C_ATTN, 'QK\u1d40 / \u221Ad\u2096', fs=10)
    # Q → top of QK^T, K → bottom of QK^T (short horizontal arrows)
    draw_arrow(x_qkv_r, qkv_cy_list[0], x, cy - 12)
    draw_arrow(x_qkv_r, qkv_cy_list[1], x, cy + 12)
    x_dot_r = x + dot_w
    x += dot_w + gap

    # Causal Mask + Softmax
    mask_w = 90
    draw_box(x, cy - bh/2, mask_w, bh, C_ATTN, 'Causal Mask\n+ Softmax', fs=9.5)
    draw_arrow(x_dot_r, cy, x, cy)
    x_mask_r = x + mask_w
    x += mask_w + gap

    # Attn × V
    av_w = 78
    x_av_l = x
    draw_box(x, cy - bh/2, av_w, bh, C_ATTN, 'Attn \u00d7 V', fs=10.5)
    # Attention weights → top of Attn×V
    draw_arrow(x_mask_r, cy, x_av_l, cy - 8)
    # V path: route BELOW the QK^T and Mask boxes to avoid crossing
    # W_V right edge → down to below boxes → right → up into Attn×V bottom
    v_route_y = cy + bh / 2 + 14  # below the main-flow boxes
    v_start_x = x_qkv_r
    v_start_y = qkv_cy_list[2]  # W_V center-y
    v_end_x = x_av_l + av_w / 2  # center of Attn×V
    v_end_y = cy + bh / 2       # bottom edge of Attn×V
    # Draw the routed V path: right from W_V → down → across → up into Attn×V
    ax.plot([v_start_x, v_start_x + 10], [v_start_y, v_start_y],
            color=C_STROKE, lw=1.3, zorder=5, solid_capstyle='round')
    ax.plot([v_start_x + 10, v_start_x + 10], [v_start_y, v_route_y],
            color=C_STROKE, lw=1.3, zorder=5, solid_capstyle='round')
    ax.plot([v_start_x + 10, v_end_x], [v_route_y, v_route_y],
            color=C_STROKE, lw=1.3, zorder=5, solid_capstyle='round')
    draw_arrow(v_end_x, v_route_y, v_end_x, v_end_y, lw=1.3)
    # Label the V path
    ax.text((v_start_x + 10 + v_end_x) / 2, v_route_y + 10, 'V',
            ha='center', va='center', fontsize=9, fontweight='bold',
            color=C_STROKE, fontfamily='sans-serif', zorder=6)

    x_av_r = x + av_w
    x += av_w + gap + 6

    attn_x1 = x

    # Attention block outline (dashed) — must enclose V routing path
    attn_pad = 14
    attn_bottom = v_route_y + 22
    attn_rect_top = qkv_top - 30
    attn_rect = FancyBboxPatch(
        (attn_x0 - attn_pad, attn_rect_top), attn_x1 - attn_x0 + 2 * attn_pad, attn_bottom - attn_rect_top,
        boxstyle="round,pad=0,rounding_size=10",
        facecolor=C_ATTN_BG, edgecolor=C_ATTN_BD,
        linewidth=1.4, linestyle='--', zorder=1)
    ax.add_patch(attn_rect)
    ax.text((attn_x0 + attn_x1) / 2, qkv_top - 16,
            f'Self-Attention ({num_heads} head{"s" if num_heads > 1 else ""})',
            ha='center', va='center', fontsize=10, fontweight='bold',
            color=C_ATTN_BD, fontfamily='sans-serif', zorder=2)

    # ── Post-attention ──────────────────────────────────────────────────
    if use_residual:
        # Residual add #1: x + attn_out
        plus1_cx = x + r_plus
        draw_circle(plus1_cx, cy, r_plus)
        draw_arrow(x_av_r, cy, plus1_cx - r_plus, cy)
        # Residual path: from embedding add (+0) down-across-up to +1
        draw_residual(plus0_cx, cy + r_plus, plus1_cx, cy + r_plus,
                      label='x (skip around attention)')
        x_p1_r = plus1_cx + r_plus
        x = x_p1_r + gap
    else:
        x_p1_r = x_av_r
        x = x_av_r + gap

    # Feed-Forward
    ff_w = 125
    draw_box(x, cy - bh/2, ff_w, bh, C_LINEAR, 'Feed-Forward',
             f'Linear({n_embd},{ffwd_hidden_dim})\nReLU \u2192 Linear({ffwd_hidden_dim},{n_embd})',
             sub_fs=8)
    draw_arrow(x_p1_r, cy, x, cy)
    x_ff_r = x + ff_w
    x += ff_w + gap

    if use_residual:
        # Residual add #2: x + ffwd(x)
        plus2_cx = x + r_plus
        draw_circle(plus2_cx, cy, r_plus)
        draw_arrow(x_ff_r, cy, plus2_cx - r_plus, cy)
        # Residual path: from +1 down-across-up to +2
        draw_residual(plus1_cx, cy + r_plus, plus2_cx, cy + r_plus,
                      label='x (skip around FFN)')
        x_p2_r = plus2_cx + r_plus
        x = x_p2_r + gap
    else:
        x_p2_r = x_ff_r
        x = x_ff_r + gap

    # LM Head
    lm_w = 88
    draw_box(x, cy - bh/2, lm_w, bh, C_OUTPUT, 'LM Head',
             f'Linear({n_embd}, {vocab_size})')
    draw_arrow(x_p2_r, cy, x, cy)
    x_lm_r = x + lm_w
    x += lm_w + gap

    # Softmax
    sm_w = 76
    draw_box(x, cy - bh/2, sm_w, bh, C_OUTPUT, 'Softmax')
    draw_arrow(x_lm_r, cy, x, cy)
    x_sm_r = x + sm_w
    x += sm_w + gap

    # Output Probabilities
    out_w = 92
    draw_box(x, cy - bh/2, out_w, bh, C_OUTPUT, 'P(next)',
             f'({batch_size},{block_size},{vocab_size})')
    draw_arrow(x_sm_r, cy, x, cy)
    x += out_w + 30

    # ── Notation (top-right) ────────────────────────────────────────────
    nx = x - 115
    ny = 22
    ax.text(nx, ny, 'Notation', ha='left', va='top', fontsize=9.5,
            fontweight='bold', color=C_STROKE, fontfamily='sans-serif')
    for i, ln in enumerate([
        f'B = {batch_size}  (batch)',
        f'T = {block_size}  (sequence)',
        f'C = {n_embd}  (n_embd)',
        f'd_k = {head_size}  (head size)',
        f'vocab = {vocab_size}',
    ]):
        ax.text(nx, ny + 18 + i * 15, ln, ha='left', va='top',
                fontsize=8, color='#555', fontfamily='sans-serif')

    # ── Legend (bottom-left) ────────────────────────────────────────────
    lx, ly = 30, H_px - 26
    for i, (c, lbl) in enumerate([
        (C_EMBED, 'Embedding'), (C_ATTN, 'Attention'),
        (C_LINEAR, 'Feed-Forward'), (C_OUTPUT, 'Output'),
    ]):
        ox = lx + i * 120
        fancy = FancyBboxPatch((ox, ly - 7), 14, 14,
                               boxstyle="round,pad=0,rounding_size=3",
                               facecolor=c, edgecolor=C_STROKE, linewidth=0.8, zorder=3)
        ax.add_patch(fancy)
        ax.text(ox + 20, ly, lbl, ha='left', va='center',
                fontsize=8.5, color=C_STROKE, fontfamily='sans-serif', zorder=4)

    # ── finalise axes limits ────────────────────────────────────────────
    W_px = x + 20
    ax.set_xlim(0, W_px)
    ax.set_aspect('equal')

    # ── save ────────────────────────────────────────────────────────────
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=200, facecolor='white', pad_inches=0.15)
        # Also save SVG
        svg_path = save_path.rsplit('.', 1)[0] + '.svg'
        plt.savefig(svg_path, bbox_inches='tight', facecolor='white', pad_inches=0.15, format='svg')
        plt.close()
        print(f"Architecture diagram saved to {save_path} and {svg_path}")
    else:
        plt.show()

# -----------------------------
# Q/K Embedding Space Visualization
# -----------------------------
def plot_qk_embedding_space(model, itos, save_path: str = None, step_label: int | None = None):
    """
    Create a single scatter plot showing ALL Q and K transformed embeddings
    with both token AND position labels for every combination.
    Uses consistent format: token with position subscript (e.g. 8₃)
    
    Args:
        model: Trained TransformerLM
        itos: Index-to-string mapping for tokens
        save_path: Path to save the figure
    """
    model.eval()
    
    # Get model parameters
    vocab_size = model.token_embedding.weight.shape[0]
    block_size = model.position_embedding_table.weight.shape[0]
    
    # Get the first attention head's Q, K weights
    head = model.sa_heads.heads[0]
    W_Q = head.query.weight.detach().cpu().numpy()  # (head_size, n_embd)
    W_K = head.key.weight.detach().cpu().numpy()    # (head_size, n_embd)
    head_size = W_Q.shape[0]
    
    # Get embeddings
    token_emb = model.token_embedding.weight.detach().cpu().numpy()  # (vocab_size, n_embd)
    pos_emb = model.position_embedding_table.weight.detach().cpu().numpy()  # (block_size, n_embd)
    
    # Compute Q and K for all token-position combinations
    num_combinations = vocab_size * block_size
    
    Q_all = []
    K_all = []
    labels = []
    
    for t in range(vocab_size):
        for p in range(block_size):
            combined_emb = token_emb[t] + pos_emb[p]
            q = W_Q @ combined_emb  # (head_size,)
            k = W_K @ combined_emb  # (head_size,)
            Q_all.append(q)
            K_all.append(k)
            token_str = str(itos[t])
            labels.append(_token_pos_label(token_str, p))
    
    Q_all = np.array(Q_all)  # (num_combinations, head_size)
    K_all = np.array(K_all)  # (num_combinations, head_size)
    
    # If head_size is 2, plot directly. Otherwise, use PCA
    if head_size == 2:
        Q_2d = Q_all
        K_2d = K_all
    else:
        from sklearn.decomposition import PCA
        combined = np.vstack([Q_all, K_all])
        pca = PCA(n_components=2)
        combined_2d = pca.fit_transform(combined)
        Q_2d = combined_2d[:num_combinations]
        K_2d = combined_2d[num_combinations:]
    
    # Create single large figure
    fig, ax = plt.subplots(figsize=(20, 16))
    if step_label is not None:
        fig.suptitle(f"Step: {step_label}", fontsize=18, fontweight="bold", y=0.98)
    label_fontsize = 20
    title_fontsize = 24
    axis_fontsize = 24
    tick_fontsize = 22

    # Plot invisible scatter points (for axis scaling)
    all_x = np.concatenate([Q_2d[:, 0], K_2d[:, 0]])
    all_y = np.concatenate([Q_2d[:, 1], K_2d[:, 1]])
    ax.scatter(all_x, all_y, s=0, alpha=0)

    # Add text labels for ALL Q points (blue)
    for i in range(num_combinations):
        ax.text(Q_2d[i, 0], Q_2d[i, 1], labels[i], fontsize=label_fontsize, ha='center', va='center', color='blue')

    # Add text labels for ALL K points (red)
    for i in range(num_combinations):
        ax.text(K_2d[i, 0], K_2d[i, 1], labels[i], fontsize=label_fontsize, ha='center', va='center', color='red')

    ax.set_xlabel("Dimension 1" + (" (PCA)" if head_size != 2 else ""), fontsize=axis_fontsize)
    ax.set_ylabel("Dimension 2" + (" (PCA)" if head_size != 2 else ""), fontsize=axis_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    ax.set_title(f"Q and K Embedding Space\n{num_combinations} Q (blue) + {num_combinations} K (red) = {2*num_combinations} total\n({vocab_size} tokens × {block_size} positions)", fontsize=title_fontsize, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Legend with actual colored patches
    legend_handles = [
        Patch(facecolor='blue', edgecolor='black', label='Query'),
        Patch(facecolor='red', edgecolor='black', label='Key'),
    ]
    leg = ax.legend(handles=legend_handles, loc='upper left', fontsize=axis_fontsize)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()
        print(f"Q/K embedding space plot saved to {save_path}")
    else:
        plt.show()


def plot_qk_embedding_space_focused_query(model, itos, token_str="+", position=5, save_path=None, grid_resolution=150):
    """
    One query only (e.g. +_5): show that query, all keys (keys with position >= focus position grayed),
    and background heatmap of dot product between (x,y) and the focus query vector.
    """
    model.eval()
    vocab_size = model.token_embedding.weight.shape[0]
    block_size = model.position_embedding_table.weight.shape[0]
    head = model.sa_heads.heads[0]
    W_Q = head.query.weight.detach().cpu().numpy()
    W_K = head.key.weight.detach().cpu().numpy()
    head_size = W_Q.shape[0]
    token_emb = model.token_embedding.weight.detach().cpu().numpy()
    pos_emb = model.position_embedding_table.weight.detach().cpu().numpy()

    num_combinations = vocab_size * block_size
    Q_all = []
    K_all = []
    labels = []
    for t in range(vocab_size):
        for p in range(block_size):
            combined_emb = token_emb[t] + pos_emb[p]
            q = W_Q @ combined_emb
            k = W_K @ combined_emb
            Q_all.append(q)
            K_all.append(k)
            labels.append(_token_pos_label(str(itos[t]), p))
    Q_all = np.array(Q_all)
    K_all = np.array(K_all)

    if head_size != 2:
        from sklearn.decomposition import PCA
        combined = np.vstack([Q_all, K_all])
        pca = PCA(n_components=2)
        combined_2d = pca.fit_transform(combined)
        Q_2d = combined_2d[:num_combinations]
        K_2d = combined_2d[num_combinations:]
    else:
        Q_2d = Q_all
        K_2d = K_all

    # Index of focus query: token_str at position
    t_focus = None
    for t in range(vocab_size):
        if str(itos[t]) == token_str:
            t_focus = t
            break
    if t_focus is None:
        print(f"plot_qk_embedding_space_focused_query: token '{token_str}' not found in vocab. Skipping.")
        return
    idx_focus = t_focus * block_size + position
    q_focus = Q_2d[idx_focus]  # (2,)

    # Extent with margin
    all_x = np.concatenate([Q_2d[:, 0], K_2d[:, 0]])
    all_y = np.concatenate([Q_2d[:, 1], K_2d[:, 1]])
    margin_x = max(0.5, (all_x.max() - all_x.min()) * 0.1)
    margin_y = max(0.5, (all_y.max() - all_y.min()) * 0.1)
    x_min, x_max = all_x.min() - margin_x, all_x.max() + margin_x
    y_min, y_max = all_y.min() - margin_y, all_y.max() + margin_y

    # Grid for background dot-product heatmap
    xx = np.linspace(x_min, x_max, grid_resolution)
    yy = np.linspace(y_min, y_max, grid_resolution)
    Xgrid, Ygrid = np.meshgrid(xx, yy)
    # At each (x,y) the "key" is (x,y); dot product with q_focus
    dot_grid = Xgrid * q_focus[0] + Ygrid * q_focus[1]

    fig, ax = plt.subplots(figsize=(14, 12))
    label_fontsize = 12
    title_fontsize = 18
    axis_fontsize = 20
    tick_fontsize = 18
    legend_fontsize = 14

    # Background heatmap (dot product with focus query)
    im = ax.pcolormesh(xx, yy, dot_grid, cmap='Greens', shading='auto')
    
    # All other queries in very light blue (background context)
    for i in range(num_combinations):
        if i == idx_focus:
            continue
        ax.text(Q_2d[i, 0], Q_2d[i, 1], labels[i], fontsize=label_fontsize - 2, ha='center', va='center', color='#A0C4E8', alpha=0.8, zorder=2)

    # Key points: red if position < position_focus, gray otherwise (masked by causal mask)
    for i in range(num_combinations):
        p = i % block_size
        color = 'red' if p < position else '#666666'
        ax.text(K_2d[i, 0], K_2d[i, 1], labels[i], fontsize=label_fontsize, ha='center', va='center', color=color, zorder=3)

    # Focused query point (bold blue, on top)
    ax.text(Q_2d[idx_focus, 0], Q_2d[idx_focus, 1], labels[idx_focus], fontsize=label_fontsize + 4, ha='center', va='center', color='blue', fontweight='bold', zorder=4)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Dimension 1" + (" (PCA)" if head_size != 2 else ""), fontsize=axis_fontsize)
    ax.set_ylabel("Dimension 2" + (" (PCA)" if head_size != 2 else ""), fontsize=axis_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    ax.set_title(f"Q/K space: focus on query {_token_pos_label(token_str, position)}\nBackground = dot product with this query; keys with position \u2265 {position} grayed", fontsize=title_fontsize, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    legend_handles = [
        Patch(facecolor='blue', edgecolor='black', label=f'Query {_token_pos_label(token_str, position)}'),
        Patch(facecolor='#A0C4E8', edgecolor='black', label='Other queries'),
        Patch(facecolor='red', edgecolor='black', label=f'Key (position < {position})'),
        Patch(facecolor='#666666', edgecolor='black', label=f'Key (position \u2265 {position})'),
    ]
    ax.legend(handles=legend_handles, loc='upper left', fontsize=legend_fontsize, framealpha=0.9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()
        print(f"Q/K embedding space (focused query) plot saved to {save_path}")
    else:
        plt.show()


def plot_lm_head_probability_heatmaps(model, itos, save_path=None, grid_resolution=80, extent_margin=0.5):
    """
    For n_embd==2 only: plot one heatmap per token (digit) showing P(token | (x,y))
    over the 2D input space to the LM head. Each point (x,y) in the plane is passed
    through the LM head and softmax to get the probability of each output token.

    Args:
        model: Trained model (BigramLanguageModel)
        itos: Index-to-string mapping for tokens
        save_path: Path to save the figure
        grid_resolution: Number of points per axis (default 80)
        extent_margin: Extra margin around embedding extent (default 0.5)
    """
    model.eval()
    vocab_size = model.token_embedding.weight.shape[0]
    n_embd = model.lm_head.in_features
    if n_embd != 2:
        print(f"plot_lm_head_probability_heatmaps: n_embd={n_embd}, need 2. Skipping.")
        return

    with torch.no_grad():
        W = model.lm_head.weight.detach().cpu().numpy()   # (vocab_size, 2)
        b = model.lm_head.bias.detach().cpu().numpy()     # (vocab_size,)
        token_emb = model.token_embedding.weight.detach().cpu().numpy()
        pos_emb = model.position_embedding_table.weight.detach().cpu().numpy()
        combined = token_emb[:, None, :] + pos_emb[None, :, :]  # (vocab, block, 2)
        flat = combined.reshape(-1, 2)
        x_min, x_max = flat[:, 0].min() - extent_margin, flat[:, 0].max() + extent_margin
        y_min, y_max = flat[:, 1].min() - extent_margin, flat[:, 1].max() + extent_margin

    xs = np.linspace(x_min, x_max, grid_resolution)
    ys = np.linspace(y_min, y_max, grid_resolution)
    xx, yy = np.meshgrid(xs, ys)
    points = np.stack([xx.ravel(), yy.ravel()], axis=1)  # (N, 2)
    logits = points @ W.T + b                             # (N, vocab_size)
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)             # (N, vocab_size)

    n_cols = min(4, vocab_size)
    n_rows = (vocab_size + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows), sharex=True, sharey=True)
    axes = np.atleast_2d(axes).flatten()

    vmin, vmax = 0.0, 1.0
    for d in range(vocab_size):
        ax = axes[d]
        Z = probs[:, d].reshape(grid_resolution, grid_resolution)
        im = ax.pcolormesh(xx, yy, Z, cmap='viridis', vmin=vmin, vmax=vmax, shading='auto')
        ax.set_title(f"P({itos[d]})", fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        plt.colorbar(im, ax=ax, label='probability')
    for j in range(vocab_size, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("LM head: P(digit | point in 2D input space)", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()
        print(f"LM head probability heatmaps saved to {save_path}")
    else:
        plt.show()


def _get_v_before_after_for_sequence(model, idx):
    """
    Run first attention head on sequence idx (1, T), return V_before and V_after as numpy (T, head_size).
    """
    model.eval()
    with torch.no_grad():
        B, T = idx.shape
        token_emb = model.token_embedding(idx)
        positions = torch.arange(T, device=idx.device) % model.block_size
        pos_emb = model.position_embedding_table(positions)
        x = (token_emb + pos_emb)  # (1, T, n_embd)
        head = model.sa_heads.heads[0]
        q = head.query(x)
        k = head.key(x)
        v = head.value(x)  # V_before
        weight = q @ k.transpose(-2, -1) / (head.head_size ** 0.5)
        weight = weight.masked_fill(head.tril[:T, :T].to(x.device) == 0, float("-inf"))
        weight = F.softmax(weight, dim=-1)
        out = weight @ v  # V_after
        v_np = v[0].cpu().numpy()   # (T, head_size)
        out_np = out[0].cpu().numpy()  # (T, head_size)
        x_np = x[0].cpu().numpy()     # (T, n_embd)
    return x_np, v_np, out_np


def plot_v_before_after_demo_sequences(model, itos, sequences, save_dir=None, arrow_scale=0.15, grid_resolution=60):
    """
    For each demo sequence, create one figure. Each figure has one panel per output token.
    On each panel: 2D embedding-space grid; at each position where the NEXT token equals
    that panel's token, draw an arrow from (position + scale*V_before) to (position + scale*V_after).

    Args:
        model: Trained model (n_embd=2, head_size=2).
        itos: Index-to-string for tokens.
        sequences: List of sequences (each sequence = list of token ids, length <= block_size).
        save_dir: Directory to save figures (one file per sequence, e.g. v_before_after_demo_0.png).
        arrow_scale: Scale factor for V vectors so arrows fit on grid (default 0.15).
        grid_resolution: Grid resolution for background heatmap (default 60).
    """
    model.eval()
    vocab_size = model.token_embedding.weight.shape[0]
    block_size = model.block_size
    n_embd = model.lm_head.in_features
    if n_embd != 2:
        print(f"plot_v_before_after_demo_sequences: n_embd={n_embd}, need 2. Skipping.")
        return

    # Embedding extent and P(token) grid for background (same as lm_head heatmaps)
    with torch.no_grad():
        token_emb = model.token_embedding.weight.detach().cpu().numpy()
        pos_emb = model.position_embedding_table.weight.detach().cpu().numpy()
        combined = token_emb[:, None, :] + pos_emb[None, :, :]
        flat = combined.reshape(-1, 2)
        x_min, x_max = flat[:, 0].min() - 0.5, flat[:, 0].max() + 0.5
        y_min, y_max = flat[:, 1].min() - 0.5, flat[:, 1].max() + 0.5
        W = model.lm_head.weight.detach().cpu().numpy()
        b = model.lm_head.bias.detach().cpu().numpy()

    xs = np.linspace(x_min, x_max, grid_resolution)
    ys = np.linspace(y_min, y_max, grid_resolution)
    xx, yy = np.meshgrid(xs, ys)
    points = np.stack([xx.ravel(), yy.ravel()], axis=1)
    logits = points @ W.T + b
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)  # (N, vocab_size)
    argmax_token = logits.argmax(axis=1)  # (N,) which token has highest logit at each point

    n_data_rows = 3   # V values, Embed with arrows to Final, Final position
    n_cols = vocab_size  # one column per predicted digit
    n_rows = n_data_rows

    arrow_data_list = []  # for big arrow-only figure: (x_np, v_before, v_after, correct, seq, plot_x_min, plot_x_max, plot_y_min, plot_y_max, T)

    for seq_idx, seq in enumerate(sequences):
        if len(seq) < 2:
            continue
        seq = seq[:block_size]
        T = len(seq)
        dev = next(model.parameters()).device
        idx = torch.tensor([seq], dtype=torch.long, device=dev)
        with torch.no_grad():
            logits, _ = model(idx)
            pred_next = logits[0].argmax(dim=-1).cpu().numpy()
        correct = np.array([i < T - 1 and pred_next[i] == seq[i + 1] for i in range(T)])
        with torch.no_grad():
            g_seed = 42 + seq_idx
            torch.manual_seed(g_seed)
            start = torch.tensor([[seq[0]]], dtype=torch.long, device=dev)
            generated = model.generate(start, max_new_tokens=T - 1)[0].tolist()
        x_np, v_before, v_after = _get_v_before_after_for_sequence(model, idx)
        land_x = x_np[:, 0] + v_after[:, 0]
        land_y = x_np[:, 1] + v_after[:, 1]
        margin = 0.6
        # Include V values (v_before) in plot limits
        plot_x_min = min(x_min, x_np[:, 0].min(), v_before[:, 0].min(), v_after[:, 0].min(), land_x.min()) - margin
        plot_x_max = max(x_max, x_np[:, 0].max(), v_before[:, 0].max(), v_after[:, 0].max(), land_x.max()) + margin
        plot_y_min = min(y_min, x_np[:, 1].min(), v_before[:, 1].min(), v_after[:, 1].min(), land_y.min()) - margin
        plot_y_max = max(y_max, x_np[:, 1].max(), v_before[:, 1].max(), v_after[:, 1].max(), land_y.max()) + margin
        arrow_data_list.append((x_np.copy(), v_before.copy(), v_after.copy(), correct.copy(), seq[:], plot_x_min, plot_x_max, plot_y_min, plot_y_max, T))
        xs_plot = np.linspace(plot_x_min, plot_x_max, grid_resolution)
        ys_plot = np.linspace(plot_y_min, plot_y_max, grid_resolution)
        xx_plot, yy_plot = np.meshgrid(xs_plot, ys_plot)
        points_plot = np.stack([xx_plot.ravel(), yy_plot.ravel()], axis=1)
        dev = next(model.parameters()).device
        with torch.no_grad():
            pts = torch.tensor(points_plot, dtype=torch.float32, device=dev)
            h = pts + model.ffwd(pts)
            logits_plot = model.lm_head(h).cpu().numpy()
        probs_plot = np.exp(logits_plot - logits_plot.max(axis=1, keepdims=True))
        probs_plot /= probs_plot.sum(axis=1, keepdims=True)

        # Increase figure size and resolution for better quality
        figsize_mult = 1.5 if seq_idx == 0 else 1.0  # Larger for first demo (plot 17)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols * figsize_mult, 2.5 * n_rows * figsize_mult), sharex=True, sharey=True)
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        # Increase font sizes for seq_idx==0 (plot 15) and seq_idx==2 (plot 17)
        title_fs = 14 if (seq_idx == 0 or seq_idx == 2) else 10
        label_fs = 11 if (seq_idx == 0 or seq_idx == 2) else 9
        fs = 7.0 if (seq_idx == 0 or seq_idx == 2) else 4.5  # Font size for point labels
        
        for r in range(n_rows):
            for c in range(n_cols):
                ax = axes[r, c]
                ax.set_xlim(plot_x_min, plot_x_max)
                ax.set_ylim(plot_y_min, plot_y_max)
                ax.set_aspect('equal')
                Z = probs_plot[:, c].reshape(grid_resolution, grid_resolution)
                ax.pcolormesh(xx_plot, yy_plot, Z, cmap='viridis', vmin=0, vmax=1, shading='auto')
                # Add faint dashed lines to indicate origin (0,0)
                ax.axhline(y=0, color='white', linestyle='--', linewidth=0.5, alpha=0.3, zorder=3)
                ax.axvline(x=0, color='white', linestyle='--', linewidth=0.5, alpha=0.3, zorder=3)
                if c == 0:
                    ax.set_ylabel("dim 1", fontsize=label_fs)
                if r == 0:
                    ax.set_title(f"P(next = {itos[c]})", fontsize=title_fs)
                if r == n_rows - 1:
                    ax.set_xlabel("dim 0", fontsize=label_fs)
                # Increase tick label font sizes for plots 15 and 17
                ax.tick_params(axis='both', which='major', labelsize=label_fs)
        
        # Row 0: V values for the specific sequence
        for i in range(T):
            lbl = _token_pos_label(itos[seq[i]], i)
            v_pos = v_before[i]  # V value for this token-position pair
            for c in range(n_cols):
                ax = axes[0, c]
                ax.text(v_pos[0], v_pos[1], lbl, fontsize=fs, fontweight='bold', ha='center', va='center', color='white', zorder=5,
                        path_effects=[pe.withStroke(linewidth=0.8, foreground='black')])
        
        # Row 1: Embed with arrows to Final (x_np with arrows to x_np + v_after)
        for i in range(T):
            lbl = _token_pos_label(itos[seq[i]], i)
            px0, py0 = x_np[i, 0], x_np[i, 1]  # Embed position
            px_final, py_final = x_np[i, 0] + v_after[i, 0], x_np[i, 1] + v_after[i, 1]  # Final position
            for c in range(n_cols):
                ax = axes[1, c]
                # Draw arrow from embed to final (skip arrows for seq_idx==1, but keep labels)
                if seq_idx != 1:  # Only draw arrows if NOT seq_idx==1 (plot 16)
                    dx = px_final - px0
                    dy = py_final - py0
                    arrow_length = np.sqrt(dx**2 + dy**2)
                    head_size = max(0.1, min(0.2, arrow_length * 0.1))
                    ax.arrow(px0, py0, dx, dy,
                            head_width=head_size, head_length=head_size, fc='white', ec='white', alpha=0.8, length_includes_head=True, width=0.01, zorder=4)
                # Annotate at beginning (embed position) - keep this for all sequences
                ax.text(px0, py0, lbl, fontsize=fs, fontweight='bold', ha='center', va='center', color='white', zorder=5,
                        path_effects=[pe.withStroke(linewidth=0.8, foreground='black')])
        
        # Row 2: Final position; white text, stroke color = correct (green) / wrong (red) - keep this for all sequences
        for i in range(T):
            lbl = _token_pos_label(itos[seq[i]], i)
            px_final, py_final = x_np[i, 0] + v_after[i, 0], x_np[i, 1] + v_after[i, 1]  # Final position
            end_color = '#2E7D32' if correct[i] else '#C62828'
            for c in range(n_cols):
                ax = axes[2, c]
                ax.text(px_final, py_final, lbl, fontsize=fs, fontweight='bold', ha='center', va='center', color='white', zorder=5,
                        path_effects=[pe.withStroke(linewidth=0.8, foreground=end_color)])

        # Row labels
        axes[0, 0].set_ylabel("V values\ndim 1", fontsize=label_fs)
        axes[1, 0].set_ylabel("Embed → Final\ndim 1", fontsize=label_fs)
        axes[2, 0].set_ylabel("Final\ndim 1", fontsize=label_fs)

        seq_str = " ".join(str(itos[t]) for t in seq[:25])
        if len(seq) > 25:
            seq_str += "..."
        gen_str = " ".join(str(itos[generated[i]]) for i in range(min(T, 25)))
        if T > 25:
            gen_str += "..."
        n_correct = correct.sum()
        print(f"Demo {seq_idx}  Sequence: {' '.join(str(itos[t]) for t in seq)}")
        print(f"Demo {seq_idx}  Generated (sampled): {' '.join(str(itos[generated[i]]) for i in range(T))}")
        print(f"Demo {seq_idx}  Pred next (argmax at each pos): {' '.join(str(itos[pred_next[i]]) for i in range(T))}")
        # Only add title for seq_idx != 1 (plot 16 is seq_idx=1, remove title)
        # For seq_idx==0 (plot 15) and seq_idx==2 (plot 17), remove only the "Generated: {gen_str}" part
        title_fontsize = 13 if (seq_idx == 0 or seq_idx == 2) else 9  # Bigger title font for plots 15 and 17
        if seq_idx != 1:
            if seq_idx == 0 or seq_idx == 2:
                # Remove only the generated sequence part
                fig.suptitle(f"Demo {seq_idx}: {seq_str}  |  Correct: {n_correct}/{T-1} (row 2: green=correct, red=wrong)", fontsize=title_fontsize, fontweight='bold', y=1.01)
            else:
                # Keep full title for other sequences (if any)
                fig.suptitle(f"Demo {seq_idx}: {seq_str}  |  Generated: {gen_str}  |  Correct: {n_correct}/{T-1} (row 2: green=correct, red=wrong)", fontsize=title_fontsize, fontweight='bold', y=1.01)
        plt.tight_layout()
        if save_dir:
            path = os.path.join(save_dir, f"v_before_after_demo_{seq_idx}.png")
            # Higher resolution for seq_idx=0 (plot 17)
            dpi_val = 300 if seq_idx == 0 else 150
            plt.savefig(path, bbox_inches='tight', dpi=dpi_val, facecolor='white')
            plt.close()
            print(f"V before/after demo figure saved to {path}")
        else:
            plt.show()

    if save_dir and sequences:
        print(f"Saved {min(len(sequences), len([s for s in sequences if len(s) >= 2]))} demo sequence figures to {save_dir}")


def plot_final_on_output_heatmap_grid(
    model, itos, sequence, save_path=None, grid_resolution=60, extent_margin=0.5
):
    """
    One figure for the given sequence: output-token probability heatmaps in a grid (not one column per token).
    Overlay only the *final* (embed + V_transformed) positions with labels; colors by (token, position) as in residuals plot.
    """
    import matplotlib.cm as cm
    model.eval()
    vocab_size = model.token_embedding.weight.shape[0]
    block_size = model.block_size
    n_embd = model.lm_head.in_features
    if n_embd != 2:
        print("plot_final_on_output_heatmap_grid: n_embd != 2. Skipping.")
        return
    if len(sequence) < 2:
        return
    seq = sequence[:block_size]
    T = len(seq)
    # (token, position) -> color, same scheme as residuals plot (tab20)
    token_positions = [(seq[i], i) for i in range(T)]
    unique_tp = sorted(set(token_positions))
    cmap = cm.get_cmap('tab20')
    colors_list = [cmap(i % 20) for i in range(len(unique_tp))]
    token_pos_to_color = {tp: colors_list[j] for j, tp in enumerate(unique_tp)}
    dev = next(model.parameters()).device
    idx = torch.tensor([seq], dtype=torch.long, device=dev)
    x_np, v_before, v_after = _get_v_before_after_for_sequence(model, idx)
    final_x = x_np + v_after  # (T, 2)

    with torch.no_grad():
        token_emb = model.token_embedding.weight.detach().cpu().numpy()
        pos_emb = model.position_embedding_table.weight.detach().cpu().numpy()
        combined = token_emb[:, None, :] + pos_emb[None, :, :]
        flat = combined.reshape(-1, 2)
        x_min = min(flat[:, 0].min(), final_x[:, 0].min()) - extent_margin
        x_max = max(flat[:, 0].max(), final_x[:, 0].max()) + extent_margin
        y_min = min(flat[:, 1].min(), final_x[:, 1].min()) - extent_margin
        y_max = max(flat[:, 1].max(), final_x[:, 1].max()) + extent_margin
    # Force square extent so heatmaps are square
    x_c, y_c = (x_min + x_max) / 2, (y_min + y_max) / 2
    half = max(x_max - x_min, y_max - y_min) / 2
    x_min, x_max = x_c - half, x_c + half
    y_min, y_max = y_c - half, y_c + half

    xs = np.linspace(x_min, x_max, grid_resolution)
    ys = np.linspace(y_min, y_max, grid_resolution)
    xx, yy = np.meshgrid(xs, ys)
    points = np.stack([xx.ravel(), yy.ravel()], axis=1)
    pts = torch.tensor(points, dtype=torch.float32, device=dev)
    with torch.no_grad():
        h = pts + model.ffwd(pts)
        logits = model.lm_head(h).cpu().numpy()
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)

    n_cols = min(4, vocab_size)
    n_rows = (vocab_size + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows), sharex=True, sharey=True)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for d in range(vocab_size):
        row, col = d // n_cols, d % n_cols
        ax = axes[row, col]
        Z = probs[:, d].reshape(grid_resolution, grid_resolution)
        ax.pcolormesh(xx, yy, Z, cmap='viridis', vmin=0, vmax=1, shading='auto')
        ax.axhline(y=0, color='white', linestyle='--', linewidth=0.5, alpha=0.3, zorder=3)
        ax.axvline(x=0, color='white', linestyle='--', linewidth=0.5, alpha=0.3, zorder=3)
        for i in range(T):
            lbl = _token_pos_label(itos[seq[i]], i)
            px, py = final_x[i, 0], final_x[i, 1]
            color = token_pos_to_color[(seq[i], i)]
            ax.text(px, py, lbl, fontsize=9, fontweight='bold', ha='center', va='center', color=color, zorder=5,
                    path_effects=[pe.withStroke(linewidth=0.8, foreground='black')])
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.set_title(f"P(next = {itos[d]})", fontsize=11)
        if row == n_rows - 1:
            ax.set_xlabel("dim 0", fontsize=9)
        if col == 0:
            ax.set_ylabel("dim 1", fontsize=9)

    for j in range(vocab_size, n_rows * n_cols):
        row, col = j // n_cols, j % n_cols
        axes[row, col].axis('off')

    seq_str = " ".join(str(itos[t]) for t in seq[:20])
    if len(seq) > 20:
        seq_str += "..."
    fig.suptitle(f"Final (embed+V_transformed) on output heatmaps  |  {seq_str}", fontsize=11, fontweight='bold', y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()
        print(f"Final-on-output heatmap grid saved to {save_path}")
    else:
        plt.show()
    model.train()


@torch.no_grad()
def plot_probability_heatmap_with_embeddings(
    model, itos, save_path=None, grid_resolution=80, extent_margin=0.5, step_label: int | None = None
):
    """
    Plot probability heatmaps for each token with all token+position combinations from
    the original embedding space overlaid on top.
    
    Args:
        model: Trained model (BigramLanguageModel)
        itos: Index-to-string mapping for tokens
        save_path: Path to save the figure
        grid_resolution: Number of points per axis (default 80)
        extent_margin: Extra margin around embedding extent (default 0.5)
    """
    model.eval()
    vocab_size = model.token_embedding.weight.shape[0]
    block_size = model.position_embedding_table.weight.shape[0]
    n_embd = model.lm_head.in_features
    if n_embd != 2:
        print(f"plot_probability_heatmap_with_embeddings: n_embd={n_embd}, need 2. Skipping.")
        return

    with torch.no_grad():
        W = model.lm_head.weight.detach().cpu().numpy()   # (vocab_size, 2)
        b = model.lm_head.bias.detach().cpu().numpy()     # (vocab_size,)
        token_emb = model.token_embedding.weight.detach().cpu().numpy()
        pos_emb = model.position_embedding_table.weight.detach().cpu().numpy()
        combined = token_emb[:, None, :] + pos_emb[None, :, :]  # (vocab, block, 2)
        flat = combined.reshape(-1, 2)
        x_min, x_max = flat[:, 0].min() - extent_margin, flat[:, 0].max() + extent_margin
        y_min, y_max = flat[:, 1].min() - extent_margin, flat[:, 1].max() + extent_margin
    # Square extent and layout (same as fig 19 / probability_heatmap_with_values)
    x_c, y_c = (x_min + x_max) / 2, (y_min + y_max) / 2
    half = max(x_max - x_min, y_max - y_min) / 2
    x_min, x_max = x_c - half, x_c + half
    y_min, y_max = y_c - half, y_c + half

    # Create probability grid - need to pass through feedforward first
    xs = np.linspace(x_min, x_max, grid_resolution)
    ys = np.linspace(y_min, y_max, grid_resolution)
    xx, yy = np.meshgrid(xs, ys)
    points = np.stack([xx.ravel(), yy.ravel()], axis=1)  # (N, 2)
    
    # Pass through feedforward + residual, then lm_head (same as v_before_after_demo)
    dev = next(model.parameters()).device
    with torch.no_grad():
        pts = torch.tensor(points, dtype=torch.float32, device=dev)
        h = pts + model.ffwd(pts)  # Feedforward + residual
        logits = model.lm_head(h).cpu().numpy()  # (N, vocab_size)
    
    # Compute probabilities from logits
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)             # (N, vocab_size)

    # Get all token+position combinations
    all_combinations = []
    labels = []
    for token_idx in range(vocab_size):
        for pos_idx in range(block_size):
            emb = token_emb[token_idx] + pos_emb[pos_idx]
            all_combinations.append(emb)
            labels.append(_token_pos_label(itos[token_idx], pos_idx))
    all_combinations = np.array(all_combinations)  # (vocab_size * block_size, 2)

    # Create figure with one subplot per token
    n_cols = min(4, vocab_size)
    n_rows = (vocab_size + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows), sharex=True, sharey=True)
    if step_label is not None:
        fig.suptitle(f"Step: {step_label}", fontsize=16, fontweight="bold", y=0.98)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for token_idx in range(vocab_size):
        row = token_idx // n_cols
        col = token_idx % n_cols
        ax = axes[row, col]
        
        # Plot probability heatmap
        Z = probs[:, token_idx].reshape(grid_resolution, grid_resolution)
        im = ax.pcolormesh(xx, yy, Z, cmap='viridis', vmin=0, vmax=1, shading='auto')
        
        # Overlay annotations only (no marker dots/circles)
        for combo_idx, (emb, label) in enumerate(zip(all_combinations, labels)):
            if vocab_size * block_size <= 200:
                ax.text(emb[0], emb[1], label, fontsize=7, ha='center', va='center',
                       color='white', weight='bold', zorder=6)
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.set_title(f"P(next = {itos[token_idx]})", fontsize=10)
        if row == n_rows - 1:
            ax.set_xlabel("dim 0", fontsize=9)
        if col == 0:
            ax.set_ylabel("dim 1", fontsize=9)

    # Hide unused subplots
    for token_idx in range(vocab_size, n_rows * n_cols):
        row = token_idx // n_cols
        col = token_idx % n_cols
        axes[row, col].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()
        print(f"Probability heatmap with embeddings saved to {save_path}")
    else:
        plt.show()
    
    model.train()


@torch.no_grad()
def plot_probability_heatmap_with_values(
    model, itos, save_path=None, grid_resolution=80, extent_margin=0.5, step_label: int | None = None
):
    """
    Plot probability heatmaps for each token with all token+position V values
    (W_V @ embedding) overlaid on top, analogous to plot_probability_heatmap_with_embeddings.
    
    Args:
        model: Trained model (BigramLanguageModel)
        itos: Index-to-string mapping for tokens
        save_path: Path to save the figure
        grid_resolution: Number of points per axis (default 80)
        extent_margin: Extra margin around extent (default 0.5)
    """
    model.eval()
    vocab_size = model.token_embedding.weight.shape[0]
    block_size = model.position_embedding_table.weight.shape[0]
    n_embd = model.lm_head.in_features
    if n_embd != 2:
        print(f"plot_probability_heatmap_with_values: n_embd={n_embd}, need 2. Skipping.")
        return

    head = model.sa_heads.heads[0]
    W_V = head.value.weight.detach().cpu().numpy()  # (head_size, n_embd)

    with torch.no_grad():
        W = model.lm_head.weight.detach().cpu().numpy()   # (vocab_size, 2)
        b = model.lm_head.bias.detach().cpu().numpy()     # (vocab_size,)
        token_emb = model.token_embedding.weight.detach().cpu().numpy()
        pos_emb = model.position_embedding_table.weight.detach().cpu().numpy()
        combined = token_emb[:, None, :] + pos_emb[None, :, :]  # (vocab, block, 2)
        flat = combined.reshape(-1, 2)
        emb_x_min, emb_x_max = flat[:, 0].min(), flat[:, 0].max()
        emb_y_min, emb_y_max = flat[:, 1].min(), flat[:, 1].max()

    # Get all V values: V = W_V @ (token_emb + pos_emb) for each (token, pos)
    all_V = []
    labels = []
    for token_idx in range(vocab_size):
        for pos_idx in range(block_size):
            emb = token_emb[token_idx] + pos_emb[pos_idx]
            v = (W_V @ emb)  # (head_size,)
            all_V.append(v)
            labels.append(_token_pos_label(itos[token_idx], pos_idx))
    all_V = np.array(all_V)  # (vocab_size * block_size, 2)

    # Grid extent: union of embedding and V extents (so both are visible)
    v_x_min, v_x_max = all_V[:, 0].min(), all_V[:, 0].max()
    v_y_min, v_y_max = all_V[:, 1].min(), all_V[:, 1].max()
    x_min = min(emb_x_min, v_x_min) - extent_margin
    x_max = max(emb_x_max, v_x_max) + extent_margin
    y_min = min(emb_y_min, v_y_min) - extent_margin
    y_max = max(emb_y_max, v_y_max) + extent_margin

    # Create probability grid
    xs = np.linspace(x_min, x_max, grid_resolution)
    ys = np.linspace(y_min, y_max, grid_resolution)
    xx, yy = np.meshgrid(xs, ys)
    points = np.stack([xx.ravel(), yy.ravel()], axis=1)  # (N, 2)

    dev = next(model.parameters()).device
    with torch.no_grad():
        pts = torch.tensor(points, dtype=torch.float32, device=dev)
        h = pts + model.ffwd(pts)
        logits = model.lm_head(h).cpu().numpy()  # (N, vocab_size)

    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)

    n_cols = min(4, vocab_size)
    n_rows = (vocab_size + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows), sharex=True, sharey=True)
    if step_label is not None:
        fig.suptitle(f"Step: {step_label}", fontsize=16, fontweight="bold", y=0.98)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for token_idx in range(vocab_size):
        row = token_idx // n_cols
        col = token_idx % n_cols
        ax = axes[row, col]

        Z = probs[:, token_idx].reshape(grid_resolution, grid_resolution)
        ax.pcolormesh(xx, yy, Z, cmap='viridis', vmin=0, vmax=1, shading='auto')

        # Overlay annotations only (no marker dots/circles)
        for v_vec, label in zip(all_V, labels):
            if vocab_size * block_size <= 200:
                ax.text(v_vec[0], v_vec[1], label, fontsize=7, ha='center', va='center',
                       color='white', weight='bold', zorder=6)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.set_title(f"P(next = {itos[token_idx]})", fontsize=10)
        if row == n_rows - 1:
            ax.set_xlabel("dim 0", fontsize=9)
        if col == 0:
            ax.set_ylabel("dim 1", fontsize=9)

    for token_idx in range(vocab_size, n_rows * n_cols):
        row = token_idx // n_cols
        col = token_idx % n_cols
        axes[row, col].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()
        print(f"Probability heatmap with V values saved to {save_path}")
    else:
        plt.show()

    model.train()


def plot_qk_full_attention_heatmap(model, itos, save_path: str = None):
    """
    Create a heatmap showing attention scores for ALL token-position combinations.
    X-axis: Key (token-position), Y-axis: Query (token-position), Value: Q·K
    
    Args:
        model: Trained TransformerLM
        itos: Index-to-string mapping for tokens
        save_path: Path to save the figure
    """
    model.eval()
    
    # Get model parameters
    vocab_size = model.token_embedding.weight.shape[0]
    block_size = model.position_embedding_table.weight.shape[0]
    
    # Get the first attention head's Q, K weights
    head = model.sa_heads.heads[0]
    W_Q = head.query.weight.detach().cpu().numpy()
    W_K = head.key.weight.detach().cpu().numpy()
    head_size = W_Q.shape[0]
    
    # Get embeddings
    token_emb = model.token_embedding.weight.detach().cpu().numpy()
    pos_emb = model.position_embedding_table.weight.detach().cpu().numpy()
    
    # Compute Q and K for all token-position combinations
    num_combinations = vocab_size * block_size
    Q_all = np.zeros((num_combinations, head_size))
    K_all = np.zeros((num_combinations, head_size))
    labels = []
    
    idx = 0
    for t in range(vocab_size):
        for p in range(block_size):
            combined_emb = token_emb[t] + pos_emb[p]
            Q_all[idx] = W_Q @ combined_emb
            K_all[idx] = W_K @ combined_emb
            labels.append(_token_pos_label(itos[t], p))
            idx += 1
    
    # Compute full attention matrix: Q_all @ K_all.T / sqrt(d)
    attention_matrix = (Q_all @ K_all.T) / np.sqrt(head_size)
    
    # Apply CAUSAL MASKING: query at position p can only attend to keys at position <= p
    query_positions = np.array([p for t in range(vocab_size) for p in range(block_size)])
    key_positions = np.array([p for t in range(vocab_size) for p in range(block_size)])
    
    # Create causal mask: mask[i,j] = True if query_pos[i] >= key_pos[j] (can attend)
    causal_mask = query_positions[:, None] >= key_positions[None, :]  # (132, 132)
    
    # Apply mask: set invalid positions to NaN (will show as white/transparent)
    masked_attention = np.where(causal_mask, attention_matrix, np.nan)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(20, 16))
    
    # Use 'nipy_spectral' - maximum color divergence across full spectrum
    im = ax.imshow(masked_attention, cmap='nipy_spectral', aspect='auto')
    
    # Simple token labels at center of each token block
    xtick_positions = []
    xtick_labels = []
    ytick_positions = []
    ytick_labels = []
    
    for t in range(vocab_size):
        mid_pos = t * block_size + block_size // 2
        xtick_positions.append(mid_pos)
        xtick_labels.append(itos[t])
        ytick_positions.append(mid_pos)
        ytick_labels.append(itos[t])
    
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels, fontsize=9, fontweight='bold')
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(ytick_labels, fontsize=9, fontweight='bold')
    
    # Add grid lines to separate token groups
    for t in range(vocab_size + 1):
        ax.axhline(y=t * block_size - 0.5, color='white', linewidth=1.5, alpha=0.8)
        ax.axvline(x=t * block_size - 0.5, color='white', linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel("Key Token", fontsize=12)
    ax.set_ylabel("Query Token", fontsize=12)
    ax.set_title(f"Full Attention Matrix: Q·K / √{head_size}\n({vocab_size} tokens × {block_size} positions, positions 0->{block_size-1} top-to-bottom within each token block)", 
                fontsize=14, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label("Attention Score (pre-softmax)", fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()
        print(f"Full Q/K attention heatmap saved to {save_path}")
    else:
        plt.show()


def plot_qk_softmax_attention_heatmap(model, itos, save_path: str = None):
    """
    Create a heatmap showing SOFTMAX attention weights for ALL token-position combinations.
    This shows the actual attention probabilities after softmax normalization.
    """
    model.eval()
    
    vocab_size = model.token_embedding.weight.shape[0]
    block_size = model.position_embedding_table.weight.shape[0]
    
    head = model.sa_heads.heads[0]
    W_Q = head.query.weight.detach().cpu().numpy()
    W_K = head.key.weight.detach().cpu().numpy()
    head_size = W_Q.shape[0]
    
    token_emb = model.token_embedding.weight.detach().cpu().numpy()
    pos_emb = model.position_embedding_table.weight.detach().cpu().numpy()
    
    num_combinations = vocab_size * block_size
    Q_all = np.zeros((num_combinations, head_size))
    K_all = np.zeros((num_combinations, head_size))
    
    idx = 0
    for t in range(vocab_size):
        for p in range(block_size):
            combined_emb = token_emb[t] + pos_emb[p]
            Q_all[idx] = W_Q @ combined_emb
            K_all[idx] = W_K @ combined_emb
            idx += 1
    
    # Compute attention scores
    attention_scores = (Q_all @ K_all.T) / np.sqrt(head_size)
    
    # Apply CAUSAL MASKING: query at (token_q, pos_q) can only attend to keys at pos_k <= pos_q
    # Build position indices for masking
    query_positions = np.array([p for t in range(vocab_size) for p in range(block_size)])  # position of each query
    key_positions = np.array([p for t in range(vocab_size) for p in range(block_size)])    # position of each key
    
    # Create causal mask: mask[i,j] = True if query_pos[i] >= key_pos[j] (can attend)
    causal_mask = query_positions[:, None] >= key_positions[None, :]  # (132, 132)
    
    # Apply mask: set invalid positions to -inf before softmax
    masked_scores = np.where(causal_mask, attention_scores, -np.inf)
    
    # Apply softmax to each row (each query attends to valid keys only)
    def softmax(x):
        # Handle rows that are all -inf (query at position 0 with no valid keys of same token)
        x_max = np.max(x, axis=-1, keepdims=True)
        x_max = np.where(np.isinf(x_max), 0, x_max)  # replace -inf max with 0
        exp_x = np.exp(x - x_max)
        exp_x = np.where(np.isinf(x), 0, exp_x)  # -inf -> 0 after exp
        sum_exp = np.sum(exp_x, axis=-1, keepdims=True)
        sum_exp = np.where(sum_exp == 0, 1, sum_exp)  # avoid division by zero
        return exp_x / sum_exp
    
    attention_probs = softmax(masked_scores)
    
    
    # Create figure
    fig, ax = plt.subplots(figsize=(20, 16))
    
    # Use nipy_spectral for max color divergence
    im = ax.imshow(attention_probs, cmap='nipy_spectral', aspect='auto')
    
    # Simple token labels
    xtick_positions = []
    xtick_labels = []
    ytick_positions = []
    ytick_labels = []
    
    for t in range(vocab_size):
        mid_pos = t * block_size + block_size // 2
        xtick_positions.append(mid_pos)
        xtick_labels.append(itos[t])
        ytick_positions.append(mid_pos)
        ytick_labels.append(itos[t])
    
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels, fontsize=9, fontweight='bold')
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(ytick_labels, fontsize=9, fontweight='bold')
    
    # Grid lines to separate token groups
    for t in range(vocab_size + 1):
        ax.axhline(y=t * block_size - 0.5, color='white', linewidth=1.5, alpha=0.8)
        ax.axvline(x=t * block_size - 0.5, color='white', linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel("Key Token", fontsize=12)
    ax.set_ylabel("Query Token", fontsize=12)
    ax.set_title(f"Softmax Attention Probabilities\n({vocab_size} tokens × {block_size} positions, positions 0->{block_size-1} top-to-bottom)", 
                fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label("Attention Probability (after softmax)", fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()
        print(f"Softmax attention heatmap saved to {save_path}")
    else:
        plt.show()


def plot_qk_space_and_attention_heatmap(model, itos, save_path: str = None, step_label: int | None = None):
    """
    Combined visualization:
    - Left: Q/K embedding space (queries in blue, keys in red) for all token–position pairs
    - Right: full pre-softmax attention matrix Q·K with causal masking

    This is designed specifically for learning-dynamics videos so we can see how
    the Q/K geometry and the induced attention pattern co-evolve over training.
    """
    model.eval()

    # Model parameters
    vocab_size = model.token_embedding.weight.shape[0]
    block_size = model.position_embedding_table.weight.shape[0]

    head = model.sa_heads.heads[0]
    W_Q = head.query.weight.detach().cpu().numpy()
    W_K = head.key.weight.detach().cpu().numpy()
    head_size = W_Q.shape[0]

    token_emb = model.token_embedding.weight.detach().cpu().numpy()
    pos_emb = model.position_embedding_table.weight.detach().cpu().numpy()

    # Compute Q and K for all token-position combinations
    num_combinations = vocab_size * block_size
    Q_all = np.zeros((num_combinations, head_size))
    K_all = np.zeros((num_combinations, head_size))
    labels = []

    idx = 0
    for t in range(vocab_size):
        for p in range(block_size):
            combined_emb = token_emb[t] + pos_emb[p]
            Q_all[idx] = W_Q @ combined_emb
            K_all[idx] = W_K @ combined_emb
            labels.append(_token_pos_label(itos[t], p))
            idx += 1

    # 2D projection for Q/K scatter (direct if head_size==2, PCA otherwise)
    if head_size == 2:
        Q_2d = Q_all
        K_2d = K_all
    else:
        from sklearn.decomposition import PCA
        combined = np.vstack([Q_all, K_all])
        pca = PCA(n_components=2)
        combined_2d = pca.fit_transform(combined)
        Q_2d = combined_2d[:num_combinations]
        K_2d = combined_2d[num_combinations:]

    # Full attention matrix Q·K / sqrt(d)
    attention_matrix = (Q_all @ K_all.T) / np.sqrt(head_size)

    # Causal masking: query at position p can only attend to keys at position <= p
    query_positions = np.array([p for t in range(vocab_size) for p in range(block_size)])
    key_positions = np.array([p for t in range(vocab_size) for p in range(block_size)])
    causal_mask = query_positions[:, None] >= key_positions[None, :]
    masked_attention = np.where(causal_mask, attention_matrix, np.nan)

    # Figure with two panels: left scatter, right heatmap
    fig, (ax_scatter, ax_heat) = plt.subplots(1, 2, figsize=(24, 10))
    if step_label is not None:
        fig.suptitle(f"Step: {step_label}", fontsize=18, fontweight="bold", y=0.98)

    # --- Left: Q/K embedding space ---
    all_x = np.concatenate([Q_2d[:, 0], K_2d[:, 0]])
    all_y = np.concatenate([Q_2d[:, 1], K_2d[:, 1]])
    ax_scatter.scatter(all_x, all_y, s=0, alpha=0)

    label_fontsize = 9
    axis_fontsize = 12

    for i in range(num_combinations):
        ax_scatter.text(
            Q_2d[i, 0],
            Q_2d[i, 1],
            labels[i],
            fontsize=label_fontsize,
            ha="center",
            va="center",
            color="blue",
        )
    for i in range(num_combinations):
        ax_scatter.text(
            K_2d[i, 0],
            K_2d[i, 1],
            labels[i],
            fontsize=label_fontsize,
            ha="center",
            va="center",
            color="red",
        )

    ax_scatter.set_xlabel("Q/K dim 1" + (" (PCA)" if head_size != 2 else ""), fontsize=axis_fontsize)
    ax_scatter.set_ylabel("Q/K dim 2" + (" (PCA)" if head_size != 2 else ""), fontsize=axis_fontsize)
    ax_scatter.set_title("Q/K embedding space", fontsize=axis_fontsize + 2, fontweight="bold")
    ax_scatter.grid(True, alpha=0.3)

    # --- Right: attention heatmap ---
    im = ax_heat.imshow(masked_attention, cmap="nipy_spectral", aspect="auto")

    xtick_positions = []
    xtick_labels = []
    ytick_positions = []
    ytick_labels = []
    for t in range(vocab_size):
        mid_pos = t * block_size + block_size // 2
        xtick_positions.append(mid_pos)
        xtick_labels.append(itos[t])
        ytick_positions.append(mid_pos)
        ytick_labels.append(itos[t])

    ax_heat.set_xticks(xtick_positions)
    ax_heat.set_xticklabels(xtick_labels, fontsize=8, fontweight="bold", rotation=45, ha="right")
    ax_heat.set_yticks(ytick_positions)
    ax_heat.set_yticklabels(ytick_labels, fontsize=8, fontweight="bold")

    for t in range(vocab_size + 1):
        ax_heat.axhline(y=t * block_size - 0.5, color="white", linewidth=1.0, alpha=0.8)
        ax_heat.axvline(x=t * block_size - 0.5, color="white", linewidth=1.0, alpha=0.8)

    ax_heat.set_xlabel("Key token", fontsize=axis_fontsize)
    ax_heat.set_ylabel("Query token", fontsize=axis_fontsize)
    ax_heat.set_title(f"Full attention matrix Q·K / √{head_size} (causal masked)", fontsize=axis_fontsize + 2, fontweight="bold")

    cbar = plt.colorbar(im, ax=ax_heat, shrink=0.7)
    cbar.set_label("Attention score (pre-softmax)", fontsize=axis_fontsize - 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150, facecolor="white")
        plt.close(fig)
        print(f"Q/K space + attention heatmap saved to {save_path}")
    else:
        plt.show()


# -----------------------------
# Checkpoint saving/loading
