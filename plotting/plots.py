"""Plotting functions for embeddings, attention, heatmaps, architecture."""
import os
import numpy as np
import torch
from torch.nn import functional as F
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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
        if num_items <= 12:
            return 20
        elif num_items <= 20:
            return 16
        elif num_items <= 40:
            return 12
        elif num_items <= 80:
            return 10
        elif num_items <= 150:
            return 8
        else:
            return 6
    
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
                ax6.text(X2_pos[i, 0], X2_pos[i, 1], f"p{i}", fontsize=pos_fontsize, fontweight='bold',
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
                ax6.text(X_pos[i, 0], X_pos[i, 1], f"p{i}", fontsize=pos_fontsize, fontweight='bold',
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
                ax6.text(X1_pos[i], i, f"p{i}", fontsize=pos_fontsize, fontweight='bold',
                        ha='center', va='center', color=pos_colors[i])
    
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
                label = f"{token_str}p{pos_idx}"
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
                label = f"{token_str}p{pos_idx}"
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
                label = f"{token_str}p{pos_idx}"
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
        if num_items <= 12:
            return 22
        elif num_items <= 20:
            return 18
        elif num_items <= 40:
            return 14
        elif num_items <= 80:
            return 11
        elif num_items <= 150:
            return 9
        else:
            return 7
    
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
                ax2.text(X2_pos[i, 0], X2_pos[i, 1], f"p{i}", fontsize=pos_fontsize, fontweight='bold',
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
                ax2.text(X2_pos[i, 0], X2_pos[i, 1], f"p{i}", fontsize=pos_fontsize, fontweight='bold',
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
                ax2.text(X1_pos[i], i, f"p{i}", fontsize=pos_fontsize, fontweight='bold',
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
                label = f"{token_str}p{pos_idx}"
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
                label = f"{token_str}p{pos_idx}"
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
                label = f"{token_str}p{pos_idx}"
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
        if num_items <= 12:
            return 18
        elif num_items <= 20:
            return 14
        elif num_items <= 40:
            return 11
        elif num_items <= 80:
            return 9
        elif num_items <= 150:
            return 7
        else:
            return 5
    
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
            combo_labels.append(f"{itos[token_idx]}p{pos_idx}")
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
        ax2.text(x_pos[i], y_pos[i], f"p{i}", fontsize=fs_pos, fontweight='bold',
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
        # For video frames, use fixed size (no bbox_inches='tight') to ensure consistent dimensions
        if step_label is not None:
            plt.savefig(save_path, dpi=150)
        else:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
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
        
        # Combine Q and K data to set axis limits properly
        all_data_2d = np.vstack([Q_2d, K_2d])
        x_min, x_max = all_data_2d[:, 0].min(), all_data_2d[:, 0].max()
        y_min, y_max = all_data_2d[:, 1].min(), all_data_2d[:, 1].max()
        x_margin = (x_max - x_min) * 0.1
        y_margin = (y_max - y_min) * 0.1
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        
        # Annotate all points with token and position (markers removed, keeping only annotations)
        for i, (token, pos) in enumerate(zip(tokens, range(len(tokens)))):
            # Annotate Q points (blue for Query)
            ax.text(Q_2d[i, 0], Q_2d[i, 1], f'{token}p{pos}', 
                   fontsize=14, fontweight='bold', ha='center', va='center', color='blue')
            # Annotate K points (red for Key)
            ax.text(K_2d[i, 0], K_2d[i, 1], f'{token}p{pos}', 
                   fontsize=14, fontweight='bold', ha='center', va='center', color='red')
        
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
            # Annotate V points (blue)
            ax.text(V_2d[i, 0], V_2d[i, 1], f'{token}p{pos}', 
                   fontsize=14, fontweight='bold', ha='center', va='center', color='blue')
            # Annotate Final Output points (red)
            ax.text(Final_Output_2d[i, 0], Final_Output_2d[i, 1], f'{token}p{pos}', 
                   fontsize=14, fontweight='bold', ha='center', va='center', color='red')
        
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
        # Create figure - scale appropriately for number of sequences
        fig_height = max(6, min(20, num_sequences * 0.6 + 2))
        fig_width = min(24, max(12, max_len * 0.5))
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
    
    # Add text annotations (the actual numbers)
    # Adjust font size based on number of sequences
    fontsize = max(7, min(10, 12 - num_sequences // 5))
    for i in range(len(sequences_to_show)):
        for j in range(max_len):
            val = data_matrix[i, j]
            if val is not None:  # Not padding
                text_color = 'black'
                ax.text(j, i, str(val), ha='center', va='center', fontsize=fontsize, color=text_color, fontweight='bold')
    
    # Set labels
    ax.set_xlabel("Position in Sequence", fontsize=11)
    ax.set_ylabel("Sequence #", fontsize=11)
    ax.set_title(title or "Generated Sequences with Rule Correctness\n(Green = Correct, Red = Incorrect)", fontsize=12)
    
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

def plot_generated_sequences_heatmap_before_after(generated_sequences_e0, generated_sequences_final, generator, save_path=None, num_sequences=5, max_length=50):
    """Plot before/after generated sequences heatmaps in one image."""
    if not generated_sequences_e0:
        return plot_generated_sequences_heatmap(
            generated_sequences_final, generator,
            save_path=save_path, num_sequences=num_sequences, max_length=max_length
        )
    
    # Create a 2-row figure (E0 on top, Final on bottom)
    fig_height = max(9, min(24, num_sequences * 1.2 + 4))
    fig_width = 24
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

def plot_training_data_heatmap(training_sequences, generator, save_path=None, num_sequences=50, max_length=50):
    """
    Plot training data sequences as an annotated heatmap showing correctness.
    Red = incorrect, Green = correct.
    Same style as generated sequences heatmap.
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
    
    # Create figure - scale appropriately for number of sequences
    fig_height = max(8, min(30, num_sequences * 0.4 + 2))
    fig_width = min(24, max(12, max_len * 0.4))
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
    
    # Add text annotations (the actual numbers)
    # Adjust font size based on number of sequences
    fontsize = max(6, min(9, 11 - num_sequences // 10))
    for i in range(len(sequences_to_show)):
        for j in range(max_len):
            val = data_matrix[i, j]
            if val is not None:  # Not padding
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
    """Generate a visual diagram of the model architecture.
    
    Args:
        config: Configuration dictionary with model parameters (used as fallback)
        save_path: Path to save the diagram (if None, displays interactively)
        model: Optional model instance to extract architecture from dynamically
        vocab_size: Optional vocab size (extracted from model if not provided)
        batch_size: Optional batch size (from config if not provided)
    """
    # Extract architecture from model if provided, otherwise use config
    has_ln1 = False
    has_ln2 = False
    
    if model is not None:
        # Inspect actual model structure
        has_ln1 = hasattr(model, 'ln1') and 'ln1' in model._modules
        has_ln2 = hasattr(model, 'ln2') and 'ln2' in model._modules
        
        # Extract dimensions directly from model
        vocab_size_model = model.token_embedding.num_embeddings
        n_embd = model.token_embedding.embedding_dim
        block_size = model.block_size
        num_heads = len(model.sa_heads.heads)
        # Get head_size from first head
        head_size = model.sa_heads.heads[0].head_size
        # Check if projection exists
        has_proj = hasattr(model, 'proj') and 'proj' in model._modules
        if has_proj:
            proj_in_dim = model.proj.in_features
            proj_out_dim = model.proj.out_features
        else:
            proj_in_dim = num_heads * head_size
            proj_out_dim = n_embd
        # Get feedforward dimensions
        ffwd_net = model.ffwd.net
        ffwd_in_dim = ffwd_net[0].in_features
        ffwd_hidden_dim = ffwd_net[0].out_features
        ffwd_out_dim = ffwd_net[2].out_features
        ffwd_mult = ffwd_hidden_dim // ffwd_in_dim
        # Get LM head dimensions
        lm_head_in_dim = model.lm_head.in_features
        lm_head_out_dim = model.lm_head.out_features
        
        # Use provided vocab_size or model's vocab_size
        if vocab_size is None:
            vocab_size = vocab_size_model
        
        # Verify consistency
        if has_proj:
            assert proj_in_dim == num_heads * head_size, f"Projection input dim mismatch: {proj_in_dim} != {num_heads * head_size}"
            assert proj_out_dim == n_embd, f"Projection output dim mismatch: {proj_out_dim} != {n_embd}"
        else:
            # Without projection, attention output must match n_embd
            assert num_heads * head_size == n_embd, f"Without projection, attention output dim {num_heads * head_size} must equal n_embd {n_embd}"
        assert ffwd_in_dim == n_embd, f"FFN input dim mismatch: {ffwd_in_dim} != {n_embd}"
        assert ffwd_out_dim == n_embd, f"FFN output dim mismatch: {ffwd_out_dim} != {n_embd}"
        assert lm_head_in_dim == n_embd, f"LM head input dim mismatch: {lm_head_in_dim} != {n_embd}"
        assert lm_head_out_dim == vocab_size, f"LM head output dim mismatch: {lm_head_out_dim} != {vocab_size}"
    else:
        # Fall back to config
        model_config = config['model']
        data_config = config['data']
        training_config = config.get('training', {})
        
        if vocab_size is None:
            vocab_size = data_config['max_value'] - data_config['min_value'] + 1
            if data_config.get('generator_type') in ['PlusLastEvenRule']:
                vocab_size += 1
        
        n_embd = model_config['n_embd']
        block_size = model_config['block_size']
        num_heads = model_config['num_heads']
        head_size = model_config['head_size']
        ffwd_mult = 16  # Default from model.py
        ffwd_hidden_dim = n_embd * ffwd_mult
        has_proj = True  # Assume projection exists in config-based mode (for backward compat)
    
    # Get batch_size
    if batch_size is None:
        training_config = config.get('training', {})
        batch_size = training_config.get('batch_size', 4)
    
    fig, ax = plt.subplots(1, 1, figsize=(24, 10))
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    
    input_color = '#E8F4FD'
    embed_color = '#D4E6F1'
    attention_color = '#FCF3CF'
    linear_color = '#FADBD8'
    output_color = '#D5F5E3'
    norm_color = '#E8DAEF'
    arrow_color = '#2C3E50'
    
    def draw_box(x, y, w, h, text, color, fontsize=9, subtext=None, mathtext=False):
        rect = plt.Rectangle((x - w/2, y - h/2), w, h,
                              facecolor=color, edgecolor='#2C3E50', linewidth=2, zorder=2)
        ax.add_patch(rect)
        # Handle multi-line text by splitting on newlines or wrapping long text
        if '\n' in text:
            lines = text.split('\n')
            line_height = 0.3
            start_y = y + (line_height * (len(lines) - 1) / 2) - (0.15 if subtext else 0)
            for i, line in enumerate(lines):
                ax.text(x, start_y - i * line_height, line, ha='center', va='center',
                        fontsize=fontsize, fontweight='bold', zorder=3, usetex=False)
        else:
            ax.text(x, y + (0.15 if subtext else 0), text, ha='center', va='center',
                    fontsize=fontsize, fontweight='bold', zorder=3, usetex=False)
        if subtext:
            # Handle multi-line subtext
            if '\n' in subtext:
                sublines = subtext.split('\n')
                subline_height = 0.22
                # Calculate total height needed for all subtext lines
                total_subtext_height = subline_height * (len(sublines) - 1)
                # Center the subtext block vertically within the bottom portion of the box
                # Position it in the lower half of the box, centered
                bottom_margin = 0.15
                available_height = h/2 - bottom_margin
                start_y = y - h/2 + bottom_margin + (available_height - total_subtext_height)/2 + total_subtext_height
                for i, line in enumerate(sublines):
                    ax.text(x, start_y - i * subline_height, line, ha='center', va='center', 
                            fontsize=7, color='#555', zorder=3)
            else:
                ax.text(x, y - 0.25, subtext, ha='center', va='center', fontsize=7, color='#555', zorder=3)
    
    def draw_arrow(x1, y1, x2, y2, label=None):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.5))
        if label:
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mid_x + 0.3, mid_y, label, fontsize=7, color='#555')
    
    def residual_arrow(x1, y1, x2, y2, label='residual'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#27AE60', lw=1.5,
                                   connectionstyle='arc3,rad=0.25'))
    
    config_name = config.get('name', 'Model')
    ax.text(12, 9.2, f"Architecture: {config_name}", ha='center', va='center',
            fontsize=14, fontweight='bold')
    
    # Notation (define B, T, C, d_k, vocab; show actual dims) - place on right side
    ax.text(20, 8.5, "Notation (symbol -> actual)", ha='center', va='center', fontsize=9, fontweight='bold')
    for i, line in enumerate([
        "B = batch size -> " + str(batch_size),
        "T = seq length -> " + str(block_size),
        "C = n_embd -> " + str(n_embd),
        r"d$_k$ = head_size -> " + str(head_size),
        "vocab -> " + str(vocab_size),
    ]):
        ax.text(20, 7.8 - 0.3 * i, line, ha='center', va='center', fontsize=7, color='#333')
    
    # 1. Input
    x_in = 1
    draw_box(x_in, 5, 1.0, 3, "Input\nTokens", input_color,
             subtext=f"(B, T) = ({batch_size}, {block_size})")
    
    # 2. Token & Position Embedding
    x_emb = 2.5
    draw_box(x_emb, 3.5, 1.3, 3.4, "Token\nEmbedding", embed_color,
             subtext=f"Embedding\n({vocab_size}, {n_embd})")
    draw_box(x_emb, 6.5, 1.3, 3.4, "Position\nEmbedding", embed_color,
             subtext=f"Embedding\n({block_size}, {n_embd})")
    draw_arrow(1.4, 5, 2, 3.5)
    draw_arrow(1.4, 5, 2, 6.5)
    
    # 3. Add embeddings
    x_add = 4
    draw_box(x_add, 5, 0.7, 2.2, "+", '#FFF', fontsize=14)
    draw_arrow(3.1, 3.5, 3.65, 4.5)
    draw_arrow(3.1, 6.5, 3.65, 5.5)
    ax.text(x_add, 6.3, rf"(B,T,C)=({batch_size},{block_size},{n_embd})", fontsize=6.5, ha='center', rotation=0)
    
    # 4. LayerNorm (Pre-LN before attention) - only if it exists
    if has_ln1:
        x_ln1 = 5.3
        draw_box(x_ln1, 5, 0.8, 2.2, "LayerNorm", norm_color, fontsize=9,
                 subtext=f"(B, T, {n_embd})")
        draw_arrow(4.3, 5, 4.9, 5)
        attn_input_x = 4.9
        attn_center = 8.5
    else:
        attn_input_x = 4.3
        attn_center = 7.5  # Move attention block closer when no LayerNorm
    
    # 5. Multi-Head Attention block (horizontal layout)
    attn_w = 5.5
    attn_left = attn_center - attn_w / 2
    attn_right = attn_center + attn_w / 2
    rect = plt.Rectangle((attn_left, 1.4), attn_w, 7.2,
                          facecolor='#FEF9E7', edgecolor='#F39C12', linewidth=2,
                          linestyle='--', zorder=1)
    ax.add_patch(rect)
    ax.text(attn_center, 8.35, f"Multi-Head Attention ({num_heads} head{'s' if num_heads > 1 else ''})",
            ha='center', va='center', fontsize=10, fontweight='bold', color='#B7950B', zorder=3)
    
    draw_arrow(attn_input_x, 5, attn_left + 0.5, 5)
    
    qkv_x = 7.5
    draw_box(qkv_x, 3.5, 1.0, 2.2, r"W$_Q$", attention_color,
             subtext=f"C->d$_k$\n({n_embd}->{head_size})")
    draw_box(qkv_x, 5, 1.0, 2.2, r"W$_K$", attention_color,
             subtext=f"C->d$_k$\n({n_embd}->{head_size})")
    draw_box(qkv_x, 6.5, 1.0, 2.2, r"W$_V$", attention_color,
             subtext=f"C->d$_k$\n({n_embd}->{head_size})")
    draw_arrow(attn_left + 0.55, 5, qkv_x - 0.4, 3.5)
    draw_arrow(attn_left + 0.55, 5, qkv_x - 0.4, 5)
    draw_arrow(attn_left + 0.55, 5, qkv_x - 0.4, 6.5)
    
    qk_x = 9.2
    qk_w = 1.0
    qk_h = 2.8
    ax.add_patch(plt.Rectangle((qk_x - qk_w/2, 3.6), qk_w, qk_h, facecolor=attention_color,
                                edgecolor='#2C3E50', linewidth=2, zorder=2))
    ax.text(qk_x, 4.95, r"QK$^\top$", ha='center', va='center', fontsize=8, fontweight='bold', zorder=3, rotation=0)
    ax.text(qk_x, 4.65, r"/ $\sqrt{d_k}$", ha='center', va='center', fontsize=7, fontweight='bold', zorder=3, rotation=0)
    ax.text(qk_x, 4.25, f"(B,T,T)=", ha='center', va='center', fontsize=6, color='#555', rotation=0)
    ax.text(qk_x, 4.0, f"({batch_size},{block_size},", ha='center', va='center', fontsize=6, color='#555', rotation=0)
    ax.text(qk_x, 3.8, f"{block_size})", ha='center', va='center', fontsize=6, color='#555', rotation=0)
    draw_arrow(qkv_x + 0.4, 3.5, qk_x - 0.4, 3.95)
    draw_arrow(qkv_x + 0.4, 5, qk_x - 0.4, 4.3)
    
    mask_x = 10.6
    draw_box(mask_x, 4.75, 0.7, 2.4, "Mask +\nSoftmax", attention_color, fontsize=8)
    draw_arrow(qk_x + 0.45, 4.75, mask_x - 0.35, 4.75)
    
    av_x = 11.9
    draw_box(av_x, 5.5, 0.7, 1.8, "Attn × V", attention_color, fontsize=8)
    draw_arrow(mask_x + 0.3, 4.75, av_x - 0.3, 5.25)
    draw_arrow(qkv_x + 0.4, 6.5, av_x - 0.3, 5.75)
    
    if num_heads > 1:
        concat_x = 12.5
        draw_box(concat_x, 5.5, 0.7, 2.2, f"Concat\n{num_heads} heads", attention_color, fontsize=7.5)
        draw_arrow(av_x + 0.35, 5.5, concat_x - 0.35, 5.5)
        ax.text(concat_x, 6.7, rf"(B,T,{num_heads*head_size})=", fontsize=5.5, ha='center', color='#555')
        ax.text(concat_x, 6.5, rf"({batch_size},{block_size},{num_heads*head_size})", fontsize=5.5, ha='center', color='#555')
        attn_output_x = concat_x + 0.35
    else:
        attn_output_x = av_x + 0.35
    
    # 6. Projection (only if it exists in the model)
    if has_proj:
        x_proj = 13.5
        draw_box(x_proj, 5, 1.0, 3.2, "Projection", linear_color,
                 subtext=f"Linear({num_heads * head_size},\n{n_embd})")
        draw_arrow(attn_output_x, 5, x_proj - 0.4, 5)
        plus1_input_x = x_proj + 0.45
        x_plus1 = 14.8
    else:
        # Skip projection, go directly to residual - make arrow short
        plus1_input_x = attn_output_x
        x_plus1 = 12.9  # Move plus much closer since no projection
    
    # 7. Add & residual (from Add embeddings)
    draw_box(x_plus1, 5, 0.7, 2.2, "+", '#FFF', fontsize=14)
    draw_arrow(plus1_input_x, 5, x_plus1 - 0.35, 5)
    residual_arrow(x_add, 4.0, x_plus1, 4.0)
    ax.text((x_add + x_plus1) / 2, 3.6, "residual", fontsize=7, color='#27AE60', rotation=0, ha='center')
    
    # 8. LayerNorm (Pre-LN before FFN) - only if it exists
    if has_ln2:
        x_ln2 = 14.2
        draw_box(x_ln2, 5, 0.8, 2.2, "LayerNorm", norm_color, fontsize=9,
                 subtext=f"(B, T, {n_embd})")
        draw_arrow(x_plus1 + 0.35, 5, x_ln2 - 0.4, 5)
        ff_input_x = x_ln2 + 0.4
        x_ff = 15.5
    else:
        ff_input_x = x_plus1 + 0.35
        x_ff = 14.2  # Move feedforward closer

    # Adjust remaining positions to be more compact
    x_plus2 = 15.8
    x_lm = 17.0
    x_sm = 18.2
    x_out = 19.3
    
    # 9. Feed Forward
    draw_box(x_ff, 5, 1.4, 3.8, "Feed\nForward", linear_color,
             subtext=f"Linear({n_embd},{ffwd_hidden_dim})\n->ReLU->\nLinear({ffwd_hidden_dim},{n_embd})")
    draw_arrow(ff_input_x, 5, x_ff - 0.6, 5)
    
    # 10. Add & residual (from first +)
    draw_box(x_plus2, 5, 0.7, 2.2, "+", '#FFF', fontsize=14)
    draw_arrow(x_ff + 0.6, 5, x_plus2 - 0.35, 5)
    residual_arrow(x_plus1, 4.0, x_plus2, 4.0)
    ax.text((x_plus1 + x_plus2) / 2, 3.6, "residual", fontsize=7, color='#27AE60', rotation=0, ha='center')
    
    # 11. LM Head
    draw_box(x_lm, 5, 1.0, 3.0, "LM Head", output_color,
             subtext=f"Linear({n_embd},\n{vocab_size})")
    draw_arrow(x_plus2 + 0.35, 5, x_lm - 0.45, 5)
    
    # 12. Softmax
    draw_box(x_sm, 5, 0.8, 2.2, "Softmax", output_color)
    draw_arrow(x_lm + 0.45, 5, x_sm - 0.4, 5)
    
    # 13. Output
    draw_box(x_out, 5, 1.0, 3.8, "Output\nProbabilities", output_color,
             subtext=f"(B, T, vocab)=\n({batch_size},{block_size},\n{vocab_size})")
    draw_arrow(x_sm + 0.4, 5, x_out - 0.4, 5)
    
    # Legend
    legend_x = 1
    for i, (color, label) in enumerate([
        (embed_color, "Embeddings"), (attention_color, "Attention"),
        (linear_color, "Linear/FF"), (output_color, "Output"), (norm_color, "LayerNorm"),
    ]):
        y = 0.5 + i * 0.4
        ax.add_patch(plt.Rectangle((legend_x - 0.12, y - 0.25), 0.25, 0.5,
                                    facecolor=color, edgecolor='#2C3E50', linewidth=1))
        ax.text(legend_x + 0.35, y, label, fontsize=7, ha='left', va='center')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()
        print(f"Architecture diagram saved to {save_path}")
    else:
        plt.show()

# -----------------------------
# Q/K Embedding Space Visualization
# -----------------------------
def plot_qk_embedding_space(model, itos, save_path: str = None):
    """
    Create a single scatter plot showing ALL Q and K transformed embeddings
    with both token AND position labels for every combination.
    Uses consistent format: Q{token}p{pos} and K{token}p{pos}
    
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
            labels.append(f"{token_str}p{p}")
    
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
    
    # Plot invisible scatter points (for axis scaling)
    all_x = np.concatenate([Q_2d[:, 0], K_2d[:, 0]])
    all_y = np.concatenate([Q_2d[:, 1], K_2d[:, 1]])
    ax.scatter(all_x, all_y, s=0, alpha=0)
    
    # Add text labels for ALL Q points (blue)
    for i in range(num_combinations):
        ax.text(Q_2d[i, 0], Q_2d[i, 1], labels[i], fontsize=6, ha='center', va='center', color='blue')
    
    # Add text labels for ALL K points (red)
    for i in range(num_combinations):
        ax.text(K_2d[i, 0], K_2d[i, 1], labels[i], fontsize=6, ha='center', va='center', color='red')
    
    ax.set_xlabel("Dimension 1" + (" (PCA)" if head_size != 2 else ""), fontsize=12)
    ax.set_ylabel("Dimension 2" + (" (PCA)" if head_size != 2 else ""), fontsize=12)
    ax.set_title(f"Q and K Embedding Space\n{num_combinations} Q (blue) + {num_combinations} K (red) = {2*num_combinations} total\n({vocab_size} tokens × {block_size} positions)", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.text(0.02, 0.98, "Blue = Query, Red = Key\nFormat: {token}p{position}", 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()
        print(f"Q/K embedding space plot saved to {save_path}")
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

    n_panels = vocab_size + 1  # +1 for argmax (winner) panel
    n_cols = min(4, n_panels)
    n_rows = (n_panels + n_cols - 1) // n_cols

    for seq_idx, seq in enumerate(sequences):
        if len(seq) < 2:
            continue
        seq = seq[:block_size]
        idx = torch.tensor([seq], dtype=torch.long, device=next(model.parameters()).device)
        T = idx.shape[1]
        x_np, v_before, v_after = _get_v_before_after_for_sequence(model, idx)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows), sharex=True, sharey=True)
        axes = np.atleast_2d(axes).flatten()

        # Panel 0: discrete map of which token has largest output at each point; annotate regions; show sequence points
        ax0 = axes[0]
        ax0.set_xlim(x_min, x_max)
        ax0.set_ylim(y_min, y_max)
        ax0.set_aspect('equal')
        ax0.set_title("Argmax: predicted token", fontsize=11, fontweight='bold')
        Z_argmax = argmax_token.reshape(grid_resolution, grid_resolution)
        from matplotlib.colors import ListedColormap
        _cm = plt.cm.tab20 if vocab_size > 10 else plt.cm.tab10
        cmap_discrete = ListedColormap(_cm(np.linspace(0, 1, vocab_size)))
        im0 = ax0.pcolormesh(xx, yy, Z_argmax, cmap=cmap_discrete, vmin=0, vmax=vocab_size - 0.01, shading='auto')
        # Annotate each region with token label at centroid
        xx_flat = xx.ravel()
        yy_flat = yy.ravel()
        for d in range(vocab_size):
            mask = argmax_token == d
            if mask.any():
                cx, cy = xx_flat[mask].mean(), yy_flat[mask].mean()
                ax0.text(cx, cy, str(itos[d]), fontsize=11, fontweight='bold', ha='center', va='center',
                         color='black', zorder=3,
                         path_effects=[pe.withStroke(linewidth=2, foreground='white')])
        # Arrows and labels on argmax panel (no circles)
        arrow_color = '#E63946'
        for i in range(T):
            x0_pt, y0_pt = x_np[i, 0], x_np[i, 1]
            tail_x = x0_pt + arrow_scale * v_before[i, 0]
            tail_y = y0_pt + arrow_scale * v_before[i, 1]
            ax0.annotate('', xy=(x0_pt + arrow_scale * v_after[i, 0], y0_pt + arrow_scale * v_after[i, 1]),
                         xytext=(tail_x, tail_y),
                         arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.5), zorder=3)
            lbl = f"{itos[seq[i]]}p{i}"
            ax0.text(tail_x, tail_y, lbl, fontsize=5, fontweight='bold', ha='center', va='center',
                     color='black', zorder=5,
                     path_effects=[pe.withStroke(linewidth=1.2, foreground='white')])
        ax0.set_xlabel('dim 0')
        ax0.set_ylabel('dim 1')

        for d in range(vocab_size):
            ax = axes[d + 1]
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_aspect('equal')
            ax.set_title(f"Next token = {itos[d]}", fontsize=11, fontweight='bold')
            # Background: P(this token) heatmap
            Z = probs[:, d].reshape(grid_resolution, grid_resolution)
            ax.pcolormesh(xx, yy, Z, cmap='viridis', vmin=0, vmax=1, shading='auto')
            # Arrows from original V (tail) to V after; annotate token p position (no circles)
            arrow_color = '#E63946'
            for i in range(T):
                x0, y0 = x_np[i, 0], x_np[i, 1]
                tail_x = x0 + arrow_scale * v_before[i, 0]
                tail_y = y0 + arrow_scale * v_before[i, 1]
                head_x = x0 + arrow_scale * v_after[i, 0]
                head_y = y0 + arrow_scale * v_after[i, 1]
                ax.annotate('', xy=(head_x, head_y), xytext=(tail_x, tail_y),
                            arrowprops=dict(arrowstyle='->', color=arrow_color, lw=2), zorder=3)
                label = f"{itos[seq[i]]}p{i}"
                ax.text(tail_x, tail_y, label, fontsize=5, fontweight='bold', ha='center', va='center',
                        color='black', zorder=5,
                        path_effects=[pe.withStroke(linewidth=1.2, foreground='white')])
            ax.set_xlabel('dim 0')
            ax.set_ylabel('dim 1')

        for j in range(n_panels, len(axes)):
            axes[j].set_visible(False)
        seq_str = " ".join(str(itos[t]) for t in seq[:25])
        if len(seq) > 25:
            seq_str += "..."
        fig.suptitle(f"Demo sequence {seq_idx}: V before → after transform (arrows at each position)\nSequence: {seq_str}", fontsize=10, fontweight='bold', y=1.02)
        plt.tight_layout()
        if save_dir:
            path = os.path.join(save_dir, f"v_before_after_demo_{seq_idx}.png")
            plt.savefig(path, bbox_inches='tight', dpi=150, facecolor='white')
            plt.close()
            print(f"V before/after demo figure saved to {path}")
        else:
            plt.show()

    if save_dir and sequences:
        print(f"Saved {min(len(sequences), len([s for s in sequences if len(s) >= 2]))} demo sequence figures to {save_dir}")


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
            labels.append(f"{itos[t]}@{p}")
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

# -----------------------------
# Checkpoint saving/loading
