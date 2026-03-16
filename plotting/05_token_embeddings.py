"""Plotting: plot_token_embeddings_heatmap, plot_bigram_logits_heatmap, plot_token_embeddings_pca_2d_with_hclust, plot_bigram_probability_heatmap_hclust, plot_bigram_probability_heatmap, plot_embeddings_pca, plot_embeddings_scatterplots_only."""
import os
import numpy as np
import torch
from torch.nn import functional as F
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import matplotlib.patheffects as pe
import matplotlib.text as _mtext
import matplotlib.colors as mcolors
import seaborn as sns

from data import get_batch_from_sequences

import plotting._utils as _u
from plotting._utils import (
    _constrain_figsize, _update_font_scale_for_figure,
    set_journal_mode, clear_journal_mode,
    _label_panels, annotate_sequence, sparse_ticks,
    _SUBSCRIPT_DIGITS, _pos_subscript, _token_pos_label, _pos_only_label,
    collect_epoch_stats, get_attention_snapshot_from_X, get_multihead_snapshot_from_X,
    estimate_loss, estimate_rule_error,
)


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


@torch.no_grad()
def plot_embeddings_pca(model, itos, save_path=None):
    """
    Plot embeddings: token embeddings (heatmap, clustered, PCA), position embeddings, and token+position combinations.
    """
    import matplotlib.colors as mcolors
    
    # Dynamic font sizing based on number of items
    def get_fontsize(num_items):
        if num_items <= 12:
            return 8 if _u._JOURNAL_MODE else 22
        elif num_items <= 20:
            return 7 if _u._JOURNAL_MODE else 18
        elif num_items <= 40:
            return 6 if _u._JOURNAL_MODE else 14
        elif num_items <= 80:
            return 5 if _u._JOURNAL_MODE else 12
        elif num_items <= 150:
            return 4 if _u._JOURNAL_MODE else 10
        else:
            return 3 if _u._JOURNAL_MODE else 8
    
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
    
    # Create figure: journal=strict 4×2 grid (rows 1–3 equal height, row 4 = g); standard=2 rows × 4 cols
    from matplotlib.gridspec import GridSpec

    def _square_axis_limits(x, y, margin=0.15):
        """Return (xlo, xhi, ylo, yhi) with equal span so aspect='equal' fills the cell."""
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
        sx = max(xmax - xmin, 0.5)
        sy = max(ymax - ymin, 0.5)
        span = max(sx, sy) * (1 + margin)
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        h = span / 2
        return (cx - h, cx + h, cy - h, cy + h)

    def _set_limits_to_match_box_aspect(ax, x, y, margin=0.15):
        """Set xlim/ylim so that with aspect='equal' the plot fills the subplot (same width as heatmaps)."""
        pos = ax.get_position()
        box_aspect = pos.width / pos.height
        xmin, xmax = x.min(), x.max()
        ymin, ymax = y.min(), y.max()
        sx = max(xmax - xmin, 0.5)
        sy = max(ymax - ymin, 0.5)
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        pad = 1 + margin
        h_y = max(sy / 2 * pad, (sx / 2 * pad) / box_aspect)
        h_x = h_y * box_aspect
        if h_x < sx / 2 * pad:
            h_x = sx / 2 * pad
            h_y = h_x / box_aspect
        ax.set_xlim(cx - h_x, cx + h_x)
        ax.set_ylim(cy - h_y, cy + h_y)

    if _u._JOURNAL_MODE:
        fig = plt.figure(figsize=(11.0, 13.0), constrained_layout=True)
        fig.set_constrained_layout_pads(w_pad=2/72, h_pad=2/72, wspace=0.02, hspace=0.04)
        gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 1, 1.8])
        pos_display_labels = [f"P{i}" for i in range(block_size)]
    else:
        pos_display_labels = None
        fig = plt.figure(figsize=(19, 9))
        gs = GridSpec(2, 4, figure=fig, width_ratios=[1, 1, 1, 1], hspace=0.3, wspace=0.28)
    pos_labels = [f"P{i}" for i in range(block_size)] if _u._JOURNAL_MODE else [_pos_only_label(i) for i in range(block_size)]
    ax1 = fig.add_subplot(gs[0, 0])
    x_labels = list(range(embeddings.shape[1]))
    _cbar_kw_a = {'shrink': 0.6, 'aspect': 15, 'pad': 0.002} if _u._JOURNAL_MODE else {'orientation': 'vertical', 'pad': 0.03, 'aspect': 20, 'shrink': 0.8}
    _cbar_kw_b = {'shrink': 0.6, 'aspect': 15, 'pad': 0.01} if _u._JOURNAL_MODE else {'orientation': 'vertical', 'pad': 0.03, 'aspect': 20, 'shrink': 0.8}
    sns.heatmap(
        embeddings,
        yticklabels=y_labels,
        xticklabels=x_labels,
        cmap="RdBu_r",
        center=0,
        ax=ax1,
        cbar_kws=_cbar_kw_a,
    )
    if _u._JOURNAL_MODE:
        ax1.set_aspect('auto')
    _title_fs = 8 if _u._JOURNAL_MODE else 11
    ax1.set_title(f"Token Embeddings (vocab×embd={vocab_size}×{n_embd})" if not _u._JOURNAL_MODE else f"Token Embeddings (vocab x embd={vocab_size}x{n_embd})", fontsize=_title_fs)
    ax1.set_xlabel("Embedding dim")
    ax1.set_ylabel("Token")
    if _u._JOURNAL_MODE:
        ax1.tick_params(axis='both', labelsize=6)
    
    # Token embeddings scatter: journal row 1 col 0; standard row 1 col 0
    ax3 = fig.add_subplot(gs[1, 0])
    if n_embd > 2:
        # Do PCA for dimensions > 2
        _, _, Vt = np.linalg.svd(X_emb, full_matrices=False)
        X2 = X_emb @ Vt[:2].T
        if _u._JOURNAL_MODE:
            xlo, xhi, ylo, yhi = _square_axis_limits(X2[:, 0], X2[:, 1])
            ax3.set_xlim(xlo, xhi)
            ax3.set_ylim(ylo, yhi)
        else:
            margin = 0.15 * max(X2[:, 0].max() - X2[:, 0].min(), X2[:, 1].max() - X2[:, 1].min())
            ax3.set_xlim(X2[:, 0].min() - margin, X2[:, 0].max() + margin)
            ax3.set_ylim(X2[:, 1].min() - margin, X2[:, 1].max() + margin)
        ax3.set_title(f"Token Embeddings PCA 2D (vocab={vocab_size})", fontsize=_title_fs, fontweight='bold')
        ax3.set_xlabel("PC1")
        ax3.set_ylabel("PC2")
        ax3.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=0.5)
        ax3.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=0.5)
        ax3.grid(True, alpha=0.3)
        # In journal mode, keep cell sizes consistent with heatmaps; avoid forcing equal aspect.
        if _u._JOURNAL_MODE:
            ax3.set_aspect('auto')
        else:
            ax3.set_aspect('equal', adjustable='box')
        token_fontsize = get_fontsize(vocab_size)
        if vocab_size <= 80:
            for i in range(vocab_size):
                ax3.text(X2[i, 0], X2[i, 1], itos[i], fontsize=token_fontsize, fontweight='bold',
                        ha='center', va='center', color=token_colors[i])
        if _u._JOURNAL_MODE:
            ax3.tick_params(axis='both', labelsize=6)
    elif n_embd == 2:
        # For 2D embeddings, show raw data
        if _u._JOURNAL_MODE:
            xlo, xhi, ylo, yhi = _square_axis_limits(X_emb[:, 0], X_emb[:, 1])
            ax3.set_xlim(xlo, xhi)
            ax3.set_ylim(ylo, yhi)
        else:
            margin = 0.15 * max(X_emb[:, 0].max() - X_emb[:, 0].min(), X_emb[:, 1].max() - X_emb[:, 1].min())
            ax3.set_xlim(X_emb[:, 0].min() - margin, X_emb[:, 0].max() + margin)
            ax3.set_ylim(X_emb[:, 1].min() - margin, X_emb[:, 1].max() + margin)
        ax3.set_title(f"Token Embeddings (vocab={vocab_size})", fontsize=_title_fs, fontweight='bold')
        ax3.set_xlabel("Dim 0")
        ax3.set_ylabel("Dim 1")
        ax3.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=0.5)
        ax3.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=0.5)
        ax3.grid(True, alpha=0.3)
        if _u._JOURNAL_MODE:
            ax3.set_aspect('auto')
        else:
            ax3.set_aspect('equal', adjustable='box')
        token_fontsize = get_fontsize(vocab_size)
        if vocab_size <= 80:
            for i in range(vocab_size):
                ax3.text(X_emb[i, 0], X_emb[i, 1], itos[i], fontsize=token_fontsize, fontweight='bold',
                        ha='center', va='center', color=token_colors[i])
        if _u._JOURNAL_MODE:
            ax3.tick_params(axis='both', labelsize=6)
    else:
        # For 1D embeddings, just plot the single dimension
        X1 = X_emb[:, 0]
        margin = 0.15 * (X1.max() - X1.min())
        ax3.set_xlim(X1.min() - margin, X1.max() + margin)
        ax3.set_ylim(-0.5, 0.5)
        ax3.set_title(f"Token Embeddings 1D (vocab={vocab_size})", fontsize=_title_fs, fontweight='bold')
        ax3.set_xlabel("Embedding value")
        ax3.set_ylabel("")
        ax3.grid(True, alpha=0.3)
        token_fontsize = get_fontsize(vocab_size)
        if vocab_size <= 80:
            for i in range(vocab_size):
                ax3.text(X1[i], 0, itos[i], fontsize=token_fontsize, fontweight='bold',
                        ha='center', va='center', color=token_colors[i])
        ax3.set_yticks([])
        if _u._JOURNAL_MODE:
            ax3.tick_params(axis='both', labelsize=6)
    
    # Row 0, Col 1: Position embeddings raw
    ax4 = fig.add_subplot(gs[0, 1])
    x_labels = list(range(pos_emb_all.shape[1]))
    sns.heatmap(
        pos_emb_all,
        yticklabels=pos_labels,
        xticklabels=x_labels,
        cmap="RdBu_r",
        center=0,
        ax=ax4,
        cbar_kws=_cbar_kw_b,
    )
    if _u._JOURNAL_MODE:
        ax4.set_aspect('auto')
    ax4.set_title(f"Position Embeddings (block_size×embd={block_size}×{n_embd})" if not _u._JOURNAL_MODE else f"Position Embeddings (block_size x embd={block_size}x{n_embd})", fontsize=_title_fs)
    ax4.set_xlabel("Embedding dim")
    ax4.set_ylabel("Position")
    if _u._JOURNAL_MODE:
        ax4.tick_params(axis='both', labelsize=6)
    
    # Row 1, Col 1: Position embeddings scatter
    ax6 = fig.add_subplot(gs[1, 1])
    if n_embd > 2:
        # Do PCA for dimensions > 2
        _, _, Vt_pos = np.linalg.svd(X_pos, full_matrices=False)
        X2_pos = X_pos @ Vt_pos[:2].T
        if _u._JOURNAL_MODE:
            xlo, xhi, ylo, yhi = _square_axis_limits(X2_pos[:, 0], X2_pos[:, 1])
            ax6.set_xlim(xlo, xhi)
            ax6.set_ylim(ylo, yhi)
        else:
            margin = 0.15 * max(X2_pos[:, 0].max() - X2_pos[:, 0].min(), X2_pos[:, 1].max() - X2_pos[:, 1].min())
            ax6.set_xlim(X2_pos[:, 0].min() - margin, X2_pos[:, 0].max() + margin)
            ax6.set_ylim(X2_pos[:, 1].min() - margin, X2_pos[:, 1].max() + margin)
        ax6.set_title(f"Position Embeddings PCA 2D (block_size={block_size})", fontsize=_title_fs, fontweight='bold')
        ax6.set_xlabel("PC1")
        ax6.set_ylabel("PC2")
        ax6.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=0.5)
        ax6.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=0.5)
        ax6.grid(True, alpha=0.3)
        if _u._JOURNAL_MODE:
            ax6.set_aspect('auto')
        else:
            ax6.set_aspect('equal', adjustable='box')
        pos_fontsize = get_fontsize(block_size)
        if block_size <= 80:
            for i in range(block_size):
                ax6.text(X2_pos[i, 0], X2_pos[i, 1], pos_labels[i], fontsize=pos_fontsize, fontweight='bold',
                        ha='center', va='center', color=pos_colors[i])
        if _u._JOURNAL_MODE:
            ax6.tick_params(axis='both', labelsize=6)
    elif n_embd == 2:
        # For 2D embeddings, show raw data
        if _u._JOURNAL_MODE:
            xlo, xhi, ylo, yhi = _square_axis_limits(X_pos[:, 0], X_pos[:, 1])
            ax6.set_xlim(xlo, xhi)
            ax6.set_ylim(ylo, yhi)
        else:
            margin = 0.15 * max(X_pos[:, 0].max() - X_pos[:, 0].min(), X_pos[:, 1].max() - X_pos[:, 1].min())
            ax6.set_xlim(X_pos[:, 0].min() - margin, X_pos[:, 0].max() + margin)
            ax6.set_ylim(X_pos[:, 1].min() - margin, X_pos[:, 1].max() + margin)
        ax6.set_title(f"Position Embeddings (block_size={block_size})", fontsize=_title_fs, fontweight='bold')
        ax6.set_xlabel("Dim 0")
        ax6.set_ylabel("Dim 1")
        ax6.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=0.5)
        ax6.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=0.5)
        ax6.grid(True, alpha=0.3)
        if _u._JOURNAL_MODE:
            ax6.set_aspect('auto')
        else:
            ax6.set_aspect('equal', adjustable='box')
        pos_fontsize = get_fontsize(block_size)
        if block_size <= 80:
            for i in range(block_size):
                ax6.text(X_pos[i, 0], X_pos[i, 1], pos_labels[i], fontsize=pos_fontsize, fontweight='bold',
                        ha='center', va='center', color=pos_colors[i])
        if _u._JOURNAL_MODE:
            ax6.tick_params(axis='both', labelsize=6)
    else:
        # For 1D embeddings, just plot the single dimension
        X1_pos = X_pos[:, 0]
        margin = 0.15 * (X1_pos.max() - X1_pos.min())
        ax6.set_xlim(X1_pos.min() - margin, X1_pos.max() + margin)
        ax6.set_ylim(-0.5, block_size - 0.5)
        ax6.set_title(f"Position Embeddings 1D (block_size={block_size})", fontsize=_title_fs, fontweight='bold')
        ax6.set_xlabel("Embedding value")
        ax6.set_ylabel("Position index")
        ax6.grid(True, alpha=0.3)
        pos_fontsize = get_fontsize(block_size)
        if block_size <= 80:
            for i in range(block_size):
                ax6.text(X1_pos[i], i, pos_labels[i], fontsize=pos_fontsize, fontweight='bold',
                        ha='center', va='center', color=pos_colors[i])
        if _u._JOURNAL_MODE:
            ax6.tick_params(axis='both', labelsize=6)
    
    # Create all token-position combinations (ALL tokens including special characters)
    max_token_idx = vocab_size
    num_combinations = max_token_idx * block_size
    all_combinations = np.zeros((num_combinations, n_embd))
    for token_idx in range(max_token_idx):
        for pos_idx in range(block_size):
            idx = token_idx * block_size + pos_idx
            all_combinations[idx] = embeddings[token_idx] + pos_emb_all[pos_idx]
    
    token_labels = [itos[i] for i in range(max_token_idx)]

    if not _u._JOURNAL_MODE:
        # Token+Position Dim 0 heatmap (standard layout only)
        ax10 = fig.add_subplot(gs[0, 2])
        dim0_heatmap = np.zeros((max_token_idx, block_size))
        for token_idx in range(max_token_idx):
            for pos_idx in range(block_size):
                idx = token_idx * block_size + pos_idx
                dim0_heatmap[token_idx, pos_idx] = all_combinations[idx, 0]
        sns.heatmap(dim0_heatmap, yticklabels=token_labels, xticklabels=pos_labels, cmap="RdBu_r", center=0, ax=ax10)
        ax10.set_title("Token+Position: Dim 0 (tokens×positions)", fontsize=_title_fs)
        ax10.set_xlabel("Position")
        ax10.set_ylabel("Token")
        ax10.set_xticklabels(ax10.get_xticklabels(), rotation=0)

        # Token+Position Dim 1 heatmap (standard layout only)
        if n_embd >= 2:
            ax10b = fig.add_subplot(gs[0, 3])
            dim1_heatmap = np.zeros((max_token_idx, block_size))
            for token_idx in range(max_token_idx):
                for pos_idx in range(block_size):
                    idx = token_idx * block_size + pos_idx
                    dim1_heatmap[token_idx, pos_idx] = all_combinations[idx, 1]
            sns.heatmap(dim1_heatmap, yticklabels=token_labels, xticklabels=pos_labels, cmap="RdBu_r", center=0, ax=ax10b)
            ax10b.set_title("Token+Position: Dim 1 (tokens×positions)", fontsize=_title_fs)
            ax10b.set_xlabel("Position")
            ax10b.set_ylabel("Token")
            ax10b.set_xticklabels(ax10b.get_xticklabels(), rotation=0)

    # Token+Position scatter: journal rows 2–3 full width (2×2); standard row 1 cols 2–3.
    ax12 = fig.add_subplot(gs[2, :] if _u._JOURNAL_MODE else gs[1, 2:4])
    # Dynamic font size for token+position (usually more items)
    combo_fontsize = get_fontsize(num_combinations)
    if _u._JOURNAL_MODE:
        # Make token+position annotations more legible in the large bottom panel.
        combo_fontsize = max(combo_fontsize + 6, 11)
    
    if n_embd > 2:
        # Do PCA for dimensions > 2
        X_comb = all_combinations.astype(np.float64)
        X_comb = X_comb - X_comb.mean(axis=0, keepdims=True)
        _, _, Vt_comb = np.linalg.svd(X_comb, full_matrices=False)
        X2_comb = X_comb @ Vt_comb[:2].T

        if _u._JOURNAL_MODE:
            xlo, xhi, ylo, yhi = _square_axis_limits(X2_comb[:, 0], X2_comb[:, 1])
            ax12.set_xlim(xlo, xhi)
            ax12.set_ylim(ylo, yhi)
        else:
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
        
        ax12.set_title("Token+Position: PCA (all tokens)", fontsize=_title_fs, fontweight='bold')
        ax12.set_xlabel("PC1")
        ax12.set_ylabel("PC2")
        ax12.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=0.5)
        ax12.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=0.5)
        ax12.grid(True, alpha=0.3)
        if _u._JOURNAL_MODE:
            ax12.set_aspect('auto')
        else:
            ax12.set_aspect('equal', adjustable='box')
        if _u._JOURNAL_MODE:
            ax12.tick_params(axis='both', labelsize=6)
    elif n_embd == 2:
        # For 2D embeddings, show raw data
        X_comb = all_combinations.astype(np.float64)

        if _u._JOURNAL_MODE:
            xlo, xhi, ylo, yhi = _square_axis_limits(X_comb[:, 0], X_comb[:, 1])
            ax12.set_xlim(xlo, xhi)
            ax12.set_ylim(ylo, yhi)
        else:
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
        
        ax12.set_title("Token+Position: Raw (all tokens)", fontsize=_title_fs, fontweight='bold')
        ax12.set_xlabel("Dim 0")
        ax12.set_ylabel("Dim 1")
        ax12.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=0.5)
        ax12.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=0.5)
        ax12.grid(True, alpha=0.3)
        if _u._JOURNAL_MODE:
            ax12.set_aspect('auto')
        else:
            ax12.set_aspect('equal', adjustable='box')
        if _u._JOURNAL_MODE:
            ax12.tick_params(axis='both', labelsize=6)
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
        
        ax12.set_title("Token+Position: 1D (all tokens)", fontsize=_title_fs, fontweight='bold')
        ax12.set_xlabel("Embedding value")
        ax12.set_ylabel("")
        ax12.grid(True, alpha=0.3)
        ax12.set_yticks([])
        if _u._JOURNAL_MODE:
            ax12.tick_params(axis='both', labelsize=6)
    
    if _u._JOURNAL_MODE:
        _emb_axes = [ax1, ax4, ax3, ax6, ax12]
    else:
        _emb_axes = [ax1, ax4, ax3, ax6, ax10]
        if n_embd >= 2:
            _emb_axes.append(ax10b)
        _emb_axes.append(ax12)
    # Grid layout is handled purely via GridSpec; no manual axis repositioning.
    _label_panels(_emb_axes, y=1.12)

    if not _u._JOURNAL_MODE:
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
    # Add origin lines for 2D plots
    if n_embd > 2 or n_embd == 2:
        ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4, zorder=0.5)
        ax1.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4, zorder=0.5)
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
    # Add origin lines for 2D plots
    if n_embd > 2 or n_embd == 2:
        ax3.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4, zorder=0.5)
        ax3.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4, zorder=0.5)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()
    model.train()
