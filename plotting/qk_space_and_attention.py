"""Plotting: plot_qk_space_and_attention_heatmap."""
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


@torch.no_grad()
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
