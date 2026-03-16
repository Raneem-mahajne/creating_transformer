"""Plotting: plot_probability_heatmap_with_values."""
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
def plot_probability_heatmap_with_values(
    model, itos, save_path=None, grid_resolution=80, extent_margin=0.5, step_label: int | None = None
):
    """
    Domain = embedding space. Background = P(next) at that embedding after second residual.
    Overlay = V values (W_V @ embedding) for reference (may fall outside embedding extent).
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

    # Grid extent: embedding space only (domain = embedding)
    x_min = emb_x_min - extent_margin
    x_max = emb_x_max + extent_margin
    y_min = emb_y_min - extent_margin
    y_max = emb_y_max + extent_margin
    x_c, y_c = (x_min + x_max) / 2, (y_min + y_max) / 2
    half = max(x_max - x_min, y_max - y_min) / 2
    x_min, x_max = x_c - half, x_c + half
    y_min, y_max = y_c - half, y_c + half

    # Create probability grid (output at each embedding point = after second residual)
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

    # Journal: 3 cols for A4; else 6 cols.
    n_cols = min(3 if _u._JOURNAL_MODE else 6, vocab_size)
    n_rows = (vocab_size + n_cols - 1) // n_cols
    if _u._JOURNAL_MODE:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.0, 8.0), sharex=True, sharey=True)
    else:
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
        ax.pcolormesh(xx, yy, Z, cmap='viridis', vmin=0, vmax=1, shading='auto', zorder=0)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')

        # Overlay annotations only (no marker dots/circles)
        _v_lbl_fs = 7 if _u._JOURNAL_MODE else 7
        for v_vec, label in zip(all_V, labels):
            if vocab_size * block_size <= 200:
                ax.text(v_vec[0], v_vec[1], label, fontsize=_v_lbl_fs, ha='center', va='center',
                       color='white', weight='bold', zorder=6)

        ax.set_title(f"P(next = {itos[token_idx]})", fontsize=10)
        if row == n_rows - 1:
            ax.set_xlabel("embedding dim 0", fontsize=9)
        if col == 0:
            ax.set_ylabel("embedding dim 1", fontsize=9)

    for token_idx in range(vocab_size, n_rows * n_cols):
        row = token_idx // n_cols
        col = token_idx % n_cols
        axes[row, col].axis('off')

    plt.tight_layout()
    if _u._JOURNAL_MODE:
        plt.subplots_adjust(hspace=0.12, wspace=0.08)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()
        print(f"Probability heatmap with V values saved to {save_path}")
    else:
        plt.show()

    model.train()
