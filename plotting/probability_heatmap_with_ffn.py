"""Plotting: plot_probability_heatmap_with_ffn_positions."""
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
def plot_probability_heatmap_with_ffn_positions(
    model, itos, save_path=None, grid_resolution=80, extent_margin=0.5,
):
    """
    Domain = embedding space. Background = P(next) at that embedding after the second
    residual (skip + FFN). Overlay = final positions (emb + ffwd(emb)) for each
    token+position embedding.
    """
    model.eval()
    vocab_size = model.token_embedding.weight.shape[0]
    block_size = model.position_embedding_table.weight.shape[0]
    n_embd = model.lm_head.in_features
    if n_embd != 2:
        print(f"plot_probability_heatmap_with_ffn_positions: n_embd={n_embd}, need 2. Skipping.")
        return

    with torch.no_grad():
        token_emb = model.token_embedding.weight.detach().cpu().numpy()
        pos_emb = model.position_embedding_table.weight.detach().cpu().numpy()
        combined = token_emb[:, None, :] + pos_emb[None, :, :]
        flat = combined.reshape(-1, 2)

    # All embeddings and their FFN+skip destinations
    all_emb, labels = [], []
    for ti in range(vocab_size):
        for pi in range(block_size):
            all_emb.append(token_emb[ti] + pos_emb[pi])
            labels.append(_token_pos_label(itos[ti], pi))
    all_emb = np.array(all_emb)

    dev = next(model.parameters()).device
    with torch.no_grad():
        emb_t = torch.tensor(all_emb, dtype=torch.float32, device=dev)
        final_pos = (emb_t + model.ffwd(emb_t)).cpu().numpy()

    # Domain = embedding space only
    x_min, x_max = flat[:, 0].min() - extent_margin, flat[:, 0].max() + extent_margin
    y_min, y_max = flat[:, 1].min() - extent_margin, flat[:, 1].max() + extent_margin
    x_c, y_c = (x_min + x_max) / 2, (y_min + y_max) / 2
    half = max(x_max - x_min, y_max - y_min) / 2
    x_min, x_max = x_c - half, x_c + half
    y_min, y_max = y_c - half, y_c + half

    # Background = output at each embedding after second residual
    xs = np.linspace(x_min, x_max, grid_resolution)
    ys = np.linspace(y_min, y_max, grid_resolution)
    xx, yy = np.meshgrid(xs, ys)
    points = np.stack([xx.ravel(), yy.ravel()], axis=1)
    with torch.no_grad():
        pts = torch.tensor(points, dtype=torch.float32, device=dev)
        h = pts + model.ffwd(pts)
        logits = model.lm_head(h).cpu().numpy()
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)

    n_cols = min(3 if _u._JOURNAL_MODE else 6, vocab_size)
    n_rows = (vocab_size + n_cols - 1) // n_cols
    if _u._JOURNAL_MODE:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.0, 8.0), sharex=True, sharey=True)
    else:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows), sharex=True, sharey=True)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    _lbl_fs = 9 if _u._JOURNAL_MODE else 7
    for token_idx in range(vocab_size):
        row, col = token_idx // n_cols, token_idx % n_cols
        ax = axes[row, col]
        Z = probs[:, token_idx].reshape(grid_resolution, grid_resolution)
        ax.pcolormesh(xx, yy, Z, cmap='viridis', vmin=0, vmax=1, shading='auto', zorder=0)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        if vocab_size * block_size <= 200:
            for fp, label in zip(final_pos, labels):
                ax.text(
                    fp[0], fp[1], label, fontsize=_lbl_fs,
                    ha='center', va='center', color='white',
                    weight='bold', zorder=6,
                )
        ax.set_title(f"P(next = {itos[token_idx]})", fontsize=10)
        if row == n_rows - 1:
            ax.set_xlabel("embedding dim 0", fontsize=9)
        if col == 0:
            ax.set_ylabel("embedding dim 1", fontsize=9)

    for token_idx in range(vocab_size, n_rows * n_cols):
        row, col = token_idx // n_cols, token_idx % n_cols
        axes[row, col].axis('off')

    plt.tight_layout()
    if _u._JOURNAL_MODE:
        plt.subplots_adjust(hspace=0.12, wspace=0.08)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()
        print(f"Probability heatmap with FFN positions saved to {save_path}")
    else:
        plt.show()
    model.train()
