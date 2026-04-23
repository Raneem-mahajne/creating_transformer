"""Plotting: plot_probability_heatmap."""
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
def plot_probability_heatmap(
    model, itos, save_path=None, grid_resolution=80, extent_margin=0.5, step_label: int | None = None
):
    """
    Plot probability heatmaps for each token WITHOUT token overlays.
    Domain = embedding space (axes are embedding coordinates). At each point,
    background = P(next token) after the second residual (skip + FFN), i.e.
    softmax(lm_head(embedding + ffwd(embedding))).
    """
    model.eval()
    vocab_size = model.token_embedding.weight.shape[0]
    block_size = model.position_embedding_table.weight.shape[0]
    n_embd = model.lm_head.in_features
    if n_embd != 2:
        print(f"plot_probability_heatmap: n_embd={n_embd}, need 2. Skipping.")
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

    # Create figure with one subplot per token. Journal: 3 cols for A4; else 6 cols.
    n_cols = min(3 if _u._JOURNAL_MODE else 6, vocab_size)
    n_rows = (vocab_size + n_cols - 1) // n_cols
    if _u._JOURNAL_MODE:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.0, 7.0), sharex=True, sharey=True)
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
        
        # Plot probability heatmap only (NO token overlays)
        Z = probs[:, token_idx].reshape(grid_resolution, grid_resolution)
        im = ax.pcolormesh(xx, yy, Z, cmap='viridis', vmin=0, vmax=1, shading='auto', zorder=0)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        # Add clear separation between title text and heatmap area.
        ax.set_title(f"P(next = {itos[token_idx]})", fontsize=10, pad=6, y=1.00)
        if row == n_rows - 1:
            ax.set_xlabel("embedding dim 0", fontsize=9)
        if col == 0:
            ax.set_ylabel("embedding dim 1", fontsize=9)

    # Hide unused subplots
    for token_idx in range(vocab_size, n_rows * n_cols):
        row = token_idx // n_cols
        col = token_idx % n_cols
        axes[row, col].axis('off')

    if _u._JOURNAL_MODE:
        plt.subplots_adjust(hspace=0.34, wspace=0.08, right=0.86)
    else:
        plt.subplots_adjust(hspace=0.36, wspace=0.12, right=0.88)

    # Place a shared colorbar just to the right of the rightmost subplot column.
    all_axes = axes.ravel().tolist()
    right_edge = max(ax.get_position().x1 for ax in all_axes)
    bottom_edge = min(ax.get_position().y0 for ax in all_axes)
    top_edge = max(ax.get_position().y1 for ax in all_axes)
    cbar_pad = 0.02
    cbar_width = 0.018 if _u._JOURNAL_MODE else 0.015
    cax = fig.add_axes([right_edge + cbar_pad, bottom_edge, cbar_width, top_edge - bottom_edge])
    fig.colorbar(im, cax=cax, label="Probability")
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()
        print(f"Probability heatmap (without tokens) saved to {save_path}")
    else:
        plt.show()
    
    model.train()
