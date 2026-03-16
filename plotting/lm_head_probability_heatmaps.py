"""Plotting: plot_lm_head_probability_heatmaps."""
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
