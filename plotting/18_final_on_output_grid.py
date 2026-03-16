"""Plotting: plot_final_on_output_heatmap_grid, plot_per_token_frozen_output."""
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
from plotting.v_before_after_demo import _get_v_before_after_for_sequence


@torch.no_grad()
def plot_final_on_output_heatmap_grid(
    model, itos, sequence, save_path=None, grid_resolution=60, extent_margin=0.5
):
    """
    Domain = embedding space. Background = P(next token) at that embedding after the second
    residual (skip + FFN). Overlay = for the given sequence: embeddings and arrows to
    first-residual (embed + attn_out) positions.
    """
    import matplotlib.cm as cm
    model.eval()
    use_residual = getattr(model, "use_residual", True)
    if not use_residual:
        print("plot_final_on_output_heatmap_grid: model.use_residual is False; background uses ffwd(p) only (no skip).")
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
    # (token_str, position) -> color, same scheme as residuals plot (tab20, sorted by (token_str, pos))
    token_positions = [(itos[seq[i]], i) for i in range(T)]
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
        # Grid domain = embedding space only (background)
        x_min_g = flat[:, 0].min() - extent_margin
        x_max_g = flat[:, 0].max() + extent_margin
        y_min_g = flat[:, 1].min() - extent_margin
        y_max_g = flat[:, 1].max() + extent_margin
    x_c, y_c = (x_min_g + x_max_g) / 2, (y_min_g + y_max_g) / 2
    half = max(x_max_g - x_min_g, y_max_g - y_min_g) / 2
    x_min_g, x_max_g = x_c - half, x_c + half
    y_min_g, y_max_g = y_c - half, y_c + half
    # Axis limits: include first-residual positions so all arrows are visible
    all_xy = np.vstack([flat, final_x])
    x_min = min(x_min_g, all_xy[:, 0].min()) - extent_margin
    x_max = max(x_max_g, all_xy[:, 0].max()) + extent_margin
    y_min = min(y_min_g, all_xy[:, 1].min()) - extent_margin
    y_max = max(y_max_g, all_xy[:, 1].max()) + extent_margin
    x_c2, y_c2 = (x_min + x_max) / 2, (y_min + y_max) / 2
    half2 = max(x_max - x_min, y_max - y_min) / 2
    x_min, x_max = x_c2 - half2, x_c2 + half2
    y_min, y_max = y_c2 - half2, y_c2 + half2

    xs = np.linspace(x_min, x_max, grid_resolution)
    ys = np.linspace(y_min, y_max, grid_resolution)
    xx, yy = np.meshgrid(xs, ys)
    points = np.stack([xx.ravel(), yy.ravel()], axis=1)
    pts = torch.tensor(points, dtype=torch.float32, device=dev)
    with torch.no_grad():
        h = (pts + model.ffwd(pts)) if use_residual else model.ffwd(pts)
        logits = model.lm_head(h).cpu().numpy()
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)

    # Journal: 3 cols for A4; else 6 cols.
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

    _lbl_fs = 7 if _u._JOURNAL_MODE else 11
    _emb_fs = 14 if _u._JOURNAL_MODE else 16
    for d in range(vocab_size):
        row, col = d // n_cols, d % n_cols
        ax = axes[row, col]
        Z = probs[:, d].reshape(grid_resolution, grid_resolution)
        ax.pcolormesh(xx, yy, Z, cmap='viridis', vmin=0, vmax=1, shading='auto', zorder=0)
        # Limits include first-residual positions so all arrows are visible
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        for i in range(T):
            lbl = _token_pos_label(itos[seq[i]], i)
            ex, ey = x_np[i, 0], x_np[i, 1]
            fx, fy = final_x[i, 0], final_x[i, 1]
            color = token_pos_to_color[(itos[seq[i]], i)]
            ax.text(ex, ey, lbl, fontsize=_emb_fs, ha='center', va='center',
                    color=color, alpha=0.9, zorder=7,
                    path_effects=[pe.withStroke(linewidth=2, foreground='black')])
            dx, dy = fx - ex, fy - ey
            length = np.hypot(dx, dy)
            if length > 1e-6:
                hw = max(0.15, min(0.35, length * 0.12))
                ax.arrow(ex, ey, dx, dy,
                         head_width=hw, head_length=hw, fc=color, ec='black',
                         linewidth=0.8, length_includes_head=True, width=0.06,
                         alpha=0.9, zorder=6)
        ax.set_title(f"P(next = {itos[d]})", fontsize=10)
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
    fig.suptitle(f"Output after second residual  |  {seq_str}", fontsize=11, fontweight='bold', y=1.01)
    plt.tight_layout()
    if _u._JOURNAL_MODE:
        plt.subplots_adjust(hspace=0.15, wspace=0.08)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()
        print(f"Final-on-output heatmap grid saved to {save_path}")
    else:
        plt.show()
    model.train()


@torch.no_grad()
def plot_per_token_frozen_output(
    model, itos, sequence, save_dir=None, grid_resolution=60, extent_margin=1.0,
):
    """
    Per-token supplementary: for each token in the sequence, one figure (grid of
    output-token subplots). Background = softmax(lm_head(p + ffwd(p))) in absolute
    coordinates, zoomed to show the embedding and its first-residual destination.
    Arrow from embedding to embed + attn_out.
    """
    model.eval()
    use_residual = getattr(model, "use_residual", True)
    if not use_residual:
        print("plot_per_token_frozen_output: model.use_residual is False; background uses ffwd(p) only (no skip).")
    vocab_size = model.token_embedding.weight.shape[0]
    block_size = model.block_size
    n_embd = model.lm_head.in_features
    if n_embd != 2:
        print("plot_per_token_frozen_output: n_embd != 2. Skipping.")
        return
    if len(sequence) < 2:
        return
    seq = sequence[:block_size]
    T = len(seq)
    seq_str = " ".join(str(itos[t]) for t in seq)

    dev = next(model.parameters()).device
    idx = torch.tensor([seq], dtype=torch.long, device=dev)
    x_np, v_before, v_after = _get_v_before_after_for_sequence(model, idx)
    first_resid = x_np + v_after  # (T, 2)

    n_cols = min(3 if _u._JOURNAL_MODE else 6, vocab_size)
    n_rows = (vocab_size + n_cols - 1) // n_cols

    # Shared value-space extent (same axis range for all tokens). Origin (0,0) = no attn.
    v_absmax = np.abs(v_after).max() + extent_margin
    v_absmax = max(v_absmax, 1.5)
    v_min, v_max = -v_absmax, v_absmax
    vs = np.linspace(v_min, v_max, grid_resolution)
    vv0, vv1 = np.meshgrid(vs, vs)
    v_grid = np.stack([vv0.ravel(), vv1.ravel()], axis=1)

    for t_idx in range(T):
        token_str = itos[seq[t_idx]]
        e = x_np[t_idx]
        actual_v = v_after[t_idx]

        # Per-token background: g_e(v) = softmax(lm_head((e+v) + ffwd(e+v))). Different for each e
        # because the second residual acts on (e+v); ffwd is nonlinear so the map v -> output depends on e.
        with torch.no_grad():
            e_t = torch.tensor(e, dtype=torch.float32, device=dev)
            v_t = torch.tensor(v_grid, dtype=torch.float32, device=dev)
            p = e_t.unsqueeze(0) + v_t
            h = (p + model.ffwd(p)) if use_residual else model.ffwd(p)
            logits = model.lm_head(h).cpu().numpy()
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)

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

        lbl = _token_pos_label(token_str, t_idx)
        arrow_len = np.hypot(actual_v[0], actual_v[1])

        for d in range(vocab_size):
            row, col = d // n_cols, d % n_cols
            ax = axes[row, col]
            Z = probs[:, d].reshape(grid_resolution, grid_resolution)
            ax.pcolormesh(vv0, vv1, Z, cmap='viridis', vmin=0, vmax=1, shading='auto', zorder=0)
            ax.set_xlim(v_min, v_max)
            ax.set_ylim(v_min, v_max)
            ax.set_aspect('equal')
            # Embedding in value space is at v=0 (origin). Annotation there; arrow to actual_v.
            ax.text(0, 0, lbl, fontsize=8, ha='center', va='bottom',
                    color='white', fontweight='bold', zorder=7,
                    path_effects=[pe.withStroke(linewidth=2, foreground='black')])
            if arrow_len > 1e-6:
                hw = max(0.08, min(0.25, arrow_len * 0.12))
                ax.arrow(0, 0, actual_v[0], actual_v[1],
                         head_width=hw, head_length=hw, fc='red', ec='black',
                         linewidth=0.6, length_includes_head=True, width=0.04,
                         alpha=0.95, zorder=6)
            ax.set_title(f"P(next = {itos[d]})", fontsize=10)
            if row == n_rows - 1:
                ax.set_xlabel("value dim 0", fontsize=9)
            if col == 0:
                ax.set_ylabel("value dim 1", fontsize=9)

        for j in range(vocab_size, n_rows * n_cols):
            row, col = j // n_cols, j % n_cols
            axes[row, col].axis('off')

        tokens_with_highlight = []
        for i, t in enumerate(seq):
            s = str(itos[t])
            tokens_with_highlight.append(f"[{s}]" if i == t_idx else s)
        seq_highlight = " ".join(tokens_with_highlight)
        fig.suptitle(
            f"Output landscape for {lbl} (e=[{e[0]:.2f},{e[1]:.2f}])  |  {seq_highlight}",
            fontsize=10, fontweight='bold', y=1.01,
        )
        plt.tight_layout()
        if _u._JOURNAL_MODE:
            plt.subplots_adjust(hspace=0.15, wspace=0.08)
        if save_dir:
            path = os.path.join(save_dir, f"frozen_output_pos{t_idx}_{token_str}.png")
            plt.savefig(path, bbox_inches='tight', dpi=150, facecolor='white')
            plt.close()
        else:
            plt.show()

    model.train()
