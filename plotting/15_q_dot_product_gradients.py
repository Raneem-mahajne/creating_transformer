"""Plotting: plot_q_dot_product_gradients."""
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
from matplotlib.cm import ScalarMappable
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
def plot_q_dot_product_gradients(model, X_list, itos, save_path=None, num_sequences=1):
    """
    Plot dot product gradients for each query in a 2x4 grid.
    Each subplot shows one Q point with its dot product gradient background.
    
    Args:
        model: The model
        X_list: List of input sequences, or single sequence (will be converted to list)
        itos: Index to string mapping
        save_path: Path to save figure
        num_sequences: Number of sequences (should be 1 for this plot)
    """
    model.eval()
    
    # Handle single sequence input
    if not isinstance(X_list, list):
        X_list = [X_list]
    
    # Use first sequence
    X = X_list[0]
    tokens = [itos[i.item()] for i in X[0]]
    seq_str = " ".join(tokens)
    B, T = X.shape
    
    # Get Q and K from the sequence
    token_emb = model.token_embedding(X)
    pos = torch.arange(T, device=X.device) % model.block_size
    pos_emb = model.position_embedding_table(pos)
    x = token_emb + pos_emb
    
    # Get Q and K from first head (or average across heads)
    head = model.sa_heads.heads[0]
    Q = head.query(x)[0].cpu().numpy()  # (T, hs)
    K = head.key(x)[0].cpu().numpy()    # (T, hs)
    
    # Helper function for PCA
    def pca_2d(data):
        if data.shape[1] <= 2:
            if data.shape[1] == 2:
                return data
            elif data.shape[1] == 1:
                result = np.zeros((data.shape[0], 2))
                result[:, 0] = data[:, 0]
                return result
            else:
                return data[:, :2]
        data_centered = data - data.mean(axis=0, keepdims=True)
        U, s, Vt = np.linalg.svd(data_centered, full_matrices=False)
        return data_centered @ Vt[:2].T
    
    # Apply PCA to Q and K
    combined_QK = np.vstack([Q, K])
    if combined_QK.shape[1] > 2:
        combined_centered = combined_QK - combined_QK.mean(axis=0, keepdims=True)
        U, s, Vt = np.linalg.svd(combined_centered, full_matrices=False)
        pca_transform = Vt[:2].T
        Q_centered = Q - combined_QK.mean(axis=0, keepdims=True)
        K_centered = K - combined_QK.mean(axis=0, keepdims=True)
        Q_2d = Q_centered @ pca_transform
        K_2d = K_centered @ pca_transform
    else:
        Q_2d = pca_2d(Q)
        K_2d = pca_2d(K)
    
    # Determine extent with margin
    all_points = np.vstack([Q_2d, K_2d])
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    margin_x = max(0.5, (x_max - x_min) * 0.15)
    margin_y = max(0.5, (y_max - y_min) * 0.15)
    x_min -= margin_x
    x_max += margin_x
    y_min -= margin_y
    y_max += margin_y
    
    # Compute masked QK^T and Attention for the bottom row
    head_size = Q.shape[1]
    QK_T = (Q @ K.T) / np.sqrt(head_size)
    QK_T_torch = torch.from_numpy(QK_T).float()
    tril_mask = torch.tril(torch.ones(T, T))
    Masked_QK_T = QK_T_torch.masked_fill(tril_mask == 0, float("-inf")).numpy()

    wei_all = []
    with torch.no_grad():
        head_obj = model.sa_heads.heads[0]
        _, wei = head_obj(token_emb + pos_emb)
        wei_all.append(wei[0].cpu().numpy())
    Attention = np.stack(wei_all, axis=0).mean(axis=0)

    # Layout: title → space → two rows of Q's → space → masked title → space → heatmaps
    n_rows, n_cols = 2, 4
    num_queries_to_show = min(8, T)
    grid_resolution = 150

    fig = plt.figure(figsize=(4 * n_cols, 2.8 * n_rows + 4.5))
    # Four vertical blocks: title, two full-height rows of Q gradients, section title, heatmaps
    subfigs = fig.subfigures(4, 1, height_ratios=[0.12, 1.25, 0.08, 1.0], hspace=0.18)

    # Block 1: Title only
    subfigs[0].text(0.5, 0.5, f"Dot Product Gradients for Each Query\nSequence: {seq_str}",
                    ha='center', va='center', fontsize=13, fontweight='bold', transform=subfigs[0].transSubfigure)

    # Block 2: Two rows of Q gradient plots (extra vspace so row 1 x-ticks don't overlap row 2 titles)
    gs_q = GridSpec(2, n_cols, figure=subfigs[1], hspace=0.78, wspace=0.38,
                    left=0.08, right=0.96, top=0.96, bottom=0.06)
    x_grid = np.linspace(x_min, x_max, grid_resolution)
    y_grid = np.linspace(y_min, y_max, grid_resolution)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)

    # One shared colormap range for all gradient panels (PC-plane linear functional q·(x,y))
    dot_grids = []
    for idx in range(num_queries_to_show):
        q_focus_2d = Q_2d[idx]
        dot_grids.append(X_grid * q_focus_2d[0] + Y_grid * q_focus_2d[1])
    dot_stack = np.stack(dot_grids, axis=0)
    global_vmin = float(np.min(dot_stack))
    global_vmax = float(np.max(dot_stack))
    if global_vmin == global_vmax:
        global_vmin -= 1e-6
        global_vmax += 1e-6
    norm_bg = mcolors.Normalize(vmin=global_vmin, vmax=global_vmax)

    _grad_axes = []
    for idx in range(num_queries_to_show):
        row = idx // n_cols
        col = idx % n_cols
        ax = subfigs[1].add_subplot(gs_q[row, col])
        _grad_axes.append(ax)

        ax.pcolormesh(
            x_grid, y_grid, dot_grids[idx],
            cmap="Greens", shading="auto", zorder=0, norm=norm_bg,
        )
        
        # Add origin lines (dashed, faded)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4, zorder=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4, zorder=0.5)
        
        # Plot K points: red if position <= query position, gray if masked (future)
        _k_fs = 8 if _u._JOURNAL_MODE else 11
        for k_idx in range(len(K_2d)):
            is_visible = k_idx <= idx
            color = 'red' if is_visible else '#999999'
            alpha = 1.0 if is_visible else 0.4
            ax.text(K_2d[k_idx, 0], K_2d[k_idx, 1], 
                   _token_pos_label(tokens[k_idx], k_idx),
                   fontsize=_k_fs, ha='center', va='center', 
                   color=color, fontweight='bold', alpha=alpha, zorder=2)
        
        # Highlight the focus Q point
        _q_focus_fs = 14 if _u._JOURNAL_MODE else 16
        ax.text(Q_2d[idx, 0], Q_2d[idx, 1],
               _token_pos_label(tokens[idx], idx),
               fontsize=_q_focus_fs, fontweight='bold', ha='center', va='center',
               color='blue', zorder=3)

        # Plot other Q points (smaller in journal mode to reduce overlap)
        _q_other_fs = 7 if _u._JOURNAL_MODE else 10
        for q_idx in range(len(Q_2d)):
            if q_idx != idx:
                ax.text(Q_2d[q_idx, 0], Q_2d[q_idx, 1],
                       _token_pos_label(tokens[q_idx], q_idx),
                       fontsize=_q_other_fs, ha='center', va='center',
                       color='#A0C4E8', alpha=0.75, zorder=1)
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        pca_suffix = " (PCA)" if Q.shape[1] > 2 else ""
        if row == n_rows - 1:
            ax.set_xlabel("Dim 0" + pca_suffix, fontsize=10)
        if col == 0:
            ax.set_ylabel("Dim 1" + pca_suffix, fontsize=10, labelpad=14)
        ax.set_title(f"Q: {_token_pos_label(tokens[idx], idx)}", fontsize=12, fontweight='bold', pad=8)
        ax.grid(True, alpha=0.3)
        # ax.set_aspect('equal', adjustable='box')
        ax.tick_params(axis='y', left=True, right=False, labelleft=True, labelright=False)

    _cbar_fs = 9 if _u._JOURNAL_MODE else 10
    _sm_bg = ScalarMappable(norm=norm_bg, cmap="Greens")
    _sm_bg.set_array([])
    _cbar = fig.colorbar(
        _sm_bg, ax=_grad_axes, fraction=0.035, pad=0.02, shrink=0.85,
    )
    _cbar.set_ticks([global_vmin, global_vmax])
    _cbar.set_ticklabels([f"{global_vmin:.3g}", f"{global_vmax:.3g}"])
    _cbar.set_label(
        r"Dot product $q^{\top} k$ in PC plane (shared scale across queries)",
        fontsize=_cbar_fs,
    )
    _cbar.ax.tick_params(labelsize=_cbar_fs - 1)

    # Block 3: Section title with proper superscript for transpose
    _row_title_fs = 10 if _u._JOURNAL_MODE else 12
    subfigs[2].text(0.5, 0.5, r"Masked Q$\cdot$K$^T$ and Attention", ha='center', va='center',
                    fontsize=_row_title_fs, fontweight='bold', transform=subfigs[2].transSubfigure)

    # Block 4: Heatmaps row
    gs_heat = GridSpec(1, 2, figure=subfigs[3], wspace=0.25, left=0.08, right=0.96, top=0.92, bottom=0.08)
    ax_masked = subfigs[3].add_subplot(gs_heat[0, 0])
    annot_mask = np.empty((T, T), dtype=object)
    for i in range(T):
        for j in range(T):
            if np.isfinite(Masked_QK_T[i, j]):
                annot_mask[i, j] = f"{Masked_QK_T[i, j]:.3g}"
            else:
                annot_mask[i, j] = ""
    finite_vals = Masked_QK_T[np.isfinite(Masked_QK_T)]
    lim = np.abs(finite_vals).max() if len(finite_vals) else 1
    _annot_fs = 10 if _u._JOURNAL_MODE else 8
    sns.heatmap(Masked_QK_T, cmap="RdBu_r", center=0, xticklabels=tokens,
               yticklabels=tokens, cbar=True, ax=ax_masked, annot=annot_mask, fmt="",
               vmin=-lim, vmax=lim, annot_kws={"fontsize": _annot_fs})
    ax_masked.set_xlabel("T", fontsize=10)
    ax_masked.set_ylabel("T", fontsize=10)
    ax_masked.set_title(r"Masked Q$\cdot$K$^T$ " + f"(T×T={T}×{T})", fontsize=11, pad=12)

    ax_att = subfigs[3].add_subplot(gs_heat[0, 1])
    sns.heatmap(Attention, cmap="jet", vmin=0.0, vmax=1.0,
               xticklabels=tokens, yticklabels=tokens, cbar=True, ax=ax_att)
    ax_att.set_xlabel("T", fontsize=10)
    ax_att.set_ylabel("T", fontsize=10)
    ax_att.set_title(f"Attention (T\u00d7T={T}\u00d7{T})", fontsize=11, pad=18)

    _panel_fs = 11
    _grad_axes[0].text(-0.02, 1.06, "(A)", transform=_grad_axes[0].transAxes,
                       fontsize=_panel_fs, fontweight='bold', va='bottom', ha='left')
    ax_masked.text(-0.02, 1.06, "(B)", transform=ax_masked.transAxes,
                   fontsize=_panel_fs, fontweight='bold', va='bottom', ha='left')
    ax_att.text(-0.02, 1.06, "(C)", transform=ax_att.transAxes,
                fontsize=_panel_fs, fontweight='bold', va='bottom', ha='left')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white')
        plt.close()
        print(f"Q dot product gradients plot saved to {save_path}")
    else:
        plt.show()
    
    model.train()
