"""Plotting: plot_weights_qkv, plot_qkv_transformations."""
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

    def _set_reasonable_limits(ax, xdata, ydata, max_range=12, margin_pct=0.12):
        """Set xlim/ylim from data with margin; cap at ±max_range."""
        xmin, xmax = xdata.min(), xdata.max()
        ymin, ymax = ydata.min(), ydata.max()
        xspan = max(xmax - xmin, 0.5)
        yspan = max(ymax - ymin, 0.5)
        xm = margin_pct * xspan
        ym = margin_pct * yspan
        ax.set_xlim(np.clip(xmin - xm, -max_range, max_range), np.clip(xmax + xm, -max_range, max_range))
        ax.set_ylim(np.clip(ymin - ym, -max_range, max_range), np.clip(ymax + ym, -max_range, max_range))
    
    # Create figure
    # Journal: 2×2 — (a) original embeddings, (b) Q, (c) K, (d) V; weight as inset in b/c/d
    # Standard: 2 rows × 4 cols (original layout)
    if _u._JOURNAL_MODE:
        fig = plt.figure(figsize=(7.0, 6.2))
        gs = GridSpec(2, 2, figure=fig, wspace=0.28, hspace=0.32, left=0.06, right=0.97, top=0.94, bottom=0.08)
    else:
        fig = plt.figure(figsize=(20, 10))
        gs = GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.3)
    
    _lbl_fs = 9 if _u._JOURNAL_MODE else 9
    _pe = [pe.withStroke(linewidth=2, foreground='white')] if _u._JOURNAL_MODE else []
    _panel_fs = 13 if _u._JOURNAL_MODE else 12
    x_labels = list(range(n_embd))
    y_labels_local = list(range(head_size))

    # Journal: 4 panels — original + Q + K + V, each in own space; weight inset in Q/K/V
    if _u._JOURNAL_MODE and n_embd >= 2 and head_size >= 2:
        X_orig = all_combinations[:, [0, 1]]
        Q_2d = Q_transformed[:, [0, 1]]
        K_2d = K_transformed[:, [0, 1]]
        V_2d = V_transformed[:, [0, 1]]
        _m = 0.12
        def _sq_lims(arr, max_r=12):
            lo = arr.min(axis=0) - _m * np.maximum(arr.max(axis=0) - arr.min(axis=0), 0.5)
            hi = arr.max(axis=0) + _m * np.maximum(arr.max(axis=0) - arr.min(axis=0), 0.5)
            cx = 0.5 * (lo[0] + hi[0])
            cy = 0.5 * (lo[1] + hi[1])
            half = 0.5 * max(hi[0] - lo[0], hi[1] - lo[1])
            return (np.clip(cx - half, -max_r, max_r), np.clip(cx + half, -max_r, max_r),
                    np.clip(cy - half, -max_r, max_r), np.clip(cy + half, -max_r, max_r))
        _panels = [
            (None, "Original", "(a)", "black", X_orig),
            (W_Q, "Q", "(b)", "blue", Q_2d),
            (W_K, "K", "(c)", "red", K_2d),
            (W_V, "V", "(d)", "green", V_2d),
        ]
        for idx, (W, ttl, panel_label, ann_color, pts) in enumerate(_panels):
            row, col = idx // 2, idx % 2
            ax = fig.add_subplot(gs[row, col])
            xl, xr, yl, yr = _sq_lims(pts, max_r=8 if W is None else 12)
            ax.set_xlim(xl, xr)
            ax.set_ylim(yl, yr)
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlabel("Dim 0", fontsize=10)
            ax.set_ylabel("Dim 1" if col == 0 else "", fontsize=10)
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, zorder=0)
            ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5, zorder=0)
            ax.grid(True, alpha=0.2)
            ax.set_title(ttl, fontsize=11)
            ax.tick_params(axis='both', labelsize=9)
            ax.text(-0.05, 1.02, panel_label, transform=ax.transAxes, fontsize=_panel_fs, fontweight='bold', va='bottom')
            for i in range(len(labels)):
                ax.text(pts[i, 0], pts[i, 1], labels[i], fontsize=_lbl_fs, ha='center', va='center',
                        color=ann_color, zorder=1)
            # Weight inset for Q/K/V panels: W_Q, W_K, W_V; axis labels C, hs only (no spines/ticks); moved down to clear panel title
            if W is not None:
                inset = ax.inset_axes([0.66, 0.6, 0.30, 0.30])
                sns.heatmap(W, cmap="viridis", xticklabels=False, yticklabels=False, cbar=False,
                            ax=inset, annot=True, fmt='.1f', annot_kws={'size': 8})
                for t in inset.texts:
                    t.set_backgroundcolor('none')
                inset.set_title(rf"$W_{{{ttl}}}$", fontsize=9, pad=2)
                inset.set_xlabel("C", fontsize=8)
                inset.set_ylabel("hs", fontsize=8)
                inset.set_xticks([])
                inset.set_yticks([])
                for spine in inset.spines.values():
                    spine.set_visible(False)
    else:
        # Standard (non-journal) or 1D: keep original two-row layout
        if not _u._JOURNAL_MODE:
            ax0 = fig.add_subplot(gs[0, 0])
            if n_embd >= 2:
                X_orig = all_combinations[:, [0, 1]]
                ax0.scatter(X_orig[:, 0], X_orig[:, 1], s=0, alpha=0)
                for i in range(len(labels)):
                    ax0.text(X_orig[i, 0], X_orig[i, 1], labels[i], fontsize=_lbl_fs, ha='center', va='center',
                             zorder=2, path_effects=_pe)
                ax0.set_title(f"Original Token+Position Embeddings: Dim 0 vs Dim 1\n(All tokens, {num_combinations} combinations)", fontsize=12)
                ax0.set_xlabel("Embedding Dim 0")
                ax0.set_ylabel("Embedding Dim 1")
                ax0.axhline(y=0, color='gray', linestyle='--', linewidth=1.2, alpha=0.6, zorder=10)
                ax0.axvline(x=0, color='gray', linestyle='--', linewidth=1.2, alpha=0.6, zorder=10)
                ax0.grid(True, alpha=0.3)
                ax0.set_aspect('equal', adjustable='box')
                _set_reasonable_limits(ax0, X_orig[:, 0], X_orig[:, 1], max_range=8, margin_pct=0.15)
            else:
                X_orig_1d = all_combinations[:, 0]
                ax0.scatter(X_orig_1d, np.zeros_like(X_orig_1d), s=0, alpha=0)
                for i in range(len(labels)):
                    ax0.text(X_orig_1d[i], 0, labels[i], fontsize=9, ha='center', va='center', rotation=90)
                ax0.set_title(f"Original Token+Position Embeddings: Dim 0\n(All tokens, {num_combinations} combinations)", fontsize=12)
                ax0.set_xlabel("Embedding Dim 0")
                ax0.set_ylabel("")
                ax0.grid(True, alpha=0.3)
                ax0.set_yticks([])

        _wr1, _wc1 = (0, 1)
        ax1 = fig.add_subplot(gs[_wr1, _wc1])
        sns.heatmap(W_Q, cmap="viridis", xticklabels=x_labels, yticklabels=y_labels_local, cbar=True, ax=ax1)
        ax1.set_title(f"W_Q ({head_size}×{n_embd})", fontsize=12)
        ax1.set_xlabel("C (embedding dim)")
        ax1.set_ylabel("hs (head_size)")
        ax2 = fig.add_subplot(gs[_wr1, _wc1 + 1])
        sns.heatmap(W_K, cmap="viridis", xticklabels=x_labels, yticklabels=y_labels_local, cbar=True, ax=ax2)
        ax2.set_title(f"W_K ({head_size}×{n_embd})", fontsize=12)
        ax2.set_xlabel("C (embedding dim)")
        ax2.set_ylabel("hs (head_size)")
        ax3 = fig.add_subplot(gs[_wr1, _wc1 + 2])
        sns.heatmap(W_V, cmap="viridis", xticklabels=x_labels, yticklabels=y_labels_local, cbar=True, ax=ax3)
        ax3.set_title(f"W_V ({head_size}×{n_embd})", fontsize=12)
        ax3.set_xlabel("C (embedding dim)")
        ax3.set_ylabel("hs (head_size)")
        if not _u._JOURNAL_MODE:
            ax_empty = fig.add_subplot(gs[1, 0])
            ax_empty.axis("off")
        _wr2, _wc2 = (1, 1)
        if head_size >= 2:
            Q_2d = Q_transformed[:, [0, 1]]
            K_2d = K_transformed[:, [0, 1]]
            V_2d = V_transformed[:, [0, 1]]
            _x_all = np.concatenate([Q_2d[:, 0], K_2d[:, 0], V_2d[:, 0]])
            _y_all = np.concatenate([Q_2d[:, 1], K_2d[:, 1], V_2d[:, 1]])
            _max_range, _margin = 12, 0.12
            _xspan = max(_x_all.max() - _x_all.min(), 0.5)
            _yspan = max(_y_all.max() - _y_all.min(), 0.5)
            _xlo = np.clip(_x_all.min() - _margin * _xspan, -_max_range, _max_range)
            _xhi = np.clip(_x_all.max() + _margin * _xspan, -_max_range, _max_range)
            _ylo = np.clip(_y_all.min() - _margin * _yspan, -_max_range, _max_range)
            _yhi = np.clip(_y_all.max() + _margin * _yspan, -_max_range, _max_range)
        ax4 = fig.add_subplot(gs[_wr2, _wc2])
        if head_size >= 2:
            ax4.scatter(Q_2d[:, 0], Q_2d[:, 1], s=0, alpha=0)
            for i in range(len(labels)):
                ax4.text(Q_2d[i, 0], Q_2d[i, 1], labels[i], fontsize=_lbl_fs, ha='center', va='center',
                         color='blue', zorder=2, path_effects=_pe)
            ax4.set_title(f"Q-Transformed: Dim 0 vs Dim 1\n(All tokens, {num_combinations} combinations)", fontsize=12)
            ax4.set_xlabel("Head Dim 0")
            ax4.set_ylabel("Head Dim 1")
            ax4.axhline(y=0, color='gray', linestyle='--', linewidth=1.2, alpha=0.6, zorder=10)
            ax4.axvline(x=0, color='gray', linestyle='--', linewidth=1.2, alpha=0.6, zorder=10)
            ax4.grid(True, alpha=0.3)
            ax4.set_xlim(_xlo, _xhi)
            ax4.set_ylim(_ylo, _yhi)
            ax4.set_aspect('equal', adjustable='box')
        else:
            Q_1d = Q_transformed[:, 0]
            ax4.scatter(Q_1d, np.zeros_like(Q_1d), s=0, alpha=0)
            for i in range(len(labels)):
                ax4.text(Q_1d[i], 0, labels[i], fontsize=_lbl_fs, ha='center', va='center', rotation=90, color='blue')
            ax4.set_title(f"Q-Transformed: Dim 0\n(All tokens, {num_combinations} combinations)", fontsize=12)
            ax4.set_xlabel("Head Dim 0")
            ax4.set_ylabel("")
            ax4.grid(True, alpha=0.3)
            ax4.set_yticks([])
        ax5 = fig.add_subplot(gs[_wr2, _wc2 + 1])
        if head_size >= 2:
            ax5.scatter(K_2d[:, 0], K_2d[:, 1], s=0, alpha=0)
            for i in range(len(labels)):
                ax5.text(K_2d[i, 0], K_2d[i, 1], labels[i], fontsize=_lbl_fs, ha='center', va='center',
                         color='red', zorder=2, path_effects=_pe)
            ax5.set_title(f"K-Transformed: Dim 0 vs Dim 1\n(All tokens, {num_combinations} combinations)", fontsize=12)
            ax5.set_xlabel("Head Dim 0")
            ax5.set_ylabel("Head Dim 1")
            ax5.axhline(y=0, color='gray', linestyle='--', linewidth=1.2, alpha=0.6, zorder=10)
            ax5.axvline(x=0, color='gray', linestyle='--', linewidth=1.2, alpha=0.6, zorder=10)
            ax5.grid(True, alpha=0.3)
            ax5.set_xlim(_xlo, _xhi)
            ax5.set_ylim(_ylo, _yhi)
            ax5.set_aspect('equal', adjustable='box')
        else:
            K_1d = K_transformed[:, 0]
            ax5.scatter(K_1d, np.zeros_like(K_1d), s=0, alpha=0)
            for i in range(len(labels)):
                ax5.text(K_1d[i], 0, labels[i], fontsize=_lbl_fs, ha='center', va='center', rotation=90, color='red')
            ax5.set_title(f"K-Transformed: Dim 0\n(All tokens, {num_combinations} combinations)", fontsize=12)
            ax5.set_xlabel("Head Dim 0")
            ax5.set_ylabel("")
            ax5.grid(True, alpha=0.3)
            ax5.set_yticks([])
        ax6 = fig.add_subplot(gs[_wr2, _wc2 + 2])
        v_color = 'green'
        if head_size >= 2:
            ax6.scatter(V_2d[:, 0], V_2d[:, 1], s=0, alpha=0)
            for i in range(len(labels)):
                ax6.text(V_2d[i, 0], V_2d[i, 1], labels[i], fontsize=_lbl_fs, ha='center', va='center',
                         color=v_color, zorder=2, path_effects=_pe)
            ax6.set_title(f"V-Transformed: Dim 0 vs Dim 1\n(All tokens, {num_combinations} combinations)", fontsize=12)
            ax6.set_xlabel("Head Dim 0")
            ax6.set_ylabel("Head Dim 1")
            ax6.axhline(y=0, color='gray', linestyle='--', linewidth=1.2, alpha=0.6, zorder=10)
            ax6.axvline(x=0, color='gray', linestyle='--', linewidth=1.2, alpha=0.6, zorder=10)
            ax6.grid(True, alpha=0.3)
            ax6.set_xlim(_xlo, _xhi)
            ax6.set_ylim(_ylo, _yhi)
            ax6.set_aspect('equal', adjustable='box')
        else:
            V_1d = V_transformed[:, 0]
            margin = 0.15 * (V_1d.max() - V_1d.min())
            ax6.set_xlim(V_1d.min() - margin, V_1d.max() + margin)
            ax6.set_ylim(-0.5, 0.5)
            for i in range(len(labels)):
                ax6.text(V_1d[i], 0, labels[i], fontsize=_lbl_fs, ha='center', va='center', color=v_color)
            ax6.set_title(f"V-Transformed: Dim 0\n(All tokens, {num_combinations} combinations)", fontsize=12)
            ax6.set_xlabel("Head Dim 0")
            ax6.set_ylabel("")
            ax6.grid(True, alpha=0.3)
            ax6.set_yticks([])

    if not _u._JOURNAL_MODE:
        plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()
    model.train()
