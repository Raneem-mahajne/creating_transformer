"""Plotting: plot_qk_embedding_space, plot_qk_embedding_space_focused_query."""
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
def plot_qk_embedding_space(model, itos, save_path: str = None, step_label: int | None = None):
    """
    Create a single scatter plot showing ALL Q and K transformed embeddings
    with both token AND position labels for every combination.
    Uses consistent format: token with position subscript (e.g. 8₃)
    
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
            labels.append(_token_pos_label(token_str, p))
    
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
    if _u._JOURNAL_MODE:
        fig, ax = plt.subplots(figsize=(7.0, 6.0))
    else:
        fig, ax = plt.subplots(figsize=(20, 16))
    if step_label is not None:
        _suptitle_fs = 10 if _u._JOURNAL_MODE else 18
        fig.suptitle(f"Step: {step_label}", fontsize=_suptitle_fs, fontweight="bold", y=0.98)
    label_fontsize = 9 if _u._JOURNAL_MODE else 20
    title_fontsize = 9 if _u._JOURNAL_MODE else 24
    axis_fontsize = 8 if _u._JOURNAL_MODE else 24
    tick_fontsize = 7 if _u._JOURNAL_MODE else 22

    # Plot invisible scatter points (for axis scaling)
    all_x = np.concatenate([Q_2d[:, 0], K_2d[:, 0]])
    all_y = np.concatenate([Q_2d[:, 1], K_2d[:, 1]])
    ax.scatter(all_x, all_y, s=0, alpha=0)

    # Add text labels for ALL Q points (blue)
    for i in range(num_combinations):
        ax.text(Q_2d[i, 0], Q_2d[i, 1], labels[i], fontsize=label_fontsize, ha='center', va='center', color='blue')

    # Add text labels for ALL K points (red)
    for i in range(num_combinations):
        ax.text(K_2d[i, 0], K_2d[i, 1], labels[i], fontsize=label_fontsize, ha='center', va='center', color='red')

    ax.set_xlabel("Dimension 1" + (" (PCA)" if head_size != 2 else ""), fontsize=axis_fontsize)
    ax.set_ylabel("Dimension 2" + (" (PCA)" if head_size != 2 else ""), fontsize=axis_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    ax.set_title(f"Q and K Embedding Space\n{num_combinations} Q (blue) + {num_combinations} K (red) = {2*num_combinations} total\n({vocab_size} tokens × {block_size} positions)", fontsize=title_fontsize, fontweight='bold')
    # Add origin lines (dashed, faded)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4, zorder=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4, zorder=0.5)
    ax.grid(True, alpha=0.3)

    # Legend with actual colored patches
    legend_handles = [
        Patch(facecolor='blue', edgecolor='black', label='Query'),
        Patch(facecolor='red', edgecolor='black', label='Key'),
    ]
    _leg_fs = 8 if _u._JOURNAL_MODE else axis_fontsize
    leg = ax.legend(handles=legend_handles, loc='upper left', fontsize=_leg_fs)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()
        print(f"Q/K embedding space plot saved to {save_path}")
    else:
        plt.show()


@torch.no_grad()
def plot_qk_embedding_space_focused_query(model, itos, token_str="+", position=5, save_path=None, grid_resolution=150):
    """
    One query only (e.g. +_5): show that query, all keys (keys with position >= focus position grayed),
    and background heatmap of dot product between (x,y) and the focus query vector.
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
    Q_all = []
    K_all = []
    labels = []
    for t in range(vocab_size):
        for p in range(block_size):
            combined_emb = token_emb[t] + pos_emb[p]
            q = W_Q @ combined_emb
            k = W_K @ combined_emb
            Q_all.append(q)
            K_all.append(k)
            labels.append(_token_pos_label(str(itos[t]), p))
    Q_all = np.array(Q_all)
    K_all = np.array(K_all)

    if head_size != 2:
        from sklearn.decomposition import PCA
        combined = np.vstack([Q_all, K_all])
        pca = PCA(n_components=2)
        combined_2d = pca.fit_transform(combined)
        Q_2d = combined_2d[:num_combinations]
        K_2d = combined_2d[num_combinations:]
    else:
        Q_2d = Q_all
        K_2d = K_all

    # Index of focus query: token_str at position
    t_focus = None
    for t in range(vocab_size):
        if str(itos[t]) == token_str:
            t_focus = t
            break
    if t_focus is None:
        print(f"plot_qk_embedding_space_focused_query: token '{token_str}' not found in vocab. Skipping.")
        return
    idx_focus = t_focus * block_size + position
    q_focus = Q_2d[idx_focus]  # (2,)

    # Extent with margin
    all_x = np.concatenate([Q_2d[:, 0], K_2d[:, 0]])
    all_y = np.concatenate([Q_2d[:, 1], K_2d[:, 1]])
    margin_x = max(0.5, (all_x.max() - all_x.min()) * 0.1)
    margin_y = max(0.5, (all_y.max() - all_y.min()) * 0.1)
    x_min, x_max = all_x.min() - margin_x, all_x.max() + margin_x
    y_min, y_max = all_y.min() - margin_y, all_y.max() + margin_y

    # Grid for background dot-product heatmap
    xx = np.linspace(x_min, x_max, grid_resolution)
    yy = np.linspace(y_min, y_max, grid_resolution)
    Xgrid, Ygrid = np.meshgrid(xx, yy)
    # At each (x,y) the "key" is (x,y); dot product with q_focus
    dot_grid = Xgrid * q_focus[0] + Ygrid * q_focus[1]

    if _u._JOURNAL_MODE:
        fig, ax = plt.subplots(figsize=(7.0, 6.0))
    else:
        fig, ax = plt.subplots(figsize=(14, 12))
    label_fontsize = 12
    title_fontsize = 12 if _u._JOURNAL_MODE else 18
    axis_fontsize = 9 if _u._JOURNAL_MODE else 20
    tick_fontsize = 7 if _u._JOURNAL_MODE else 18
    legend_fontsize = 7 if _u._JOURNAL_MODE else 14

    # Background heatmap (dot product with focus query)
    im = ax.pcolormesh(xx, yy, dot_grid, cmap='Greens', shading='auto', zorder=0)
    
    # All other queries in very light blue (background context)
    for i in range(num_combinations):
        if i == idx_focus:
            continue
        ax.text(Q_2d[i, 0], Q_2d[i, 1], labels[i], fontsize=label_fontsize - 2, ha='center', va='center', color='#A0C4E8', alpha=0.8, zorder=2)

    # Key points: red if position < position_focus, gray otherwise (masked by causal mask).
    # Exception: keep the key at the focused (+, position) red for readability.
    for i in range(num_combinations):
        p = i % block_size
        t = i // block_size
        tok = str(itos[t])
        is_focus_key = (tok == token_str) and (p == position)
        color = 'red' if (p < position or is_focus_key) else '#666666'
        ax.text(K_2d[i, 0], K_2d[i, 1], labels[i], fontsize=label_fontsize, ha='center', va='center', color=color, zorder=3)

    # Focused query point (bold blue, on top)
    ax.text(Q_2d[idx_focus, 0], Q_2d[idx_focus, 1], labels[idx_focus], fontsize=label_fontsize + 4, ha='center', va='center', color='blue', fontweight='bold', zorder=4)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Dimension 1" + (" (PCA)" if head_size != 2 else ""), fontsize=axis_fontsize)
    ax.set_ylabel("Dimension 2" + (" (PCA)" if head_size != 2 else ""), fontsize=axis_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    ax.set_title(f"Q/K space: focus on query {_token_pos_label(token_str, position)}\nBackground = dot product with this query; keys with position \u2265 {position} grayed", fontsize=title_fontsize, fontweight='bold')
    # Add origin lines (dashed, faded) - make them more visible
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.2, alpha=0.6, zorder=10)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1.2, alpha=0.6, zorder=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    legend_handles = [
        Patch(facecolor='blue', edgecolor='black', label=f'Query {_token_pos_label(token_str, position)}'),
        Patch(facecolor='#A0C4E8', edgecolor='black', label='Other queries'),
        Patch(facecolor='red', edgecolor='black', label=f'Key (position < {position})'),
        Patch(facecolor='#666666', edgecolor='black', label=f'Key (position \u2265 {position})'),
    ]
    ax.legend(handles=legend_handles, loc='upper left', fontsize=legend_fontsize, framealpha=0.9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()
        print(f"Q/K embedding space (focused query) plot saved to {save_path}")
    else:
        plt.show()


@torch.no_grad()
def plot_qk_embedding_space_combined(
    model,
    itos,
    token_str: str = "+",
    position: int = 5,
    save_path: str | None = None,
    grid_resolution: int = 150,
    step_label: int | None = None,
):
    """
    Combine Figure 09 (full Q/K space) and Figure 10 (focused query heatmap) into a single 2-panel figure.

    Top: all Q (blue) and K (red) token–position labels in the Q/K plane.
    Bottom: dot-product background with a focused query (e.g. +_5), with keys at positions >= focus grayed.
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
    Q_all = []
    K_all = []
    labels = []
    for t in range(vocab_size):
        for p in range(block_size):
            combined_emb = token_emb[t] + pos_emb[p]
            Q_all.append(W_Q @ combined_emb)
            K_all.append(W_K @ combined_emb)
            labels.append(_token_pos_label(str(itos[t]), p))
    Q_all = np.array(Q_all)
    K_all = np.array(K_all)

    if head_size != 2:
        from sklearn.decomposition import PCA
        combined = np.vstack([Q_all, K_all])
        pca = PCA(n_components=2)
        combined_2d = pca.fit_transform(combined)
        Q_2d = combined_2d[:num_combinations]
        K_2d = combined_2d[num_combinations:]
        dim_suffix = " (PCA)"
    else:
        Q_2d = Q_all
        K_2d = K_all
        dim_suffix = ""

    # Focus query index
    t_focus = None
    for t in range(vocab_size):
        if str(itos[t]) == token_str:
            t_focus = t
            break
    if t_focus is None:
        print(f"plot_qk_embedding_space_combined: token '{token_str}' not found in vocab. Skipping.")
        return
    idx_focus = t_focus * block_size + position
    q_focus = Q_2d[idx_focus]

    # Shared extent (use all Q/K points, add margin)
    all_x = np.concatenate([Q_2d[:, 0], K_2d[:, 0]])
    all_y = np.concatenate([Q_2d[:, 1], K_2d[:, 1]])
    margin_x = max(0.5, (all_x.max() - all_x.min()) * 0.08)
    margin_y = max(0.5, (all_y.max() - all_y.min()) * 0.08)
    x_min, x_max = all_x.min() - margin_x, all_x.max() + margin_x
    y_min, y_max = all_y.min() - margin_y, all_y.max() + margin_y

    # Prefer constrained layout for multi-panel text-heavy figures.
    if _u._JOURNAL_MODE:
        fig = plt.figure(figsize=(7.0, 8.8), constrained_layout=True)
    else:
        fig = plt.figure(figsize=(18, 18.5), constrained_layout=True)
    # Reduce vertical whitespace between stacked panels in the combined figure.
    fig.set_constrained_layout_pads(hspace=0.0, h_pad=0.0, wspace=0.02, w_pad=0.02)

    if step_label is not None:
        _suptitle_fs = 10 if _u._JOURNAL_MODE else 18
        fig.suptitle(f"Step: {step_label}", fontsize=_suptitle_fs, fontweight="bold", y=0.99)

    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.0, 1.03], hspace=0.0)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_bottom = fig.add_subplot(gs[1, 0])

    # -------------------
    # Left panel: full Q/K space
    # -------------------
    label_fontsize_left = 9 if _u._JOURNAL_MODE else 18
    title_fontsize = 9 if _u._JOURNAL_MODE else 18
    axis_fontsize = 8 if _u._JOURNAL_MODE else 18
    tick_fontsize = 7 if _u._JOURNAL_MODE else 16

    ax_top.scatter(all_x, all_y, s=0, alpha=0)  # scale axes
    for i in range(num_combinations):
        ax_top.text(
            Q_2d[i, 0], Q_2d[i, 1], labels[i],
            fontsize=label_fontsize_left, ha="center", va="center", color="blue",
        )
    for i in range(num_combinations):
        ax_top.text(
            K_2d[i, 0], K_2d[i, 1], labels[i],
            fontsize=label_fontsize_left, ha="center", va="center", color="red",
        )
    ax_top.set_xlim(x_min, x_max)
    ax_top.set_ylim(y_min, y_max)
    ax_top.set_xlabel(f"Dimension 1{dim_suffix}", fontsize=axis_fontsize)
    ax_top.set_ylabel(f"Dimension 2{dim_suffix}", fontsize=axis_fontsize)
    ax_top.tick_params(axis="both", labelsize=tick_fontsize)
    ax_top.set_title(
        f"Q and K Embedding Space\n{num_combinations} Q (blue) and {num_combinations} K (red)",
        fontsize=title_fontsize, fontweight="bold", pad=2, linespacing=1.0,
    )
    ax_top.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.4, zorder=0.5)
    ax_top.axvline(x=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.4, zorder=0.5)
    ax_top.grid(True, alpha=0.25)
    ax_top.legend(
        handles=[
            Patch(facecolor="blue", edgecolor="black", label="Query"),
            Patch(facecolor="red", edgecolor="black", label="Key"),
        ],
        loc="upper left",
        fontsize=(8 if _u._JOURNAL_MODE else axis_fontsize),
        framealpha=0.9,
    )

    # -------------------
    # Right panel: focused query heatmap
    # -------------------
    # Grid for background dot-product heatmap
    xx = np.linspace(x_min, x_max, grid_resolution)
    yy = np.linspace(y_min, y_max, grid_resolution)
    Xgrid, Ygrid = np.meshgrid(xx, yy)
    dot_grid = Xgrid * q_focus[0] + Ygrid * q_focus[1]
    ax_bottom.pcolormesh(xx, yy, dot_grid, cmap="Greens", shading="auto", zorder=0)

    label_fontsize_right = 9 if _u._JOURNAL_MODE else 15
    legend_fontsize = 8 if _u._JOURNAL_MODE else 13

    # Other queries (light blue)
    for i in range(num_combinations):
        if i == idx_focus:
            continue
        ax_bottom.text(
            Q_2d[i, 0], Q_2d[i, 1], labels[i],
            fontsize=max(6, label_fontsize_right - 2),
            ha="center", va="center", color="#A0C4E8", alpha=0.75, zorder=2,
        )
    # Keys (red if unmasked, gray otherwise). Exception: keep the key at the focused (+, position) red.
    for i in range(num_combinations):
        p = i % block_size
        t = i // block_size
        tok = str(itos[t])
        is_focus_key = (tok == token_str) and (p == position)
        color = "red" if (p < position or is_focus_key) else "#666666"
        ax_bottom.text(
            K_2d[i, 0], K_2d[i, 1], labels[i],
            fontsize=label_fontsize_right, ha="center", va="center", color=color, zorder=3,
        )
    # Focus query (bold)
    ax_bottom.text(
        Q_2d[idx_focus, 0], Q_2d[idx_focus, 1], labels[idx_focus],
        fontsize=label_fontsize_right + 4,
        ha="center", va="center", color="blue", fontweight="bold", zorder=4,
    )

    ax_bottom.set_xlim(x_min, x_max)
    ax_bottom.set_ylim(y_min, y_max)
    ax_bottom.set_xlabel(f"Dimension 1{dim_suffix}", fontsize=axis_fontsize)
    ax_bottom.set_ylabel(f"Dimension 2{dim_suffix}", fontsize=axis_fontsize)
    ax_bottom.tick_params(axis="both", labelsize=tick_fontsize)
    bottom_title_fontsize = (title_fontsize + 1) if _u._JOURNAL_MODE else title_fontsize
    ax_bottom.set_title(
        f"Q/K space: focus on query {_token_pos_label(token_str, position)}",
        fontsize=bottom_title_fontsize, fontweight="bold", pad=2, linespacing=1.0,
    )
    ax_bottom.axhline(y=0, color="gray", linestyle="--", linewidth=1.0, alpha=0.55, zorder=10)
    ax_bottom.axvline(x=0, color="gray", linestyle="--", linewidth=1.0, alpha=0.55, zorder=10)
    ax_bottom.grid(True, alpha=0.25)
    ax_bottom.set_aspect("equal", adjustable="box")
    ax_bottom.legend(
        handles=[
            Patch(facecolor="blue", edgecolor="black", label=f"Query {_token_pos_label(token_str, position)}"),
            Patch(facecolor="#A0C4E8", edgecolor="black", label="Other queries"),
            Patch(facecolor="red", edgecolor="black", label=f"Key (position < {position})"),
            Patch(facecolor="#666666", edgecolor="black", label=f"Key (position ≥ {position})"),
        ],
        loc="upper left",
        fontsize=legend_fontsize,
        framealpha=0.9,
    )

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150, facecolor="white")
        plt.close(fig)
        print(f"Q/K combined figure saved to {save_path}")
    else:
        plt.show()
