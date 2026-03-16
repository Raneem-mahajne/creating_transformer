"""Plotting: plot_qk_full_attention_heatmap, plot_qk_full_attention_heatmap_last_row, plot_qk_full_attention_combined, plot_qk_softmax_attention_heatmap."""
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
def plot_qk_full_attention_heatmap(model, itos, save_path: str = None):
    """
    Create a heatmap showing attention scores for ALL token-position combinations.
    X-axis: Key (token-position), Y-axis: Query (token-position), Value: Q·K
    
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
    W_Q = head.query.weight.detach().cpu().numpy()
    W_K = head.key.weight.detach().cpu().numpy()
    head_size = W_Q.shape[0]
    
    # Get embeddings
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
    
    # Compute full attention matrix: Q_all @ K_all.T / sqrt(d)
    attention_matrix = (Q_all @ K_all.T) / np.sqrt(head_size)
    
    # Apply CAUSAL MASKING: query at position p can only attend to keys at position <= p
    query_positions = np.array([p for t in range(vocab_size) for p in range(block_size)])
    key_positions = np.array([p for t in range(vocab_size) for p in range(block_size)])
    
    # Create causal mask: mask[i,j] = True if query_pos[i] >= key_pos[j] (can attend)
    causal_mask = query_positions[:, None] >= key_positions[None, :]  # (132, 132)
    
    # Apply mask: set invalid positions to NaN (will show as white/transparent)
    masked_attention = np.where(causal_mask, attention_matrix, np.nan)
    
    # Create figure
    if _u._JOURNAL_MODE:
        fig, ax = plt.subplots(figsize=(7.0, 7.0))
    else:
        fig, ax = plt.subplots(figsize=(20, 16))
    
    # Use 'nipy_spectral' - maximum color divergence across full spectrum
    im = ax.imshow(masked_attention, cmap='nipy_spectral', aspect='auto')
    
    # Simple token labels at center of each token block
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
    
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels, fontsize=18, fontweight='bold')
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(ytick_labels, fontsize=18, fontweight='bold')
    
    # Add grid lines to separate token groups
    for t in range(vocab_size + 1):
        ax.axhline(y=t * block_size - 0.5, color='white', linewidth=1.5, alpha=0.8)
        ax.axvline(x=t * block_size - 0.5, color='white', linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel("Key Token", fontsize=12)
    ax.set_ylabel("Query Token", fontsize=12)
    ax.set_title(f"Full Attention Matrix: Q·K / √{head_size}\n({vocab_size} tokens × {block_size} positions, positions 0->{block_size-1} top-to-bottom within each token block)", 
                fontsize=14, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label("Attention Score (pre-softmax)", fontsize=10)
    cbar.ax.tick_params(labelsize=18)  # Make colorbar tick labels bigger
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()
        print(f"Full Q/K attention heatmap saved to {save_path}")
    else:
        plt.show()


@torch.no_grad()
def plot_qk_full_attention_heatmap_last_row(model, itos, save_path: str = None):
    """
    Create a zoomed-in view of the last row (Query Token '+') from the full attention matrix.
    Shows each cell as a larger 8x8 sub-heatmap for better clarity.
    
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
    W_Q = head.query.weight.detach().cpu().numpy()
    W_K = head.key.weight.detach().cpu().numpy()
    head_size = W_Q.shape[0]
    
    # Get embeddings
    token_emb = model.token_embedding.weight.detach().cpu().numpy()
    pos_emb = model.position_embedding_table.weight.detach().cpu().numpy()
    
    # Find the '+' token index
    plus_token_idx = None
    for t in range(vocab_size):
        if str(itos[t]) == '+':
            plus_token_idx = t
            break
    
    if plus_token_idx is None:
        print("plot_qk_full_attention_heatmap_last_row: '+' token not found. Skipping.")
        return
    
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
    
    # Compute full attention matrix: Q_all @ K_all.T / sqrt(d)
    attention_matrix = (Q_all @ K_all.T) / np.sqrt(head_size)
    
    # Apply CAUSAL MASKING: query at position p can only attend to keys at position <= p
    query_positions = np.array([p for t in range(vocab_size) for p in range(block_size)])
    key_positions = np.array([p for t in range(vocab_size) for p in range(block_size)])
    
    # Create causal mask: mask[i,j] = True if query_pos[i] >= key_pos[j] (can attend)
    causal_mask = query_positions[:, None] >= key_positions[None, :]
    
    # Apply mask: set invalid positions to NaN
    masked_attention = np.where(causal_mask, attention_matrix, np.nan)
    
    # Extract the last row: Query Token '+' (all positions)
    # The '+' token rows are: plus_token_idx * block_size to (plus_token_idx + 1) * block_size - 1
    plus_query_start = plus_token_idx * block_size
    plus_query_end = (plus_token_idx + 1) * block_size
    last_row_attention = masked_attention[plus_query_start:plus_query_end, :]  # (block_size, num_combinations)
    
    # Journal: 4 cols for A4; else 6 cols.
    n_cols = 4 if _u._JOURNAL_MODE else 6
    n_rows = (vocab_size + n_cols - 1) // n_cols
    if _u._JOURNAL_MODE:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.0, 5.5), sharey=True)
    else:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4.5 * n_rows), sharey=True)
    
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    else:
        axes = np.atleast_2d(axes)
    
    # Get global min/max for consistent color scale
    valid_values = masked_attention[np.isfinite(masked_attention)]
    vmin = valid_values.min()
    vmax = valid_values.max()
    
    for key_token_idx in range(vocab_size):
        row = key_token_idx // n_cols
        col = key_token_idx % n_cols
        ax = axes[row, col]
        
        # Extract the 8x8 sub-matrix for Query '+' vs Key token
        key_start = key_token_idx * block_size
        key_end = (key_token_idx + 1) * block_size
        sub_matrix = last_row_attention[:, key_start:key_end]  # (block_size, block_size)
        
        # Plot the 8x8 heatmap
        im = ax.imshow(sub_matrix, cmap='nipy_spectral', aspect='auto', vmin=vmin, vmax=vmax)
        
        # Add grid lines to separate positions
        for i in range(block_size + 1):
            ax.axhline(y=i - 0.5, color='white', linewidth=1, alpha=0.6)
            ax.axvline(x=i - 0.5, color='white', linewidth=1, alpha=0.6)
        
        # Set ticks to show positions - make them larger and clearer
        _tick_fs = 9 if _u._JOURNAL_MODE else 10
        ax.set_xticks(range(block_size))
        ax.set_xticklabels([f"p{i}" for i in range(block_size)], fontsize=_tick_fs)
        ax.set_yticks(range(block_size))
        ax.set_yticklabels([f"p{i}" for i in range(block_size)], fontsize=_tick_fs)
        
        # Title shows the Key Token - make it larger
        ax.set_title(f"Key: {itos[key_token_idx]}", fontsize=13, fontweight='bold', pad=10)
        
        # Y-label only on first column
        if col == 0:
            ax.set_ylabel("Query '+' Position", fontsize=12, fontweight='bold')
        
        # X-label only on bottom row
        if row == n_rows - 1:
            ax.set_xlabel("Key Position", fontsize=11)
    
    # Hide any unused subplots (if vocab_size < n_rows * n_cols)
    for idx in range(vocab_size, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].axis('off')
    
    # Add overall title - emphasize that each subplot is itself a matrix
    fig.suptitle(f"Query Token '+' Attention to All Key Tokens\n(Each subplot is an {block_size}×{block_size} attention matrix: Query '+' positions (rows) vs Key positions (columns))", 
                 fontsize=14, fontweight='bold', y=0.97)
    
    # Colorbar: anchor to the last subplot only so it sits on the right, not in the middle
    plt.tight_layout(rect=[0, 0, 0.92, 0.94])
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])  # [left, bottom, width, height] in figure coords
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Attention Score (pre-softmax)", fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=200, facecolor='white')
        plt.close()
        print(f"Full Q/K attention heatmap (last row zoom) saved to {save_path}")
    else:
        plt.show()
    
    model.train()


@torch.no_grad()
def plot_qk_full_attention_combined(model, itos, save_path: str = None):
    """
    Full attention heatmap only (no bottom row, no bottom title).
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

    attention_matrix = (Q_all @ K_all.T) / np.sqrt(head_size)

    query_positions = np.array([p for t in range(vocab_size) for p in range(block_size)])
    key_positions = np.array([p for t in range(vocab_size) for p in range(block_size)])
    causal_mask = query_positions[:, None] >= key_positions[None, :]
    masked_attention = np.where(causal_mask, attention_matrix, np.nan)

    valid_values = masked_attention[np.isfinite(masked_attention)]
    vmin, vmax = valid_values.min(), valid_values.max()

    if _u._JOURNAL_MODE:
        fig, ax_top = plt.subplots(figsize=(7.0, 7.0))
    else:
        fig, ax_top = plt.subplots(figsize=(16, 14))
    im_top = ax_top.imshow(masked_attention, cmap='nipy_spectral', aspect='auto', vmin=vmin, vmax=vmax)

    xtick_positions, xtick_labels_list = [], []
    ytick_positions, ytick_labels_list = [], []
    for t in range(vocab_size):
        mid_pos = t * block_size + block_size // 2
        xtick_positions.append(mid_pos)
        xtick_labels_list.append(itos[t])
        ytick_positions.append(mid_pos)
        ytick_labels_list.append(itos[t])

    _tf = 10 if _u._JOURNAL_MODE else 18
    ax_top.set_xticks(xtick_positions)
    ax_top.set_xticklabels(xtick_labels_list, fontsize=_tf, fontweight='bold')
    ax_top.set_yticks(ytick_positions)
    ax_top.set_yticklabels(ytick_labels_list, fontsize=_tf, fontweight='bold')

    for t in range(vocab_size + 1):
        ax_top.axhline(y=t * block_size - 0.5, color='white', linewidth=1.5, alpha=0.8)
        ax_top.axvline(x=t * block_size - 0.5, color='white', linewidth=1.5, alpha=0.8)

    ax_top.set_xlabel("Key Token", fontsize=10 if _u._JOURNAL_MODE else 12)
    ax_top.set_ylabel("Query Token", fontsize=10 if _u._JOURNAL_MODE else 12)
    _top_title = (f"Full Attention Matrix: Q\u00b7K / \u221a{head_size}\n"
                  f"({vocab_size} tokens \u00d7 {block_size} positions)")
    ax_top.set_title(_top_title, fontsize=11 if _u._JOURNAL_MODE else 14, fontweight='bold')
    cbar = plt.colorbar(im_top, ax=ax_top, shrink=0.6, pad=0.02)
    cbar.set_label("Attention Score (pre-softmax)", fontsize=8 if _u._JOURNAL_MODE else 10)
    cbar.ax.tick_params(labelsize=7 if _u._JOURNAL_MODE else 14)

    _label_panels([ax_top], fontsize=12, y=1.04)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=200, facecolor='white')
        plt.close()
        print(f"Combined full Q/K attention heatmap saved to {save_path}")
    else:
        plt.show()

    model.train()


@torch.no_grad()
def plot_qk_softmax_attention_heatmap(model, itos, save_path: str = None):
    """
    Create a heatmap showing SOFTMAX attention weights for ALL token-position combinations.
    This shows the actual attention probabilities after softmax normalization.
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
    Q_all = np.zeros((num_combinations, head_size))
    K_all = np.zeros((num_combinations, head_size))
    
    idx = 0
    for t in range(vocab_size):
        for p in range(block_size):
            combined_emb = token_emb[t] + pos_emb[p]
            Q_all[idx] = W_Q @ combined_emb
            K_all[idx] = W_K @ combined_emb
            idx += 1
    
    # Compute attention scores
    attention_scores = (Q_all @ K_all.T) / np.sqrt(head_size)
    
    # Apply CAUSAL MASKING: query at (token_q, pos_q) can only attend to keys at pos_k <= pos_q
    # Build position indices for masking
    query_positions = np.array([p for t in range(vocab_size) for p in range(block_size)])  # position of each query
    key_positions = np.array([p for t in range(vocab_size) for p in range(block_size)])    # position of each key
    
    # Create causal mask: mask[i,j] = True if query_pos[i] >= key_pos[j] (can attend)
    causal_mask = query_positions[:, None] >= key_positions[None, :]  # (132, 132)
    
    # Apply mask: set invalid positions to -inf before softmax
    masked_scores = np.where(causal_mask, attention_scores, -np.inf)
    
    # Apply softmax to each row (each query attends to valid keys only)
    def softmax(x):
        # Handle rows that are all -inf (query at position 0 with no valid keys of same token)
        x_max = np.max(x, axis=-1, keepdims=True)
        x_max = np.where(np.isinf(x_max), 0, x_max)  # replace -inf max with 0
        exp_x = np.exp(x - x_max)
        exp_x = np.where(np.isinf(x), 0, exp_x)  # -inf -> 0 after exp
        sum_exp = np.sum(exp_x, axis=-1, keepdims=True)
        sum_exp = np.where(sum_exp == 0, 1, sum_exp)  # avoid division by zero
        return exp_x / sum_exp
    
    attention_probs = softmax(masked_scores)
    
    
    # Create figure
    fig, ax = plt.subplots(figsize=(20, 16))
    
    # Use nipy_spectral for max color divergence
    im = ax.imshow(attention_probs, cmap='nipy_spectral', aspect='auto')
    
    # Simple token labels
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
    
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels, fontsize=9, fontweight='bold')
    ax.set_yticks(ytick_positions)
    ax.set_yticklabels(ytick_labels, fontsize=9, fontweight='bold')
    
    # Grid lines to separate token groups
    for t in range(vocab_size + 1):
        ax.axhline(y=t * block_size - 0.5, color='white', linewidth=1.5, alpha=0.8)
        ax.axvline(x=t * block_size - 0.5, color='white', linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel("Key Token", fontsize=12)
    ax.set_ylabel("Query Token", fontsize=12)
    ax.set_title(f"Softmax Attention Probabilities\n({vocab_size} tokens × {block_size} positions, positions 0->{block_size-1} top-to-bottom)", 
                fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label("Attention Probability (after softmax)", fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()
        print(f"Softmax attention heatmap saved to {save_path}")
    else:
        plt.show()
