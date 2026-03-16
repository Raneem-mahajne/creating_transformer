"""Plotting: plot_sequence_embeddings."""
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
def plot_sequence_embeddings(model, X, itos, save_path=None):
    """
    Plot embeddings for a specific sequence showing token embeddings, position embeddings,
    and their combined embeddings in 2D space.
    
    Args:
        model: The model
        X: Input sequence tensor of shape (B, T) - will use first sequence if batch
        itos: Index to string mapping
        save_path: Path to save figure
    """
    model.eval()
    
    # Extract first sequence if batch
    if X.dim() == 2 and X.shape[0] > 1:
        X = X[0:1]  # Take first sequence
    elif X.dim() == 1:
        X = X.unsqueeze(0)  # Add batch dimension
    
    B, T = X.shape
    tokens = [itos[i.item()] for i in X[0]]
    seq_str = " ".join(tokens)
    
    # Get embeddings
    token_emb = model.token_embedding(X)  # (B, T, n_embd)
    positions = torch.arange(T, device=X.device) % model.block_size
    pos_emb = model.position_embedding_table(positions)  # (T, n_embd)
    combined_emb = token_emb + pos_emb  # (B, T, n_embd)
    
    # Convert to numpy
    token_emb_np = token_emb[0].cpu().numpy()  # (T, n_embd)
    pos_emb_np = pos_emb.cpu().numpy()  # (T, n_embd)
    combined_emb_np = combined_emb[0].cpu().numpy()  # (T, n_embd)
    
    n_embd = token_emb_np.shape[1]
    
    # Get all possible embeddings for overlay
    vocab_size = model.token_embedding.weight.shape[0]
    block_size = model.block_size
    all_token_emb = model.token_embedding.weight.detach().cpu().numpy()  # (vocab_size, n_embd)
    all_pos_emb = model.position_embedding_table.weight.detach().cpu().numpy()  # (block_size, n_embd)
    
    # Create all token-position combinations
    all_combined_emb = np.zeros((vocab_size * block_size, n_embd))
    for token_idx in range(vocab_size):
        for pos_idx in range(block_size):
            idx = token_idx * block_size + pos_idx
            all_combined_emb[idx] = all_token_emb[token_idx] + all_pos_emb[pos_idx]
    
    # Create figure: 2 rows, 3 columns with equal row heights
    if _u._JOURNAL_MODE:
        fig = plt.figure(figsize=(7.0, 5.5))
        gs = GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.5, height_ratios=[1, 1])
    else:
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3, height_ratios=[1, 1])
    
    if n_embd >= 2:
        # Row 1: Heatmaps
        _title_fs = 11 if _u._JOURNAL_MODE else 14
        _axis_fs = 10 if _u._JOURNAL_MODE else 12
        # Column 1: Token embeddings heatmap
        ax1 = fig.add_subplot(gs[0, 0])
        sns.heatmap(token_emb_np, cmap="viridis", xticklabels=list(range(n_embd)),
                    yticklabels=tokens, cbar=True, ax=ax1)
        ax1.set_title("Token Embeddings", fontsize=_title_fs, fontweight='bold', pad=6)
        ax1.set_xlabel("Embedding Dim", fontsize=_axis_fs)
        ax1.set_ylabel("Token", fontsize=_axis_fs)
        
        # Column 2: Position embeddings heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        pos_labels = [f"p{i}" for i in range(T)]
        sns.heatmap(pos_emb_np, cmap="viridis", xticklabels=list(range(n_embd)),
                    yticklabels=pos_labels, cbar=True, ax=ax2)
        ax2.set_title("Position Embeddings", fontsize=_title_fs, fontweight='bold', pad=6)
        ax2.set_xlabel("Embedding Dim", fontsize=_axis_fs)
        ax2.set_ylabel("Position", fontsize=_axis_fs)
        
        # Column 3: Combined embeddings heatmap
        ax3 = fig.add_subplot(gs[0, 2])
        labels = [_token_pos_label(tokens[i], i) for i in range(T)]
        sns.heatmap(combined_emb_np, cmap="viridis", xticklabels=list(range(n_embd)),
                    yticklabels=labels, cbar=True, ax=ax3)
        ax3.set_title("Token+Position Embeddings", fontsize=_title_fs, fontweight='bold', pad=6, loc='right')
        ax3.set_xlabel("Embedding Dim", fontsize=_axis_fs)
        ax3.set_ylabel("Token+Position", fontsize=_axis_fs)
        
        # Row 2: Scatter plots
        # Calculate consistent axis limits across all three plots for uniform sizing.
        # IMPORTANT: include ALL embeddings (background + sequence) so nothing gets clipped.
        all_x = np.concatenate(
            [
                all_token_emb[:, 0],
                all_pos_emb[:, 0],
                all_combined_emb[:, 0],
                token_emb_np[:, 0],
                pos_emb_np[:, 0],
                combined_emb_np[:, 0],
            ]
        )
        all_y = np.concatenate(
            [
                all_token_emb[:, 1],
                all_pos_emb[:, 1],
                all_combined_emb[:, 1],
                token_emb_np[:, 1],
                pos_emb_np[:, 1],
                combined_emb_np[:, 1],
            ]
        )
        
        x_min, x_max = all_x.min(), all_x.max()
        y_min, y_max = all_y.min(), all_y.max()
        
        # Add padding
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_pad = x_range * 0.1 if x_range > 0 else 0.1
        y_pad = y_range * 0.1 if y_range > 0 else 0.1
        
        x_lim = (x_min - x_pad, x_max + x_pad)
        y_lim = (y_min - y_pad, y_max + y_pad)
        
        # Column 1: Token embeddings scatter
        ax4 = fig.add_subplot(gs[1, 0])
        _scatter_fg = 12 if _u._JOURNAL_MODE else 14
        _scatter_bg = 10 if _u._JOURNAL_MODE else 14
        # Background: ALL token embeddings (annotated, shaded out)
        for token_idx in range(vocab_size):
            token_str = str(itos[token_idx])
            ax4.text(
                all_token_emb[token_idx, 0],
                all_token_emb[token_idx, 1],
                token_str,
                fontsize=_scatter_bg,
                alpha=0.5,
                ha="center",
                va="center",
                color="dimgray",
                zorder=1,
            )
        # Foreground: tokens from THIS sequence (unchanged labels)
        ax4.scatter(token_emb_np[:, 0], token_emb_np[:, 1], s=0, alpha=0, zorder=2)
        for i, token in enumerate(tokens):
            ax4.text(
                token_emb_np[i, 0],
                token_emb_np[i, 1],
                token,
                fontsize=_scatter_fg,
                fontweight="bold",
                ha="center",
                va="center",
                color="orange",
                zorder=3,
            )
        ax4.set_title("Token Embeddings (2D)", fontsize=_title_fs, fontweight='bold', pad=6)
        ax4.set_xlabel("Dim 0", fontsize=_axis_fs)
        ax4.set_ylabel("Dim 1", fontsize=_axis_fs)
        ax4.set_xlim(x_lim)
        ax4.set_ylim(y_lim)
        ax4.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4, zorder=0.5)
        ax4.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4, zorder=0.5)
        ax4.grid(True, alpha=0.3)
        # Remove aspect='equal' to allow plots to fill width like heatmaps above
        
        # Column 2: Position embeddings scatter
        ax5 = fig.add_subplot(gs[1, 1])
        # Background: ALL position embeddings (annotated, shaded out)
        for pos_idx in range(block_size):
            ax5.text(
                all_pos_emb[pos_idx, 0],
                all_pos_emb[pos_idx, 1],
                _pos_only_label(pos_idx),
                fontsize=_scatter_bg,
                alpha=0.55,
                ha="center",
                va="center",
                color="dimgray",
                zorder=1,
            )
        # Foreground: positions used in THIS sequence
        ax5.scatter(pos_emb_np[:, 0], pos_emb_np[:, 1], s=0, alpha=0, zorder=2)
        for i in range(T):
            ax5.text(
                pos_emb_np[i, 0],
                pos_emb_np[i, 1],
                f"p{i}",
                fontsize=_scatter_fg,
                fontweight="bold",
                ha="center",
                va="center",
                color="teal",
                zorder=3,
            )
        ax5.set_title("Position Embeddings (2D)", fontsize=_title_fs, fontweight='bold', pad=6)
        ax5.set_xlabel("Dim 0", fontsize=_axis_fs)
        ax5.set_ylabel("Dim 1", fontsize=_axis_fs)
        # Use position-embedding-only axis range so points are not squished (panel keeps same height)
        pos_x_min, pos_x_max = all_pos_emb[:, 0].min(), all_pos_emb[:, 0].max()
        pos_y_min, pos_y_max = all_pos_emb[:, 1].min(), all_pos_emb[:, 1].max()
        pos_x_range = pos_x_max - pos_x_min
        pos_y_range = pos_y_max - pos_y_min
        pos_x_pad = pos_x_range * 0.15 if pos_x_range > 0 else 0.1
        pos_y_pad = pos_y_range * 0.15 if pos_y_range > 0 else 0.1
        ax5.set_xlim(pos_x_min - pos_x_pad, pos_x_max + pos_x_pad)
        ax5.set_ylim(pos_y_min - pos_y_pad, pos_y_max + pos_y_pad)
        ax5.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4, zorder=0.5)
        ax5.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4, zorder=0.5)
        ax5.grid(True, alpha=0.3)
        # Remove aspect='equal' to allow plots to fill width like heatmaps above
        
        # Column 3: Combined embeddings scatter
        ax6 = fig.add_subplot(gs[1, 2])
        # Background: ALL token+position combinations (annotated, shaded out)
        for token_idx in range(vocab_size):
            token_str = str(itos[token_idx])
            for pos_idx in range(block_size):
                idx = token_idx * block_size + pos_idx
                label = _token_pos_label(token_str, pos_idx)
                ax6.text(
                    all_combined_emb[idx, 0],
                    all_combined_emb[idx, 1],
                    label,
                    fontsize=_scatter_bg,
                    alpha=0.4,
                    ha="center",
                    va="center",
                    color="dimgray",
                    zorder=1,
                )
        # Foreground: (token, position) pairs from THIS sequence
        ax6.scatter(combined_emb_np[:, 0], combined_emb_np[:, 1], s=0, alpha=0, zorder=2)
        for i in range(T):
            ax6.text(
                combined_emb_np[i, 0],
                combined_emb_np[i, 1],
                labels[i],
                fontsize=_scatter_fg,
                fontweight="bold",
                ha="center",
                va="center",
                color="indigo",
                zorder=3,
            )
        ax6.set_title("Token+Position\nEmbeddings (2D)", fontsize=_title_fs, fontweight='bold', pad=6)
        ax6.set_xlabel("Dim 0", fontsize=_axis_fs)
        ax6.set_ylabel("Dim 1", fontsize=_axis_fs)
        ax6.set_xlim(x_lim)
        ax6.set_ylim(y_lim)
        ax6.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4, zorder=0.5)
        ax6.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4, zorder=0.5)
        ax6.grid(True, alpha=0.3)
        # Remove aspect='equal' to allow plots to fill width like heatmaps above
        
        _label_panels([ax1, ax2, ax3, ax4, ax5, ax6], fontsize=10, y=1.12)

        # Add sequence as supertitle
        _suptitle_fs = 10 if _u._JOURNAL_MODE else 16
        fig.suptitle(f"Sequence Embeddings: {seq_str}", fontsize=_suptitle_fs, fontweight='bold', y=0.98)
    else:
        # 1D case - simpler visualization
        _suptitle_fs = 10 if _u._JOURNAL_MODE else 16
        fig.suptitle(f"Sequence Embeddings (1D): {seq_str}", fontsize=_suptitle_fs, fontweight='bold')
        ax = fig.add_subplot(gs[:, :])
        ax.scatter(combined_emb_np[:, 0], np.zeros(T), s=100)
        for i in range(T):
            ax.text(combined_emb_np[i, 0], 0, labels[i], fontsize=12, ha='center', va='center')
        ax.set_xlabel("Embedding Dim 0", fontsize=12)
        ax.set_yticks([])
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()
        print(f"Sequence embeddings plot saved to {save_path}")
    else:
        plt.show()
    model.train()
