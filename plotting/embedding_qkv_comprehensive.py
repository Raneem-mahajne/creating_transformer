"""Plotting: plot_embedding_qkv_comprehensive, plot_tokenpos_qkv_simple."""
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
def plot_embedding_qkv_comprehensive(model, itos, save_path=None, fixed_limits=None, step_label=None):
    """
    Comprehensive figure showing all key embedding spaces:
    Row 1: Token embeddings, Position embeddings, Token+Position (sum)
    Row 2: Q-transformed, K-transformed, V-transformed
    Row 3: Q and K together (red/blue), Attention-weighted output
    
    Args:
        model: The model to visualize
        itos: Index-to-string mapping
        save_path: Path to save the figure
        fixed_limits: Optional dict with limits for consistent animation
        step_label: Optional step number for title
    """
    import matplotlib.colors as mcolors
    
    model.eval()
    
    # Dynamic font sizing
    def get_fontsize(num_items):
        # Subscript labels are more compact, so we can use larger fonts
        if num_items <= 12:
            return 20
        elif num_items <= 20:
            return 16
        elif num_items <= 40:
            return 13
        elif num_items <= 80:
            return 11
        elif num_items <= 150:
            return 9
        else:
            return 7
    
    # Get embeddings
    embeddings = model.token_embedding.weight.detach().cpu().numpy()
    vocab_size, n_embd = embeddings.shape
    block_size = model.block_size
    pos_emb_all = model.position_embedding_table.weight.detach().cpu().numpy()
    
    # Color maps
    token_cmap = plt.cm.get_cmap('YlOrRd')
    token_colors = [token_cmap(0.3 + 0.6 * i / max(vocab_size - 1, 1)) for i in range(vocab_size)]
    pos_cmap = plt.cm.get_cmap('cool')
    pos_colors = [pos_cmap(0.2 + 0.7 * i / max(block_size - 1, 1)) for i in range(block_size)]
    
    def blend_colors(tc, pc, w=0.6):
        tc_rgb = np.array(mcolors.to_rgb(tc))
        pc_rgb = np.array(mcolors.to_rgb(pc))
        return tuple(w * tc_rgb + (1 - w) * pc_rgb)
    
    # Create token+position combinations
    num_combinations = vocab_size * block_size
    all_combinations = np.zeros((num_combinations, n_embd))
    combo_labels = []
    combo_colors = []
    
    for token_idx in range(vocab_size):
        for pos_idx in range(block_size):
            idx = token_idx * block_size + pos_idx
            all_combinations[idx] = embeddings[token_idx] + pos_emb_all[pos_idx]
            combo_labels.append(_token_pos_label(itos[token_idx], pos_idx))
            combo_colors.append(blend_colors(token_colors[token_idx], pos_colors[pos_idx]))
    
    # Get QKV weights from first head
    head = model.sa_heads.heads[0]
    W_Q = head.query.weight.detach().cpu().numpy()
    W_K = head.key.weight.detach().cpu().numpy()
    W_V = head.value.weight.detach().cpu().numpy()
    head_size = W_Q.shape[0]
    
    # Transform by Q, K, V
    Q_transformed = all_combinations @ W_Q.T
    K_transformed = all_combinations @ W_K.T
    V_transformed = all_combinations @ W_V.T
    
    # Masked attention output only: causal position masking (pos p attends only to 0..p)
    attn_outputs_masked = []
    for query_idx in range(num_combinations):
        pos_p = query_idx % block_size
        q = Q_transformed[query_idx:query_idx+1]  # (1, hs)
        scores = (q @ K_transformed.T) / np.sqrt(head_size)  # (1, N)
        # Mask: attend only to keys at position <= pos_p
        mask = np.ones(num_combinations, dtype=bool)
        for key_idx in range(num_combinations):
            pos_q = key_idx % block_size
            if pos_q > pos_p:
                scores[0, key_idx] = -1e9
                mask[key_idx] = False
        row_max = scores[0, mask].max() if mask.any() else 0.0
        scores_exp = np.exp(scores - row_max)
        scores_exp[0, ~mask] = 0.0
        total = scores_exp.sum()
        attn_weights = scores_exp / total if total > 0 else scores_exp
        out = attn_weights @ V_transformed  # (1, hs)
        attn_outputs_masked.append(out[0])
    attn_outputs_masked = np.array(attn_outputs_masked)  # (N, hs)
    
    # Create figure: 3 rows, 3 columns
    # Smaller size for video frames to avoid ffmpeg encoder issues
    fig_size = (16, 13) if step_label is not None else (24, 20)
    fig = plt.figure(figsize=fig_size)
    
    if step_label is not None:
        fig.suptitle(f"Step: {step_label}", fontsize=20, fontweight='bold', y=0.98)
    
    # Helper to get 2D coords (raw or first 2 dims)
    def get_2d(data):
        if data.shape[1] >= 2:
            return data[:, 0], data[:, 1]
        else:
            return data[:, 0], np.zeros(len(data))
    
    def set_limits(ax, key, x, y):
        if fixed_limits and key in fixed_limits:
            ax.set_xlim(fixed_limits[key][0])
            ax.set_ylim(fixed_limits[key][1])
        else:
            margin_x = 0.15 * (x.max() - x.min()) if x.max() != x.min() else 1
            margin_y = 0.15 * (y.max() - y.min()) if y.max() != y.min() else 1
            ax.set_xlim(x.min() - margin_x, x.max() + margin_x)
            ax.set_ylim(y.min() - margin_y, y.max() + margin_y)
    
    # Row 1: Token, Position, Token+Position
    # Token embeddings
    ax1 = fig.add_subplot(3, 3, 1)
    X_emb = embeddings - embeddings.mean(axis=0)
    x_tok, y_tok = get_2d(X_emb)
    set_limits(ax1, 'token', x_tok, y_tok)
    fs_tok = get_fontsize(vocab_size)
    for i in range(vocab_size):
        ax1.text(x_tok[i], y_tok[i], itos[i], fontsize=fs_tok, fontweight='bold',
                ha='center', va='center', color=token_colors[i])
    ax1.set_title("Token Embeddings", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Dim 0")
    ax1.set_ylabel("Dim 1")
    ax1.grid(True, alpha=0.3)
    
    # Position embeddings
    ax2 = fig.add_subplot(3, 3, 2)
    X_pos = pos_emb_all - pos_emb_all.mean(axis=0)
    x_pos, y_pos = get_2d(X_pos)
    set_limits(ax2, 'position', x_pos, y_pos)
    fs_pos = get_fontsize(block_size)
    for i in range(block_size):
        ax2.text(x_pos[i], y_pos[i], _pos_only_label(i), fontsize=fs_pos, fontweight='bold',
                ha='center', va='center', color=pos_colors[i])
    ax2.set_title("Position Embeddings", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Dim 0")
    ax2.set_ylabel("Dim 1")
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4, zorder=0.5)
    ax2.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4, zorder=0.5)
    ax2.grid(True, alpha=0.3)
    
    # Token+Position
    ax3 = fig.add_subplot(3, 3, 3)
    x_comb, y_comb = get_2d(all_combinations)
    set_limits(ax3, 'combined', x_comb, y_comb)
    fs_comb = get_fontsize(num_combinations)
    for i in range(num_combinations):
        ax3.text(x_comb[i], y_comb[i], combo_labels[i], fontsize=fs_comb, fontweight='bold',
                ha='center', va='center', color=combo_colors[i])
    ax3.set_title("Token+Position (Sum)", fontsize=14, fontweight='bold')
    ax3.set_xlabel("Dim 0")
    ax3.set_ylabel("Dim 1")
    ax3.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4, zorder=0.5)
    ax3.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4, zorder=0.5)
    ax3.grid(True, alpha=0.3)
    
    # Row 2: Q, K, V transformed
    # Q-transformed
    ax4 = fig.add_subplot(3, 3, 4)
    x_q, y_q = get_2d(Q_transformed)
    set_limits(ax4, 'Q', x_q, y_q)
    for i in range(num_combinations):
        ax4.text(x_q[i], y_q[i], combo_labels[i], fontsize=fs_comb, fontweight='bold',
                ha='center', va='center', color='blue', alpha=0.7)
    ax4.set_title("Q-Transformed", fontsize=14, fontweight='bold')
    ax4.set_xlabel("Head Dim 0")
    ax4.set_ylabel("Head Dim 1")
    ax4.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4, zorder=0.5)
    ax4.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4, zorder=0.5)
    ax4.grid(True, alpha=0.3)
    
    # K-transformed
    ax5 = fig.add_subplot(3, 3, 5)
    x_k, y_k = get_2d(K_transformed)
    set_limits(ax5, 'K', x_k, y_k)
    for i in range(num_combinations):
        ax5.text(x_k[i], y_k[i], combo_labels[i], fontsize=fs_comb, fontweight='bold',
                ha='center', va='center', color='red', alpha=0.7)
    ax5.set_title("K-Transformed", fontsize=14, fontweight='bold')
    ax5.set_xlabel("Head Dim 0")
    ax5.set_ylabel("Head Dim 1")
    ax5.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4, zorder=0.5)
    ax5.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4, zorder=0.5)
    ax5.grid(True, alpha=0.3)
    
    # V-transformed
    ax6 = fig.add_subplot(3, 3, 6)
    x_v, y_v = get_2d(V_transformed)
    set_limits(ax6, 'V', x_v, y_v)
    v_cmap = plt.cm.get_cmap('Greens')
    v_colors = [v_cmap(0.3 + 0.6 * i / max(num_combinations - 1, 1)) for i in range(num_combinations)]
    for i in range(num_combinations):
        ax6.text(x_v[i], y_v[i], combo_labels[i], fontsize=fs_comb, fontweight='bold',
                ha='center', va='center', color=v_colors[i])
    ax6.set_title("V-Transformed", fontsize=14, fontweight='bold')
    ax6.set_xlabel("Head Dim 0")
    ax6.set_ylabel("Head Dim 1")
    ax6.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4, zorder=0.5)
    ax6.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4, zorder=0.5)
    ax6.grid(True, alpha=0.3)
    
    # Row 3: Q+K combined, Attention output
    # Q and K together
    ax7 = fig.add_subplot(3, 3, 7)
    # Combine Q and K limits
    all_x = np.concatenate([x_q, x_k])
    all_y = np.concatenate([y_q, y_k])
    set_limits(ax7, 'QK', all_x, all_y)
    # Plot Q in blue
    for i in range(num_combinations):
        ax7.text(x_q[i], y_q[i], combo_labels[i], fontsize=fs_comb, fontweight='bold',
                ha='center', va='center', color='blue', alpha=0.7)
    # Plot K in red
    for i in range(num_combinations):
        ax7.text(x_k[i], y_k[i], combo_labels[i], fontsize=fs_comb, fontweight='bold',
                ha='center', va='center', color='red', alpha=0.7)
    ax7.set_title("Q (blue) and K (red) Together", fontsize=14, fontweight='bold')
    ax7.set_xlabel("Head Dim 0")
    ax7.set_ylabel("Head Dim 1")
    ax7.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4, zorder=0.5)
    ax7.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4, zorder=0.5)
    ax7.grid(True, alpha=0.3)
    # Add legend
    ax7.plot([], [], 'o', color='blue', label='Q', markersize=8)
    ax7.plot([], [], 'o', color='red', label='K', markersize=8)
    ax7.legend(loc='upper right', fontsize=10)
    
    # Masked attention output only (causal: pos p attends only to 0..p) — what we actually use
    ax8 = fig.add_subplot(3, 3, 8)
    x_attn_m, y_attn_m = get_2d(attn_outputs_masked)
    set_limits(ax8, 'attn_masked', x_attn_m, y_attn_m)
    purple_cmap = plt.cm.get_cmap('Purples')
    attn_m_colors = [purple_cmap(0.3 + 0.6 * i / max(num_combinations - 1, 1)) for i in range(num_combinations)]
    for i in range(num_combinations):
        ax8.text(x_attn_m[i], y_attn_m[i], combo_labels[i], fontsize=fs_comb, fontweight='bold',
                ha='center', va='center', color=attn_m_colors[i])
    ax8.set_title("Attention Output (causal mask: p→0..p)", fontsize=14, fontweight='bold')
    ax8.set_xlabel("Head Dim 0")
    ax8.set_ylabel("Head Dim 1")
    ax8.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4, zorder=0.5)
    ax8.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4, zorder=0.5)
    ax8.grid(True, alpha=0.3)
    
    # Hide unused subplot
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96] if step_label else [0, 0, 1, 1])
    
    if save_path:
        # For static figures, keep tight layout
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()


@torch.no_grad()
def plot_tokenpos_qkv_simple(model, itos, save_path=None, fixed_limits=None, step_label: int | None = None):
    """
    Simpler 2x2 figure:
    - Token+position (sum) embedding space
    - Q-transformed space
    - K-transformed space
    - V-transformed space

    This matches the requested \"embeddings, Q, K, V\" learning-dynamics view.
    """
    model.eval()

    # Get embeddings
    embeddings = model.token_embedding.weight.detach().cpu().numpy()
    vocab_size, n_embd = embeddings.shape
    block_size = model.block_size
    pos_emb_all = model.position_embedding_table.weight.detach().cpu().numpy()

    # Create token+position combinations
    num_combinations = vocab_size * block_size
    all_combinations = np.zeros((num_combinations, n_embd))
    combo_labels: list[str] = []
    for token_idx in range(vocab_size):
        for pos_idx in range(block_size):
            idx = token_idx * block_size + pos_idx
            all_combinations[idx] = embeddings[token_idx] + pos_emb_all[pos_idx]
            combo_labels.append(_token_pos_label(itos[token_idx], pos_idx))

    # First head Q/K/V
    head = model.sa_heads.heads[0]
    W_Q = head.query.weight.detach().cpu().numpy()
    W_K = head.key.weight.detach().cpu().numpy()
    W_V = head.value.weight.detach().cpu().numpy()

    Q_transformed = all_combinations @ W_Q.T
    K_transformed = all_combinations @ W_K.T
    V_transformed = all_combinations @ W_V.T

    def get_2d(data):
        if data.shape[1] >= 2:
            return data[:, 0], data[:, 1]
        else:
            return data[:, 0], np.zeros(len(data))

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    if step_label is not None:
        fig.suptitle(f"Step: {step_label}", fontsize=18, fontweight="bold", y=0.98)

    # Panel 1: token+position embeddings (black)
    ax_emb = axes[0, 0]
    x_e, y_e = get_2d(all_combinations)
    ax_emb.scatter(x_e, y_e, s=0, alpha=0)
    for i, label in enumerate(combo_labels):
        ax_emb.text(x_e[i], y_e[i], label, fontsize=9, ha="center", va="center", color="black")
    ax_emb.set_title("Token+Position embeddings", fontsize=12, fontweight="bold")
    ax_emb.set_xlabel("Dim 0", fontsize=10)
    ax_emb.set_ylabel("Dim 1", fontsize=10)
    ax_emb.grid(True, alpha=0.3)

    # Panel 2: Q (blue)
    ax_q = axes[0, 1]
    x_q, y_q = get_2d(Q_transformed)
    ax_q.scatter(x_q, y_q, s=0, alpha=0)
    for i, label in enumerate(combo_labels):
        ax_q.text(x_q[i], y_q[i], label, fontsize=9, ha="center", va="center", color="blue")
    ax_q.set_title("Q-transformed", fontsize=12, fontweight="bold")
    ax_q.set_xlabel("Head dim 0", fontsize=10)
    ax_q.set_ylabel("Head dim 1", fontsize=10)
    ax_q.grid(True, alpha=0.3)

    # Panel 3: K (red)
    ax_k = axes[1, 0]
    x_k, y_k = get_2d(K_transformed)
    ax_k.scatter(x_k, y_k, s=0, alpha=0)
    for i, label in enumerate(combo_labels):
        ax_k.text(x_k[i], y_k[i], label, fontsize=9, ha="center", va="center", color="red")
    ax_k.set_title("K-transformed", fontsize=12, fontweight="bold")
    ax_k.set_xlabel("Head dim 0", fontsize=10)
    ax_k.set_ylabel("Head dim 1", fontsize=10)
    ax_k.grid(True, alpha=0.3)

    # Panel 4: V (green)
    ax_v = axes[1, 1]
    x_v, y_v = get_2d(V_transformed)
    ax_v.scatter(x_v, y_v, s=0, alpha=0)
    for i, label in enumerate(combo_labels):
        ax_v.text(x_v[i], y_v[i], label, fontsize=9, ha="center", va="center", color="green")
    ax_v.set_title("V-transformed", fontsize=12, fontweight="bold")
    ax_v.set_xlabel("Head dim 0", fontsize=10)
    ax_v.set_ylabel("Head dim 1", fontsize=10)
    ax_v.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96] if step_label is not None else None)

    if save_path:
        # For video frames (step_label not None), avoid bbox_inches='tight' so all frames share the same size.
        if step_label is not None:
            plt.savefig(save_path, dpi=150, facecolor="white")
        else:
            plt.savefig(save_path, bbox_inches="tight", dpi=150, facecolor="white")
        plt.close(fig)
    else:
        plt.show()
    
    model.train()
