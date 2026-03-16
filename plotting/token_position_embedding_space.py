"""Plotting: plot_token_position_embedding_space."""
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
def plot_token_position_embedding_space(model, itos, save_path=None):
    """
    BIG figure showing all token-position combinations in original embedding space and PCA space.
    Shows both RAW (first 2 dims) and PCA scatter plots side by side.
    """
    model.eval()
    
    # Get token and position embeddings
    embeddings = model.token_embedding.weight.detach().cpu().numpy()  # (vocab, N_EMBD)
    vocab_size, n_embd = embeddings.shape
    block_size = model.block_size
    pos_emb_all = model.position_embedding_table.weight.detach().cpu().numpy()  # (block_size, n_embd)
    
    # Create all token-position combinations (ALL tokens including special characters)
    num_combinations = vocab_size * block_size
    all_combinations = np.zeros((num_combinations, n_embd))
    labels = []
    
    for token_idx in range(vocab_size):
        token_str = str(itos[token_idx])
        for pos_idx in range(block_size):
            idx = token_idx * block_size + pos_idx
            all_combinations[idx] = embeddings[token_idx] + pos_emb_all[pos_idx]
            labels.append(_token_pos_label(token_str, pos_idx))
    
    # Create BIG figure: 1 row, 2 columns (RAW and PCA)
    fig, axes = plt.subplots(1, 2, figsize=(24, 12))
    
    # Left plot: Original embedding space (Dim 0 vs Dim 1)
    ax1 = axes[0]
    if n_embd >= 2:
        X_orig = all_combinations[:, [0, 1]]
        ax1.scatter(X_orig[:, 0], X_orig[:, 1], s=0, alpha=0)  # s=0 for invisible dots
        for i in range(len(labels)):
            ax1.text(X_orig[i, 0], X_orig[i, 1], labels[i], fontsize=9, ha='center', va='center')
        ax1.set_title(f"Original Token+Position Embeddings: Dim 0 vs Dim 1\n(All tokens, {num_combinations} combinations)", fontsize=14)
        ax1.set_xlabel("Embedding Dim 0", fontsize=12)
        ax1.set_ylabel("Embedding Dim 1", fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
    else:
        # 1D case
        X_orig_1d = all_combinations[:, 0]
        ax1.scatter(X_orig_1d, np.zeros_like(X_orig_1d), s=0, alpha=0)
        for i in range(len(labels)):
            ax1.text(X_orig_1d[i], 0, labels[i], fontsize=9, ha='center', va='center', rotation=90)
        ax1.set_title(f"Original Token+Position Embeddings: Dim 0\n(All tokens, {num_combinations} combinations)", fontsize=14)
        ax1.set_xlabel("Embedding Dim 0", fontsize=12)
        ax1.set_ylabel("")
        ax1.grid(True, alpha=0.3)
        ax1.set_yticks([])
    
    # Right plot: PCA space
    ax2 = axes[1]
    if n_embd >= 2:
        X_comb = all_combinations.astype(np.float64)
        X_comb = X_comb - X_comb.mean(axis=0, keepdims=True)
        _, _, Vt_comb = np.linalg.svd(X_comb, full_matrices=False)
        X2_comb = X_comb @ Vt_comb[:2].T
        
        # Plot with s=0 so only annotations are visible
        ax2.scatter(X2_comb[:, 0], X2_comb[:, 1], s=0, alpha=0)
        for i in range(len(labels)):
            ax2.text(X2_comb[i, 0], X2_comb[i, 1], labels[i], fontsize=9, ha='center', va='center')
        
        ax2.set_title(f"Token+Position Embeddings: PCA 2D\n(All tokens, {num_combinations} combinations)", fontsize=14)
        ax2.set_xlabel("PC1", fontsize=12)
        ax2.set_ylabel("PC2", fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.axis('equal')
    else:
        # 1D embeddings
        X1_comb = all_combinations[:, 0]
        ax2.scatter(X1_comb, np.zeros_like(X1_comb), s=0, alpha=0)
        for i in range(len(labels)):
            ax2.text(X1_comb[i], 0, labels[i], fontsize=9, ha='center', va='center', rotation=90)
        ax2.set_title(f"Token+Position Embeddings: PCA 1D\n(All tokens, {num_combinations} combinations)", fontsize=14)
        ax2.set_xlabel("PC1", fontsize=12)
        ax2.set_ylabel("")
        ax2.grid(True, alpha=0.3)
        ax2.set_yticks([])
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()
    model.train()
