"""Plotting: plot_weights_qkv_two_sequences, plot_weights_qkv_single_rows, plot_weights_qkv_single."""
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
def plot_weights_qkv_two_sequences(model, X_list, itos, save_path=None, num_sequences=3):
    """
    Plot QKV for multiple sequences, split into two separate plots.
    Plot 1: Q, K, masked QK^T, Attention, scatter(Q vs K)
    Plot 2: Attention, V, Final Output, scatter(V vs Final Output with lines)
    
    Args:
        model: The model
        X_list: List of input sequences, or single sequence (will be converted to list)
        itos: Index to string mapping
        save_path: Base path to save figures (will save as save_path_part1.png and save_path_part2.png)
        num_sequences: Number of sequences to show (if X_list is single sequence, will use it multiple times)
    """
    model.eval()
    
    # Handle single sequence input
    if not isinstance(X_list, list):
        X_list = [X_list]
    
    # Use provided sequences up to num_sequences
    sequences_to_plot = X_list[:num_sequences]
    num_sequences = len(sequences_to_plot)
    
    # Helper function to compute PCA for 2D visualization
    def pca_2d(data):
        """Reduce data to 2D using PCA only when dimension > 2"""
        if data.shape[1] <= 2:
            # Use raw data directly when dimension is 2 or less
            if data.shape[1] == 2:
                return data
            elif data.shape[1] == 1:
                # Pad with zeros if only 1 dimension
                result = np.zeros((data.shape[0], 2))
                result[:, 0] = data[:, 0]
                return result
            else:
                return data[:, :2]
        # Center the data
        data_centered = data - data.mean(axis=0, keepdims=True)
        # SVD for PCA
        U, s, Vt = np.linalg.svd(data_centered, full_matrices=False)
        # Project to first 2 principal components
        return data_centered @ Vt[:2].T
    
    # Prepare data for all sequences
    all_data = []
    for seq_idx, X in enumerate(sequences_to_plot):
        tokens = [itos[i.item()] for i in X[0]]
        seq_str = " ".join(tokens)
        B, T = X.shape
        
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
        
        # Compute masked QK^T (apply causal mask before softmax)
        QK_T_torch = torch.from_numpy(QK_T).float()
        # Create causal mask (lower triangular)
        tril_mask = torch.tril(torch.ones(T, T))
        Masked_QK_T = QK_T_torch.masked_fill(tril_mask == 0, float("-inf")).numpy()  # (T, T)
        
        # Get attention weights (softmax(QK^T) after masking) from all heads
        wei_all = []
        for h in model.sa_heads.heads:
            _, wei = h(x)  # wei: (B, T, T)
            wei_all.append(wei[0].cpu().numpy())  # (T, T)
        
        # Average attention weights across heads
        Attention = np.stack(wei_all, axis=0).mean(axis=0)  # (T, T)
        
        # Compute final output: Attention @ V (this is what actually gets used)
        Final_Output = Attention @ V  # (T, T) @ (T, hs) = (T, hs)
        
        all_data.append({
            'tokens': tokens,
            'seq_str': seq_str,
            'seq_idx': seq_idx,
            'Q': Q,
            'K': K,
            'V': V,
            'Masked_QK_T': Masked_QK_T,
            'Attention': Attention,
            'Final_Output': Final_Output,
            'T': T
        })
    
    # Get all possible Q/K combinations for overlay
    vocab_size = model.token_embedding.weight.shape[0]
    block_size = model.block_size
    head = model.sa_heads.heads[0]  # Use first head for overlay
    W_Q = head.query.weight.detach().cpu().numpy()  # (head_size, n_embd)
    W_K = head.key.weight.detach().cpu().numpy()  # (head_size, n_embd)
    all_token_emb = model.token_embedding.weight.detach().cpu().numpy()  # (vocab_size, n_embd)
    all_pos_emb = model.position_embedding_table.weight.detach().cpu().numpy()  # (block_size, n_embd)
    
    # Create all Q and K combinations
    all_Q_combinations = []
    all_K_combinations = []
    all_QK_labels = []
    for token_idx in range(vocab_size):
        token_str = str(itos[token_idx])
        for pos_idx in range(block_size):
            combined_emb = all_token_emb[token_idx] + all_pos_emb[pos_idx]
            q_vec = W_Q @ combined_emb  # (head_size,)
            k_vec = W_K @ combined_emb  # (head_size,)
            all_Q_combinations.append(q_vec)
            all_K_combinations.append(k_vec)
            all_QK_labels.append(_token_pos_label(token_str, pos_idx))
    
    all_Q_combinations = np.array(all_Q_combinations)  # (vocab_size * block_size, head_size)
    all_K_combinations = np.array(all_K_combinations)  # (vocab_size * block_size, head_size)
    
    # ========== PLOT 1: Q, K, scatter(Q vs K); optionally masked QK^T + Attention on second row ==========
    use_two_rows = num_sequences == 1
    if use_two_rows:
        if _u._JOURNAL_MODE:
            # Single row only: Q, K, Q vs K (no bottom row)
            n_rows_1, n_cols_plot1 = 1, 3
            fig1 = plt.figure(figsize=(7.0, 2.8))
            gs1 = GridSpec(n_rows_1, n_cols_plot1, figure=fig1, hspace=0.4, wspace=0.5)
        else:
            n_rows_1, n_cols_plot1 = 1, 5
            fig1 = plt.figure(figsize=(5 * n_cols_plot1, 5))
            gs1 = GridSpec(n_rows_1, n_cols_plot1, figure=fig1, hspace=0.35, wspace=0.3)
    else:
        num_cols_plot1 = 5
        fig1 = plt.figure(figsize=(6 * num_cols_plot1, 4 * num_sequences))
        gs1 = GridSpec(num_sequences, num_cols_plot1, figure=fig1, hspace=0.4, wspace=0.3)
    
    # Collect all Q and K from all sequences to compute consistent PCA
    all_Q_data = []
    all_K_data = []
    for data_dict in all_data:
        all_Q_data.append(data_dict['Q'])
        all_K_data.append(data_dict['K'])
    all_Q_data = np.vstack(all_Q_data)  # (total_T, head_size)
    all_K_data = np.vstack(all_K_data)  # (total_T, head_size)
    
    # Compute PCA transformation from ALL sequence data (to apply consistently)
    combined_QK_for_pca = np.vstack([all_Q_data, all_K_data, all_Q_combinations, all_K_combinations])
    pca_transform = None
    if combined_QK_for_pca.shape[1] > 2:
        # Compute PCA transformation from all data
        data_centered = combined_QK_for_pca - combined_QK_for_pca.mean(axis=0, keepdims=True)
        U, s, Vt = np.linalg.svd(data_centered, full_matrices=False)
        pca_transform = Vt[:2].T  # (head_size, 2)
        # Apply PCA to all combinations
        all_Q_centered = all_Q_combinations - combined_QK_for_pca.mean(axis=0, keepdims=True)
        all_K_centered = all_K_combinations - combined_QK_for_pca.mean(axis=0, keepdims=True)
        all_Q_2d_overlay = all_Q_centered @ pca_transform
        all_K_2d_overlay = all_K_centered @ pca_transform
    else:
        # Use raw dimensions
        all_Q_2d_overlay = all_Q_combinations[:, :2] if all_Q_combinations.shape[1] >= 2 else pca_2d(all_Q_combinations)
        all_K_2d_overlay = all_K_combinations[:, :2] if all_K_combinations.shape[1] >= 2 else pca_2d(all_K_combinations)
    
    _fig1_axes = []
    for data_dict in all_data:
        seq_idx = data_dict['seq_idx']
        tokens = data_dict['tokens']
        seq_str = data_dict['seq_str']
        Q = data_dict['Q']
        K = data_dict['K']
        Masked_QK_T = data_dict['Masked_QK_T']
        Attention = data_dict['Attention']
        T = data_dict['T']
        if use_two_rows:
            if _u._JOURNAL_MODE:
                r0, r1, c_q, c_k, c_scat, c_masked, c_att = 0, -1, 0, 1, 2, 0, 1  # 1x3 only; r1 unused
            else:
                r0, r1, c_q, c_k, c_scat, c_masked, c_att = 0, 0, 0, 1, 2, 3, 4  # all on row 0
        else:
            r0 = r1 = seq_idx
            c_q, c_k, c_scat, c_masked, c_att = 0, 1, 2, 3, 4

        # Q
        ax = fig1.add_subplot(gs1[r0, c_q])
        _fig1_axes.append(ax)
        dim_str = f"(T×hs={Q.shape[0]}×{Q.shape[1]})"
        sns.heatmap(Q, cmap="viridis", xticklabels=list(range(Q.shape[1])), 
                   yticklabels=tokens, cbar=True, ax=ax)
        ax.set_xlabel("Head size dim", fontsize=10)
        ax.set_ylabel(f"Seq {seq_idx+1}\n{seq_str}\n" if not use_two_rows else f"{seq_str}", fontsize=9)
        if _u._JOURNAL_MODE:
            ax.tick_params(axis='both', labelsize=7)
        ax.set_title("Q " + dim_str, fontsize=11, pad=10, loc='center', color='blue')
        
        # K
        ax = fig1.add_subplot(gs1[r0, c_k])
        _fig1_axes.append(ax)
        dim_str = f"(T×hs={K.shape[0]}×{K.shape[1]})"
        sns.heatmap(K, cmap="viridis", xticklabels=list(range(K.shape[1])), 
                   yticklabels=tokens, cbar=True, ax=ax)
        ax.set_xlabel("Head size dim", fontsize=10)
        ax.set_ylabel("Sequence position", fontsize=10)
        if _u._JOURNAL_MODE:
            ax.tick_params(axis='both', labelsize=7)
        ax.set_title("K " + dim_str, fontsize=11, pad=10, loc='center', color='red')
        
        # Scatter plot Q vs K
        ax = fig1.add_subplot(gs1[r0, c_scat])
        _fig1_axes.append(ax)
        
        # Apply consistent PCA transformation to sequence-specific Q/K
        if pca_transform is not None:
            Q_centered = Q - combined_QK_for_pca.mean(axis=0, keepdims=True)
            K_centered = K - combined_QK_for_pca.mean(axis=0, keepdims=True)
            Q_2d = Q_centered @ pca_transform
            K_2d = K_centered @ pca_transform
        else:
            Q_2d = pca_2d(Q)
            K_2d = pca_2d(K)
        
        # Combine Q and K data (including overlay) to set axis limits properly
        all_data_2d = np.vstack([Q_2d, K_2d, all_Q_2d_overlay, all_K_2d_overlay])
        x_min, x_max = all_data_2d[:, 0].min(), all_data_2d[:, 0].max()
        y_min, y_max = all_data_2d[:, 1].min(), all_data_2d[:, 1].max()
        x_margin = (x_max - x_min) * 0.1
        y_margin = (y_max - y_min) * 0.1
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        
        # Background overlay: ALL Q/K combinations (annotated, lighter grey)
        # Draw overlay FIRST so it's underneath
        for i, label in enumerate(all_QK_labels):
            x_q, y_q = all_Q_2d_overlay[i, 0], all_Q_2d_overlay[i, 1]
            x_k, y_k = all_K_2d_overlay[i, 0], all_K_2d_overlay[i, 1]
            # Check if points are within axis limits before drawing
            if x_min - x_margin <= x_q <= x_max + x_margin and y_min - y_margin <= y_q <= y_max + y_margin:
                ax.text(x_q, y_q, label,
                       fontsize=11, alpha=0.7, ha='center', va='center', 
                       color='#808080', zorder=1)  # Lighter grey
            if x_min - x_margin <= x_k <= x_max + x_margin and y_min - y_margin <= y_k <= y_max + y_margin:
                ax.text(x_k, y_k, label,
                       fontsize=11, alpha=0.7, ha='center', va='center', 
                       color='#808080', zorder=1)  # Lighter grey
        
        # Foreground: Sequence-specific Q/K points
        for i, (token, pos) in enumerate(zip(tokens, range(len(tokens)))):
            # Annotate Q points (blue for Query)
            ax.text(Q_2d[i, 0], Q_2d[i, 1], _token_pos_label(token, pos), 
                   fontsize=14, fontweight='bold', ha='center', va='center', color='blue', zorder=3)
            # Annotate K points (red for Key)
            ax.text(K_2d[i, 0], K_2d[i, 1], _token_pos_label(token, pos), 
                   fontsize=14, fontweight='bold', ha='center', va='center', color='red', zorder=3)
        
        # Update axis labels based on whether PCA was used
        if Q.shape[1] > 2:
            ax.set_xlabel("Principal component 1", fontsize=10)
            ax.set_ylabel("Principal component 2", fontsize=10)
            title_suffix = " (PCA)"
        else:
            ax.set_xlabel("Dimension 1", fontsize=10)
            ax.set_ylabel("Dimension 2", fontsize=10)
            title_suffix = " (raw)"
        ax.set_title("Q vs K" + title_suffix, fontsize=11, pad=10, loc='center')
        # Add origin lines (dashed, faded)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4, zorder=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4, zorder=0.5)
        ax.grid(True, alpha=0.3)
        
        # Bottom row (masked QK^T, Attention) only when layout has second row
        if r1 >= 0:
            # masked Q K^T (clear spacing so it reads "Q K" not "QK")
            ax = fig1.add_subplot(gs1[r1, c_masked])
            _fig1_axes.append(ax)
            dim_str = f"(T×T={T}×{T})"
            # Annotate cells: show values in lower triangle, "—" for masked (upper triangle)
            annot_mask = np.empty((T, T), dtype=object)
            for i in range(T):
                for j in range(T):
                    if np.isfinite(Masked_QK_T[i, j]):
                        annot_mask[i, j] = f"{Masked_QK_T[i, j]:.3g}"
                    else:
                        annot_mask[i, j] = "—"
            # Color scale: diverging (red/blue) so differences are apparent; symmetric about 0
            finite_vals = Masked_QK_T[np.isfinite(Masked_QK_T)]
            if len(finite_vals):
                lim = np.abs(finite_vals).max()
                vmin_m, vmax_m = -lim, lim
            else:
                vmin_m, vmax_m = -1, 1
            sns.heatmap(Masked_QK_T, cmap="RdBu_r", center=0, xticklabels=tokens,
                       yticklabels=tokens, cbar=True, ax=ax, annot=annot_mask, fmt="",
                       vmin=vmin_m, vmax=vmax_m, annot_kws={"fontsize": 5 if _u._JOURNAL_MODE else 6})
            ax.set_xlabel("Sequence position", fontsize=10)
            if _u._JOURNAL_MODE:
                ax.tick_params(axis='both', labelsize=7)
            ax.set_ylabel("Sequence position", fontsize=10)
            ax.set_title("masked QK^T " + dim_str, fontsize=10, pad=10, loc='center')
            
            # Attention: red/blue diverging so differences are more apparent (no annotations)
            ax = fig1.add_subplot(gs1[r1, c_att])
            _fig1_axes.append(ax)
            dim_str = f"(T×T={T}×{T})"
            sns.heatmap(Attention, cmap="jet", vmin=0.0, vmax=1.0,
                       xticklabels=tokens, yticklabels=tokens, cbar=True, ax=ax)
            ax.set_xlabel("Sequence position", fontsize=10)
            ax.set_ylabel("Sequence position", fontsize=10)
            if _u._JOURNAL_MODE:
                ax.tick_params(axis='both', labelsize=7)
            ax.set_title(f"Attention {dim_str}", fontsize=11, pad=10, loc='center')
    
    # Subtitle with sequence (single-row figure)
    _label_panels(_fig1_axes, fontsize=10, y=1.22)
    if use_two_rows and all_data:
        seq_str_sub = all_data[0]['seq_str']
        fig1.suptitle(f"Sequence: {seq_str_sub}", fontsize=10, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor='white')
        plt.close()
    else:
        plt.show()
    
    # Get all possible V combinations for overlay
    W_V = head.value.weight.detach().cpu().numpy()  # (head_size, n_embd)
    all_V_combinations = []
    all_V_labels = []
    for token_idx in range(vocab_size):
        token_str = str(itos[token_idx])
        for pos_idx in range(block_size):
            combined_emb = all_token_emb[token_idx] + all_pos_emb[pos_idx]
            v_vec = W_V @ combined_emb  # (head_size,)
            all_V_combinations.append(v_vec)
            all_V_labels.append(_token_pos_label(token_str, pos_idx))
    
    all_V_combinations = np.array(all_V_combinations)  # (vocab_size * block_size, head_size)
    
    # ========== PLOT 2: Attention, V, Final Output, scatter(V), scatter(Final Output) ==========
    # Final Output = Attention @ V: weighted sum of value vectors where attention weights determine
    # how much each position contributes. For position i: Final_Output[i] = sum_j(Attention[i,j] * V[j])
    use_two_rows_2 = num_sequences == 1
    if use_two_rows_2:
        if _u._JOURNAL_MODE:
            n_rows_2, n_cols_plot2 = 2, 3
            fig2 = plt.figure(figsize=(7.0, 5.5))
            gs2 = GridSpec(n_rows_2, n_cols_plot2, figure=fig2, hspace=0.4, wspace=0.5, height_ratios=[1, 1])
        else:
            n_rows_2, n_cols_plot2 = 1, 5
            fig2 = plt.figure(figsize=(5 * n_cols_plot2, 5))
            gs2 = GridSpec(n_rows_2, n_cols_plot2, figure=fig2, hspace=0.35, wspace=0.3)
    else:
        num_cols_plot2 = 5
        fig2 = plt.figure(figsize=(6 * num_cols_plot2, 4 * num_sequences))
        gs2 = GridSpec(num_sequences, num_cols_plot2, figure=fig2, hspace=0.4, wspace=0.3)
    
    # First pass: collect all V data to compute consistent PCA
    all_V_data = []
    for data_dict in all_data:
        all_V_data.append(data_dict['V'])
    all_V_data = np.vstack(all_V_data)  # (total_T, head_size)
    
    # Compute PCA transformation from ALL V data (to apply consistently)
    combined_V_for_pca = np.vstack([all_V_data, all_V_combinations])
    pca_transform_V = None
    if combined_V_for_pca.shape[1] > 2:
        # Compute PCA transformation from all data
        data_centered = combined_V_for_pca - combined_V_for_pca.mean(axis=0, keepdims=True)
        U, s, Vt = np.linalg.svd(data_centered, full_matrices=False)
        pca_transform_V = Vt[:2].T
        # Apply PCA to all combinations
        all_V_centered = all_V_combinations - combined_V_for_pca.mean(axis=0, keepdims=True)
        all_V_2d_overlay = all_V_centered @ pca_transform_V
    else:
        all_V_2d_overlay = all_V_combinations[:, :2] if all_V_combinations.shape[1] >= 2 else pca_2d(all_V_combinations)
    
    # Collect all scatter data to calculate shared axis limits
    all_V_2d = []
    all_Final_Output_2d = []
    for data_dict in all_data:
        V = data_dict['V']
        Final_Output = data_dict['Final_Output']
        # Apply consistent PCA transformation
        if pca_transform_V is not None:
            V_centered = V - combined_V_for_pca.mean(axis=0, keepdims=True)
            V_2d = V_centered @ pca_transform_V
        else:
            V_2d = pca_2d(V)
        Final_Output_2d = pca_2d(Final_Output)
        all_V_2d.append(V_2d)
        all_Final_Output_2d.append(Final_Output_2d)
    
    # Calculate shared axis limits from all scatter data (including overlay)
    all_scatter_data = np.vstack(all_V_2d + all_Final_Output_2d + [all_V_2d_overlay])
    x_min_shared, x_max_shared = all_scatter_data[:, 0].min(), all_scatter_data[:, 0].max()
    y_min_shared, y_max_shared = all_scatter_data[:, 1].min(), all_scatter_data[:, 1].max()
    x_range_shared = x_max_shared - x_min_shared if x_max_shared != x_min_shared else 1.0
    y_range_shared = y_max_shared - y_min_shared if y_max_shared != y_min_shared else 1.0
    padding = 0.1
    xlim_shared = (x_min_shared - padding * x_range_shared, x_max_shared + padding * x_range_shared)
    ylim_shared = (y_min_shared - padding * y_range_shared, y_max_shared + padding * y_range_shared)
    
    # Second pass: create plots with shared axis limits
    _fig2_axes = []
    for idx, data_dict in enumerate(all_data):
        seq_idx = data_dict['seq_idx']
        tokens = data_dict['tokens']
        seq_str = data_dict['seq_str']
        V = data_dict['V']
        Attention = data_dict['Attention']
        Final_Output = data_dict['Final_Output']
        T = data_dict['T']
        if use_two_rows_2:
            if _u._JOURNAL_MODE:
                r0_2, r1_2 = 0, 1
                c_att2, c_v2, c_final2, c_vscat2, c_finalscat2 = 0, 1, 2, 1, 2  # row0: Att,V,Final; row1: empty,Vscat,Finalscat
            else:
                r0_2 = r1_2 = 0  # all on row 0
                c_att2, c_v2, c_final2, c_vscat2, c_finalscat2 = 0, 1, 2, 3, 4
        else:
            r0_2 = r1_2 = seq_idx
            c_att2, c_v2, c_final2, c_vscat2, c_finalscat2 = 0, 1, 2, 3, 4
        
        # Get pre-computed 2D projections
        V_2d = all_V_2d[idx]
        Final_Output_2d = all_Final_Output_2d[idx]
        
        # Attention: RdBu_r so differences are more apparent (same as 13_qk_attention)
        ax = fig2.add_subplot(gs2[r0_2, c_att2])
        _fig2_axes.append(ax)
        dim_str = f"(T×T={T}×{T})"
        sns.heatmap(Attention, cmap="jet", vmin=0.0, vmax=1.0,
                   xticklabels=tokens, yticklabels=tokens, cbar=True, ax=ax)
        ax.set_xlabel("T", fontsize=10)
        ax.set_ylabel(f"Seq {seq_idx+1}\n{seq_str}\n" if not use_two_rows_2 else seq_str, fontsize=9)
        if _u._JOURNAL_MODE:
            ax.tick_params(axis='both', labelsize=7)
        ax.set_title(f"Attention\n{dim_str}", fontsize=11, pad=6)
        
        # V: same color scheme as Q/K (viridis)
        ax = fig2.add_subplot(gs2[r0_2, c_v2])
        _fig2_axes.append(ax)
        dim_str = f"(T×hs={V.shape[0]}×{V.shape[1]})"
        sns.heatmap(V, cmap="viridis", xticklabels=list(range(V.shape[1])),
                   yticklabels=tokens, cbar=True, ax=ax)
        ax.set_xlabel("hs", fontsize=10)
        ax.set_ylabel("T", fontsize=10)
        if _u._JOURNAL_MODE:
            ax.tick_params(axis='both', labelsize=7)
        ax.set_title(f"V\n{dim_str}", fontsize=11, pad=6)
        
        # Final Output (Attention @ V): same color scheme as Q/K (viridis)
        ax = fig2.add_subplot(gs2[r0_2, c_final2])
        _fig2_axes.append(ax)
        dim_str = f"(T×hs={Final_Output.shape[0]}×{Final_Output.shape[1]})"
        sns.heatmap(Final_Output, cmap="viridis", xticklabels=list(range(Final_Output.shape[1])),
                   yticklabels=tokens, cbar=True, ax=ax)
        ax.set_xlabel("hs", fontsize=10)
        ax.set_ylabel("T", fontsize=10)
        if _u._JOURNAL_MODE:
            ax.tick_params(axis='both', labelsize=7)
        ax.set_title(f"Final Output (Attention@V)\n{dim_str}", fontsize=11, pad=18)

        # Scatter plot for V (under V heatmap)
        ax = fig2.add_subplot(gs2[r1_2, c_vscat2])
        _fig2_axes.append(ax)
        
        ax.set_xlim(xlim_shared)
        ax.set_ylim(ylim_shared)
        
        for i, label in enumerate(all_V_labels):
            x, y = all_V_2d_overlay[i, 0], all_V_2d_overlay[i, 1]
            if xlim_shared[0] <= x <= xlim_shared[1] and ylim_shared[0] <= y <= ylim_shared[1]:
                ax.text(x, y, label,
                       fontsize=5, alpha=0.6, ha='center', va='center',
                       color='#808080', zorder=1)
        for i, (token, pos) in enumerate(zip(tokens, range(len(tokens)))):
            ax.text(V_2d[i, 0], V_2d[i, 1], _token_pos_label(token, pos),
                   fontsize=10, fontweight='bold', ha='center', va='center', color='green', zorder=3)
        
        if V.shape[1] > 2:
            ax.set_xlabel("PC1", fontsize=10)
            ax.set_ylabel("PC2", fontsize=10)
            title_suffix = " (PCA)"
        else:
            ax.set_xlabel("Dim 1", fontsize=10)
            ax.set_ylabel("Dim 2", fontsize=10)
            title_suffix = " (raw)"
        ax.set_title(f"V{title_suffix}", fontsize=11, pad=6)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4, zorder=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4, zorder=0.5)
        ax.grid(True, alpha=0.3)
        
        # Scatter plot for Final Output (under Final Output heatmap)
        ax = fig2.add_subplot(gs2[r1_2, c_finalscat2])
        _fig2_axes.append(ax)
        
        ax.set_xlim(xlim_shared)
        ax.set_ylim(ylim_shared)
        
        for i, (token, pos) in enumerate(zip(tokens, range(len(tokens)))):
            ax.text(Final_Output_2d[i, 0], Final_Output_2d[i, 1], _token_pos_label(token, pos),
                   fontsize=10, fontweight='bold', ha='center', va='center', color='black')
        
        if Final_Output.shape[1] > 2:
            ax.set_xlabel("PC1", fontsize=10)
            ax.set_ylabel("PC2", fontsize=10)
            title_suffix = " (PCA)"
        else:
            ax.set_xlabel("Dim 1", fontsize=10)
            ax.set_ylabel("Dim 2", fontsize=10)
            title_suffix = " (raw)"
        ax.set_title(f"Final Output (Attention@V)\n{title_suffix}", fontsize=11, pad=14)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4, zorder=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.4, zorder=0.5)
        ax.grid(True, alpha=0.3)
    
    _label_panels(_fig2_axes, fontsize=10, y=1.12)
    # Subtitle with sequence (single-row figure)
    if use_two_rows_2 and all_data:
        seq_str_sub = all_data[0]['seq_str']
        fig2.suptitle(f"Sequence: {seq_str_sub}", fontsize=10, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    if save_path:
        # Save fig2 to a different path (value_output instead of query_key_attention)
        import os
        save_dir = os.path.dirname(save_path)
        # Replace the filename to get the value_output path
        save_path_value = os.path.join(save_dir, "16_value_output.png")
        plt.savefig(save_path_value, bbox_inches='tight', dpi=300, facecolor='white')
        plt.close()
    else:
        plt.show()
    
    model.train()


def plot_weights_qkv_single_rows(model, X, itos, fig, gs, row_offset, sequence_str=None, seq_idx=0, show_weights=True):
    """
    Plot a single QKV figure in a subplot grid, using row_offset to position it.
    row_offset: which row to start at (1, 3, 5, etc. for multiple sequences, since row 0 is weights)
    show_weights: if False, don't show weights (they're shown once at the top)
    """
    B, T = X.shape
    
    # Get actual tokens for tick labels
    tokens = [itos[i.item()] for i in X[0]]
    
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
    
    # Compute masked QK^T (apply causal mask before softmax)
    QK_T_torch = torch.from_numpy(QK_T).float()
    # Create causal mask (lower triangular)
    tril_mask = torch.tril(torch.ones(T, T))
    Masked_QK_T = QK_T_torch.masked_fill(tril_mask == 0, float("-inf")).numpy()  # (T, T)
    
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
    
    # Compute final output: Attention @ V (this is what actually gets used)
    Final_Output = Attention @ V  # (T, T) @ (T, hs) = (T, hs)
    
    if show_weights:
        # Row 1: W_Q, W_K, QK^T, W_V, Attention (old behavior for backward compatibility)
        # Get weights - average across heads
        Wq_all, Wk_all, Wv_all = [], [], []
        for h in model.sa_heads.heads:
            Wq_all.append(h.query.weight.cpu().numpy())  # (hs, C)
            Wk_all.append(h.key.weight.cpu().numpy())    # (hs, C)
            Wv_all.append(h.value.weight.cpu().numpy())  # (hs, C)
        
        W_Q = np.stack(Wq_all, axis=0).mean(axis=0).T  # (C, hs) averaged
        W_K = np.stack(Wk_all, axis=0).mean(axis=0).T  # (C, hs) averaged
        W_V = np.stack(Wv_all, axis=0).mean(axis=0).T  # (C, hs) averaged
        
        row1_titles = ["W_Q", "W_K", "QK^T", "W_V", "Attention"]
        row1_data = [W_Q, W_K, QK_T, W_V, Attention]
        
        for j, (data, title) in enumerate(zip(row1_data, row1_titles)):
            ax = fig.add_subplot(gs[row_offset, j])
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
                    if sequence_str:
                        ax.set_ylabel(f"Seq {seq_idx+1}: {sequence_str}\nC", fontsize=9)
                    else:
                        ax.set_ylabel("C", fontsize=10)
                ax.set_xlabel("hs", fontsize=10)
    else:
        # Row 1: Q, K, QK^T, V, Attention (no weights)
        row1_titles = ["Q", "K", "QK^T", "V", "Attention"]
        row1_data = [Q, K, QK_T, V, Attention]
        
        for j, (data, title) in enumerate(zip(row1_data, row1_titles)):
            ax = fig.add_subplot(gs[row_offset, j])
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
                if j == 0:
                    if sequence_str:
                        ax.set_ylabel(f"Seq {seq_idx+1}: {sequence_str}\nT", fontsize=9)
                    else:
                        ax.set_ylabel("T", fontsize=10)
                ax.set_xlabel("T", fontsize=10)
            else:
                # Q, K, V are (T, hs) matrices - use tokens for y-axis
                x_labels = list(range(data.shape[1]))
                y_labels_local = tokens
                dim_str = f"(T×hs={data.shape[0]}×{data.shape[1]})"
                
                sns.heatmap(data, cmap="viridis", xticklabels=x_labels, yticklabels=y_labels_local, cbar=True, ax=ax)
                ax.set_title(f"{title} {dim_str}", fontsize=11)
                if j == 0:
                    if sequence_str:
                        ax.set_ylabel(f"Seq {seq_idx+1}: {sequence_str}\nT", fontsize=9)
                    else:
                        ax.set_ylabel("T", fontsize=10)
                ax.set_xlabel("hs", fontsize=10)
    
    # Row 2: Q, K, masked QK^T, Attention, V, Final Output
    row2_titles = ["Q", "K", "masked QK^T", "Attention", "V", "Final Output"]
    row2_data = [Q, K, Masked_QK_T, Attention, V, Final_Output]
    
    for j, (data, title) in enumerate(zip(row2_data, row2_titles)):
        ax = fig.add_subplot(gs[row_offset + 1, j])
        if title in ["masked QK^T", "Attention"]:
            # masked QK^T and Attention are (T, T) matrices - use tokens for both axes
            x_labels = tokens
            y_labels_local = tokens
            dim_str = f"(T×T={data.shape[0]}×{data.shape[1]})"
            
            cmap = "magma" if title == "Attention" else "viridis"
            vmin, vmax = (0.0, 1.0) if title == "Attention" else (None, None)
            
            sns.heatmap(data, cmap=cmap, vmin=vmin, vmax=vmax,
                        xticklabels=x_labels, yticklabels=y_labels_local, cbar=True, ax=ax)
            ax.set_title(f"{title} {dim_str}", fontsize=11)
            if j == 0:
                ax.set_ylabel("T", fontsize=10)
            ax.set_xlabel("T", fontsize=10)
        elif title == "Final Output":
            # Final Output is (T, head_size) matrix - use tokens for y-axis
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


def plot_weights_qkv_single(model, X, itos, fig, col_offset, sequence_str=None):
    """
    Plot a single QKV figure in a subplot grid
    col_offset: 0 for left, 1 for right (each sequence takes 5 columns)
    """
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
    
    # Create axes for this sequence (2 rows, 5 columns, starting at col_offset*5)
    gs = GridSpec(2, 10, figure=fig, hspace=0.3, wspace=0.3)
    
    # Row 1: W_Q, W_K, QK^T, W_V, Attention (switched last two columns)
    row1_titles = ["W_Q", "W_K", "QK^T", "W_V", "Attention"]
    row1_data = [W_Q, W_K, QK_T, W_V, Attention]
    
    for j, (data, title) in enumerate(zip(row1_data, row1_titles)):
        ax = fig.add_subplot(gs[0, col_offset*5 + j])
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
        ax = fig.add_subplot(gs[1, col_offset*5 + j])
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
    
    # Add sequence as title for this column block
    if sequence_str:
        # Add text at the top of this column block (centered over the 5 columns)
        center_x = 0.1 + col_offset*0.5 + 0.2  # Center of the 5-column block
        fig.text(center_x, 0.98, sequence_str, 
                fontsize=12, ha='center', va='top', weight='bold')
