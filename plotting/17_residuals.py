"""Plotting: plot_residuals, plot_ffn_second_residual_arrows."""
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
def plot_residuals(model, X_list, itos, save_path=None, num_sequences=3):
    """
    Plot residuals: V transformed (after attention), Embeddings (token+position), and Sum (residual connection).
    Shows heatmaps and scatter plots for all three, plus embeddings as arrows from origin.
    
    Args:
        model: The model
        X_list: List of input sequences, or single sequence (will be converted to list)
        itos: Index to string mapping
        save_path: Path to save figure
        num_sequences: Number of sequences to show
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
            if data.shape[1] == 2:
                return data
            elif data.shape[1] == 1:
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
        
        # Get embeddings (token + position)
        token_emb = model.token_embedding(X)  # (B, T, n_embd)
        pos = torch.arange(T, device=X.device) % model.block_size
        pos_emb = model.position_embedding_table(pos)  # (T, n_embd)
        embeddings = (token_emb + pos_emb)[0].cpu().numpy()  # (T, n_embd)
        
        # Get actual attention output from model (this is what gets added in residual connection)
        x = token_emb + pos_emb
        attn_out, _ = model.sa_heads(x)  # (B, T, n_embd) - concatenated across heads
        V_transformed = attn_out[0].cpu().numpy()  # (T, n_embd)
        
        # Also get V and Attention for visualization (averaged across heads)
        v_all = []
        wei_all = []
        for h in model.sa_heads.heads:
            v_all.append(h.value(x)[0].cpu().numpy())  # (T, hs)
            _, wei = h(x)  # wei: (B, T, T)
            wei_all.append(wei[0].cpu().numpy())  # (T, T)
        
        # Get Sum (residual connection: embeddings + V_transformed)
        # V_transformed is already (T, n_embd) from the model output
        Sum = embeddings + V_transformed  # (T, n_embd)
        
        all_data.append({
            'tokens': tokens,
            'seq_str': seq_str,
            'seq_idx': seq_idx,
            'embeddings': embeddings,
            'V_transformed': V_transformed,
            'Sum': Sum,
            'T': T
        })
    
    # When single sequence: 2 rows x 3 cols (Embed, V Trans, Final heatmaps on row 0; Embed sc, V Trans sc, Embed→Final arrows on row 1; no Final scatter).
    use_two_rows_r = num_sequences == 1
    if use_two_rows_r:
        n_rows_r, num_cols = 2, 3
        if _u._JOURNAL_MODE:
            fig = plt.figure(figsize=(7.0, 5.5))
            gs = GridSpec(n_rows_r, num_cols, figure=fig, hspace=0.45, wspace=0.5)
        else:
            fig = plt.figure(figsize=(4 * num_cols, 5 * n_rows_r))
            gs = GridSpec(n_rows_r, num_cols, figure=fig, hspace=0.4, wspace=0.3)
    else:
        num_cols = 6
        fig = plt.figure(figsize=(6 * num_cols, 4 * num_sequences))
        gs = GridSpec(num_sequences, num_cols, figure=fig, hspace=0.4, wspace=0.3)
    
    # Create consistent color mapping for each token+position combination
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    # Get all unique token+position combinations across all sequences
    all_token_pos = set()
    for data_dict in all_data:
        for token, pos in zip(data_dict['tokens'], range(len(data_dict['tokens']))):
            all_token_pos.add((token, pos))
    # Create color map - use tab20 for more distinct colors, cycle if needed
    num_unique = len(all_token_pos)
    cmap = cm.get_cmap('tab20')
    if num_unique > 20:
        # Use a larger colormap if needed
        cmap = cm.get_cmap('tab20')
        colors_list = [cmap(i % 20) for i in range(num_unique)]
    else:
        colors_list = [cmap(i) for i in range(num_unique)]
    
    # Create mapping: (token, pos) -> color
    token_pos_to_color = {}
    for idx, (token, pos) in enumerate(sorted(all_token_pos)):
        token_pos_to_color[(token, pos)] = colors_list[idx]
    
    # Global vmin/vmax for heatmaps (symmetric around 0, shared colorbar)
    all_hm_vals = []
    for data_dict in all_data:
        all_hm_vals.extend(data_dict['embeddings'].ravel())
        all_hm_vals.extend(data_dict['V_transformed'].ravel())
        all_hm_vals.extend(data_dict['Sum'].ravel())
    hm_m = max(abs(np.min(all_hm_vals)), abs(np.max(all_hm_vals))) or 1.0
    hm_vmin, hm_vmax = -hm_m, hm_m

    # First pass: collect all scatter data to calculate shared axis limits
    all_scatter_data = []
    heatmap_axes = []
    for data_dict in all_data:
        V_transformed_2d = pca_2d(data_dict['V_transformed'])
        embeddings_2d = pca_2d(data_dict['embeddings'])  # Needed for arrows column
        Sum_2d = pca_2d(data_dict['Sum'])
        all_scatter_data.append(V_transformed_2d)
        all_scatter_data.append(embeddings_2d)
        all_scatter_data.append(Sum_2d)
    
    # Calculate shared axis limits
    all_scatter_stacked = np.vstack(all_scatter_data)
    x_min_shared, x_max_shared = all_scatter_stacked[:, 0].min(), all_scatter_stacked[:, 0].max()
    y_min_shared, y_max_shared = all_scatter_stacked[:, 1].min(), all_scatter_stacked[:, 1].max()
    x_range_shared = x_max_shared - x_min_shared if x_max_shared != x_min_shared else 1.0
    y_range_shared = y_max_shared - y_min_shared if y_max_shared != y_min_shared else 1.0
    padding = 0.1
    xlim_shared = (x_min_shared - padding * x_range_shared, x_max_shared + padding * x_range_shared)
    ylim_shared = (y_min_shared - padding * y_range_shared, y_max_shared + padding * y_range_shared)
    
    # Second pass: create plots
    _resid_axes = []
    for data_dict in all_data:
        seq_idx = data_dict['seq_idx']
        tokens = data_dict['tokens']
        seq_str = data_dict['seq_str']
        embeddings = data_dict['embeddings']
        V_transformed = data_dict['V_transformed']
        Sum = data_dict['Sum']
        T = data_dict['T']
        if use_two_rows_r:
            # Col 0: Embed (hm+sc). Col 1: V_trans (hm+sc). Col 2: Final hm + Embed→Final arrows.
            r0_r, r1_r = 0, 1
            c_emb_hm, c_v_hm, c_emb_sc, c_v_sc = 0, 1, 0, 1
            c_final_hm = 2       # Final heatmap in col 2, row 0
            c_final_arrow = 2    # Embed→Final arrows in col 2, row 1
        else:
            r0_r = r1_r = r2_r = seq_idx
            c_emb_hm, c_v_hm, c_sum_hm, c_emb_sc, c_v_sc, c_arrow = 0, 1, 2, 3, 4, 5
        
        # Embeddings heatmap (row 0)
        ax = fig.add_subplot(gs[r0_r, c_emb_hm])
        _resid_axes.append(ax)
        dim_str = f"(T×d={T}×{embeddings.shape[1]})"
        sns.heatmap(embeddings, cmap="RdBu_r", center=0, vmin=hm_vmin, vmax=hm_vmax,
                   xticklabels=False, yticklabels=tokens, cbar=False, ax=ax)
        heatmap_axes.append(ax)
        ax.set_xlabel("Dim", fontsize=10)
        ax.set_ylabel(seq_str if use_two_rows_r else f"Seq {seq_idx+1}\n{seq_str}\n", fontsize=9)
        ax.set_title(f"Embed ({dim_str})" if _u._JOURNAL_MODE else f"Embeddings (Token+Pos) {dim_str}", fontsize=9 if _u._JOURNAL_MODE else 11)
        # V Transformed heatmap
        ax = fig.add_subplot(gs[r0_r, c_v_hm])
        _resid_axes.append(ax)
        dim_str = f"(T×d={T}×{V_transformed.shape[1]})"
        sns.heatmap(V_transformed, cmap="RdBu_r", center=0, vmin=hm_vmin, vmax=hm_vmax,
                   xticklabels=False, yticklabels=tokens, cbar=False, ax=ax)
        heatmap_axes.append(ax)
        ax.set_xlabel("Dim", fontsize=10)
        ax.set_ylabel(seq_str if use_two_rows_r else f"Seq {seq_idx+1}\n{seq_str}\n", fontsize=9)
        ax.set_title(f"V Trans ({dim_str})" if _u._JOURNAL_MODE else f"V Transformed (Attention@V) {dim_str}", fontsize=9 if _u._JOURNAL_MODE else 11)
        
        # Final heatmap (col 2, row 0 when single seq)
        if use_two_rows_r:
            ax = fig.add_subplot(gs[r0_r, c_final_hm])
        else:
            ax = fig.add_subplot(gs[r0_r, c_sum_hm])
        dim_str = f"(T×d={T}×{Sum.shape[1]})"
        sns.heatmap(Sum, cmap="RdBu_r", center=0, vmin=hm_vmin, vmax=hm_vmax,
                   xticklabels=False, yticklabels=tokens, cbar=False, ax=ax)
        heatmap_axes.append(ax)
        ax.set_xlabel("Dim", fontsize=10)
        ax.set_ylabel(seq_str if use_two_rows_r else f"Seq {seq_idx+1}\n{seq_str}\n", fontsize=9)
        ax.set_title(f"Final ({dim_str})" if _u._JOURNAL_MODE else f"Final (Embed+V_transformed) {dim_str}", fontsize=9 if _u._JOURNAL_MODE else 11)
        
        # Embeddings scatter (row 1, under Embed heatmap)
        ax = fig.add_subplot(gs[r1_r, c_emb_sc])
        embeddings_2d = pca_2d(embeddings)
        # Mark origin with faint dashed lines
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=5)
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=5)
        for i, (token, pos) in enumerate(zip(tokens, range(len(tokens)))):
            color = token_pos_to_color[(token, pos)]
            ax.text(embeddings_2d[i, 0], embeddings_2d[i, 1], _token_pos_label(token, pos),
                   fontsize=11, fontweight='bold', ha='center', va='center', color=color)
        ax.set_xlim(xlim_shared)
        ax.set_ylim(ylim_shared)
        used_pca = embeddings.shape[1] > 2
        if used_pca:
            ax.set_xlabel("PC1", fontsize=10)
            ax.set_ylabel("PC2", fontsize=10)
            title_suffix = " (PCA)"
        else:
            ax.set_xlabel("Dim 1", fontsize=10)
            ax.set_ylabel("Dim 2", fontsize=10)
            title_suffix = ""
        ax.set_title(f"Embed{title_suffix}", fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # V Transformed scatter (with arrows from origin to each point)
        ax = fig.add_subplot(gs[r1_r, c_v_sc])
        V_transformed_2d = pca_2d(V_transformed)
        # Mark origin with faint dashed lines
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=5)
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=5)
        for i, (token, pos) in enumerate(zip(tokens, range(len(tokens)))):
            color = token_pos_to_color[(token, pos)]
            x, y = V_transformed_2d[i, 0], V_transformed_2d[i, 1]
            # Arrow from origin to this point
            arr_len = np.sqrt(x**2 + y**2)
            head = max(0.12, min(0.25, arr_len * 0.12))
            ax.arrow(0, 0, x, y, head_width=head, head_length=head, fc=color, ec=color, alpha=0.8, length_includes_head=True, width=0.015, zorder=2)
            ax.text(x, y, _token_pos_label(token, pos),
                   fontsize=11, fontweight='bold', ha='center', va='center', color=color, zorder=3)
        ax.set_xlim(xlim_shared)
        ax.set_ylim(ylim_shared)
        used_pca = V_transformed.shape[1] > 2
        if used_pca:
            ax.set_xlabel("PC1", fontsize=10)
            ax.set_ylabel("PC2", fontsize=10)
            title_suffix = " (PCA)"
        else:
            ax.set_xlabel("Dim 1", fontsize=10)
            ax.set_ylabel("Dim 2", fontsize=10)
            title_suffix = ""
        ax.set_title(f"V Transformed{title_suffix}", fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Embeddings → Final arrows (col 2, row 1 when single seq)
        ax = fig.add_subplot(gs[r1_r, c_final_arrow] if use_two_rows_r else gs[r1_r, c_arrow])
        embeddings_2d = pca_2d(embeddings)
        Final_2d = pca_2d(Sum)
        # Mark origin with faint dashed lines
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=5)
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=5)
        # Draw arrows from embeddings to Final points
        for i, (token, pos) in enumerate(zip(tokens, range(len(tokens)))):
            color = token_pos_to_color[(token, pos)]
            # Calculate arrow length for head size scaling
            dx = Final_2d[i, 0] - embeddings_2d[i, 0]
            dy = Final_2d[i, 1] - embeddings_2d[i, 1]
            arrow_length = np.sqrt(dx**2 + dy**2)
            head_size = max(0.15, min(0.3, arrow_length * 0.15))  # Scale head size with arrow length
            # Draw arrow from embeddings to Final
            ax.arrow(embeddings_2d[i, 0], embeddings_2d[i, 1],
                    dx, dy,
                    head_width=head_size, head_length=head_size, fc=color, ec=color, alpha=0.8, length_includes_head=True, width=0.02)
            # Annotate only at beginning (embeddings point)
            ax.text(embeddings_2d[i, 0], embeddings_2d[i, 1], _token_pos_label(token, pos),
                   fontsize=11, fontweight='bold', ha='center', va='center', color=color)
        ax.set_xlim(xlim_shared)
        ax.set_ylim(ylim_shared)
        used_pca = embeddings.shape[1] > 2
        if used_pca:
            ax.set_xlabel("PC1", fontsize=10)
            ax.set_ylabel("PC2", fontsize=10)
            title_suffix = " (PCA)"
        else:
            ax.set_xlabel("Dim 1", fontsize=10)
            ax.set_ylabel("Dim 2", fontsize=10)
            title_suffix = ""
        ax.set_title(f"Embed → Final (Modified by Attention){title_suffix}", fontsize=11)
        ax.grid(True, alpha=0.3)
    
    # Subtitle with sequence (single-sequence figure)
    if use_two_rows_r and all_data:
        seq_str_sub = all_data[0]['seq_str']
        fig.suptitle(f"Sequence: {seq_str_sub}", fontsize=10, y=1.02)
    plt.tight_layout(rect=[0, 0, 0.90, 0.95])
    # Single shared colorbar in right margin (leave room so Embed->Final title is not hidden)
    if heatmap_axes and use_two_rows_r:
        mappable = heatmap_axes[0].collections[0]
        ax_cbar = fig.add_axes([0.91, 0.42, 0.015, 0.35])
        fig.colorbar(mappable, cax=ax_cbar)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()
    
    model.train()


@torch.no_grad()
def plot_ffn_second_residual_arrows(
    model, itos, save_path=None, extent_margin=0.5,
):
    """
    Two panels showing how the FFN and second residual transform each
    token+position embedding in 2D space:
      Left:  x → ffwd(x)          (FFN alone)
      Right: x → x + ffwd(x)      (second residual = skip + FFN)
    """
    model.eval()
    vocab_size = model.token_embedding.weight.shape[0]
    block_size = model.position_embedding_table.weight.shape[0]
    n_embd = model.lm_head.in_features
    if n_embd != 2:
        print(f"plot_ffn_second_residual_arrows: n_embd={n_embd}, need 2. Skipping.")
        return

    with torch.no_grad():
        token_emb = model.token_embedding.weight.detach().cpu().numpy()
        pos_emb = model.position_embedding_table.weight.detach().cpu().numpy()

    all_emb, all_labels, all_ti = [], [], []
    for ti in range(vocab_size):
        for pi in range(block_size):
            all_emb.append(token_emb[ti] + pos_emb[pi])
            all_labels.append(_token_pos_label(itos[ti], pi))
            all_ti.append(ti)
    all_emb = np.array(all_emb)

    dev = next(model.parameters()).device
    with torch.no_grad():
        emb_t = torch.tensor(all_emb, dtype=torch.float32, device=dev)
        ffn_out = model.ffwd(emb_t).cpu().numpy()

    ffn_dest = ffn_out
    resid_dest = all_emb + ffn_out

    # Shared square extent encompassing all points and destinations
    all_pts = np.vstack([all_emb, ffn_dest, resid_dest])
    x_min = all_pts[:, 0].min() - extent_margin
    x_max = all_pts[:, 0].max() + extent_margin
    y_min = all_pts[:, 1].min() - extent_margin
    y_max = all_pts[:, 1].max() + extent_margin
    x_c, y_c = (x_min + x_max) / 2, (y_min + y_max) / 2
    half = max(x_max - x_min, y_max - y_min) / 2
    x_min, x_max = x_c - half, x_c + half
    y_min, y_max = y_c - half, y_c + half

    import matplotlib.cm as cm
    cmap = cm.get_cmap('tab20')

    if _u._JOURNAL_MODE:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.8), sharex=True, sharey=True)
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

    _lbl_fs = 7 if _u._JOURNAL_MODE else 8

    for ax, dest, title in [
        (ax1, ffn_dest, "FFN: x → ffwd(x)"),
        (ax2, resid_dest, "Skip + FFN: x → x + ffwd(x)"),
    ]:
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
        for i in range(len(all_emb)):
            color = cmap(all_ti[i] % 20)
            ax.annotate(
                '', xy=(dest[i, 0], dest[i, 1]),
                xytext=(all_emb[i, 0], all_emb[i, 1]),
                arrowprops=dict(arrowstyle='->', color=color, lw=0.8, alpha=0.6),
                zorder=4,
            )
            ax.text(
                all_emb[i, 0], all_emb[i, 1], all_labels[i],
                fontsize=_lbl_fs, ha='center', va='center',
                color=color, fontweight='bold', zorder=6,
            )
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("dim 0", fontsize=9)
        ax.grid(True, alpha=0.15)

    ax1.set_ylabel("dim 1", fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()
        print(f"FFN + second residual arrows saved to {save_path}")
    else:
        plt.show()
    model.train()
