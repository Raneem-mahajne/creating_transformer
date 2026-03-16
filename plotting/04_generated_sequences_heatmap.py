"""Plotting: plot_generated_sequences_heatmap, plot_generated_sequences_heatmap_before_after."""
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


def plot_generated_sequences_heatmap(generated_sequences, generator, save_path=None, num_sequences=5, max_length=20, ax=None, title=None, show_legend=True):
    """
    Plot generated sequences as an annotated heatmap showing correctness.
    Red = incorrect, White/Green = correct.
    """
    sequences_to_show = generated_sequences[:num_sequences]
    
    # Truncate to max_length and pad to same length
    max_len = min(max_length, max(len(seq) for seq in sequences_to_show))
    
    # Create data matrix, correctness matrix, and constrained/valence matrix
    data_matrix = []
    correctness_matrix = []
    constrained_matrix = []
    
    for seq in sequences_to_show:
        seq_truncated = seq[:max_len]
        correctness, _ = generator.verify_sequence(seq_truncated)
        valence = generator.valence_mask(seq_truncated)
        
        # Pad if necessary
        while len(seq_truncated) < max_len:
            seq_truncated.append(None)  # Padding value (masked)
            correctness.append(np.nan)  # Padding marker (masked)
            valence.append(np.nan)
        
        data_matrix.append(seq_truncated)
        correctness_matrix.append(correctness)
        constrained_matrix.append(valence)
    
    data_matrix = np.array(data_matrix, dtype=object)
    correctness_matrix = np.array(correctness_matrix, dtype=float)
    constrained_matrix = np.array(constrained_matrix, dtype=float)
    
    created_fig = False
    if ax is None:
        # Fewer sequences, larger panels and fonts
        fig_height = max(6, num_sequences * 1.4 + 2)
        fig_width = max(14, max_len * 0.65)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        created_fig = True
    
    # Create color matrix: 1 (correct) = light green, 0 (incorrect) = red, 2 (neutral/no valence) = gray
    from matplotlib.colors import ListedColormap
    colors = ['#ff6b6b', '#90EE90', '#d3d3d3']  # red, light green, gray
    cmap = ListedColormap(colors)
    cmap.set_bad(color=(1, 1, 1, 0))  # transparent for padding
    
    # Mask padding so it doesn't render at all, and show neutral positions in gray
    color_indices = np.where(
        np.isnan(correctness_matrix),
        np.nan,
        np.where(constrained_matrix == 0.0, 2.0, correctness_matrix)  # 2=neutral, else 0/1
    )
    masked_colors = np.ma.masked_invalid(color_indices)
    
    # Plot the heatmap
    im = ax.imshow(masked_colors, cmap=cmap, aspect='auto', vmin=0, vmax=2)
    
    # Add text annotations (the actual numbers) — large, readable font
    _txt_fs = 12 if _u._JOURNAL_MODE else 20
    _lbl_fs = 9 if _u._JOURNAL_MODE else 16
    _tit_fs = 10 if _u._JOURNAL_MODE else 17
    _tk_fs = 7 if _u._JOURNAL_MODE else 14
    for i in range(len(sequences_to_show)):
        for j in range(max_len):
            val = data_matrix[i, j]
            if val is not None:  # Not padding
                text_color = 'black'
                ax.text(j, i, str(val), ha='center', va='center', fontsize=_txt_fs, color=text_color, fontweight='bold')
    
    # Set labels — larger fonts
    ax.set_xlabel("Position in Sequence", fontsize=_lbl_fs)
    ax.set_ylabel("Sequence #", fontsize=_lbl_fs)
    ax.set_title(title or "Generated Sequences with Rule Correctness\n(Green = Correct, Red = Incorrect)", fontsize=_tit_fs)
    ax.tick_params(axis='both', labelsize=_tk_fs)
    
    # Set ticks
    ax.set_xticks(range(0, max_len, max(1, max_len // 15)))
    ax.set_yticks(range(num_sequences))
    ax.set_yticklabels([f"Seq {i+1}" for i in range(num_sequences)])
    
    if created_fig:
        plt.tight_layout()
        if show_legend:
            legend_elements = [
                Patch(facecolor='#90EE90', label='Correct'),
                Patch(facecolor='#ff6b6b', label='Incorrect'),
                Patch(facecolor='#d3d3d3', label='Neutral'),
            ]
            _leg_fs = 8 if _u._JOURNAL_MODE else 10
            fig.legend(handles=legend_elements, loc='lower center',
                       ncol=3, fontsize=_leg_fs, framealpha=0.95, edgecolor='0.8',
                       bbox_to_anchor=(0.5, -0.01))
            fig.subplots_adjust(bottom=0.1)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            plt.close()
        else:
            plt.show()
    
    # Return summary statistics
    # Only count constrained (valenced) positions for accuracy stats
    total_correct = np.sum((correctness_matrix == 1) & (constrained_matrix == 1))
    total_incorrect = np.sum((correctness_matrix == 0) & (constrained_matrix == 1))
    total_valid = total_correct + total_incorrect
    accuracy = total_correct / total_valid if total_valid > 0 else 0
    return accuracy, total_correct, total_incorrect


def plot_generated_sequences_heatmap_before_after(generated_sequences_e0, generated_sequences_final, generator, save_path=None, num_sequences=3, max_length=20):
    """Plot before/after generated sequences heatmaps in one image."""
    if not generated_sequences_e0:
        return plot_generated_sequences_heatmap(
            generated_sequences_final, generator,
            save_path=save_path, num_sequences=num_sequences, max_length=max_length
        )
    
    # Fewer sequences, larger panels
    if _u._JOURNAL_MODE:
        fig_height = 6.0
        fig_width = 7.0
    else:
        fig_height = max(10, num_sequences * 1.6 + 5)
        fig_width = max(16, max_length * 0.65)
    fig, axes = plt.subplots(2, 1, figsize=(fig_width, fig_height))
    
    acc0, c0, i0 = plot_generated_sequences_heatmap(
        generated_sequences_e0, generator,
        save_path=None, num_sequences=num_sequences, max_length=max_length,
        ax=axes[0], title=f"E0 Generated Sequences (n={num_sequences})", show_legend=False
    )
    accf, cf, inf = plot_generated_sequences_heatmap(
        generated_sequences_final, generator,
        save_path=None, num_sequences=num_sequences, max_length=max_length,
        ax=axes[1], title=f"Final Generated Sequences (n={num_sequences})", show_legend=False
    )
    
    _label_panels([axes[0], axes[1]])

    legend_elements = [
        Patch(facecolor='#90EE90', label='Correct'),
        Patch(facecolor='#ff6b6b', label='Incorrect'),
        Patch(facecolor='#d3d3d3', label='Neutral'),
    ]
    _leg_fs = 8 if _u._JOURNAL_MODE else 10
    plt.tight_layout()
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=3, fontsize=_leg_fs, framealpha=0.95, edgecolor='0.8',
               bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(bottom=0.08)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()
    
    return (acc0, c0, i0), (accf, cf, inf)
