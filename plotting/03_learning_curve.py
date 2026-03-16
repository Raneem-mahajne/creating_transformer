"""Plotting: plot_learning_curve, estimate_loss, estimate_rule_error."""
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


def plot_learning_curve(steps, train_losses, val_losses, rule_error_history=None, save_path=None, eval_interval=None):
    """
    Plot learning curve with loss and optionally rule error on the same figure.
    Uses dual y-axes: left for loss, right for error (both should decrease).
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Left y-axis: Loss
    color1 = 'tab:blue'
    ax1.set_xlabel(f"Training Steps{' (evaluated every ' + str(eval_interval) + ' steps)' if eval_interval else ''}", fontsize=11)
    ax1.set_ylabel("Cross-entropy loss", color=color1, fontsize=11)
    line1 = ax1.plot(steps, train_losses, label="Training Loss", color='tab:blue', linewidth=2)
    line2 = ax1.plot(steps, val_losses, label="Validation Loss", color='tab:orange', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    
    # Right y-axis: Rule Error (if provided)
    if rule_error_history is not None:
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel("Rule Error (fraction of constrained positions wrong)", color=color2, fontsize=11)
        line3 = ax2.plot(steps, rule_error_history, label="Rule Error", color=color2, linewidth=2, linestyle='--')
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.set_ylim(0, 1.05)
        
        # Combine legends
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='best')
        
        plt.title("Learning Curve: Loss and Rule Error During Training", fontsize=13)
    else:
        ax1.legend(loc='best')
        plt.title("Learning Curve: Training vs Validation Loss", fontsize=13)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()



# estimate_loss and estimate_rule_error live in plotting._utils
