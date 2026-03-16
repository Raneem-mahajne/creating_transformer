"""Plotting: _get_v_before_after_for_sequence, plot_v_before_after_demo_sequences."""
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
def _get_v_before_after_for_sequence(model, idx):
    """
    Run first attention head on sequence idx (1, T), return V_before and V_after as numpy (T, head_size).
    """
    model.eval()
    with torch.no_grad():
        B, T = idx.shape
        token_emb = model.token_embedding(idx)
        positions = torch.arange(T, device=idx.device) % model.block_size
        pos_emb = model.position_embedding_table(positions)
        x = (token_emb + pos_emb)  # (1, T, n_embd)
        head = model.sa_heads.heads[0]
        q = head.query(x)
        k = head.key(x)
        v = head.value(x)  # V_before
        weight = q @ k.transpose(-2, -1) / (head.head_size ** 0.5)
        weight = weight.masked_fill(head.tril[:T, :T].to(x.device) == 0, float("-inf"))
        weight = F.softmax(weight, dim=-1)
        out = weight @ v  # V_after
        v_np = v[0].cpu().numpy()   # (T, head_size)
        out_np = out[0].cpu().numpy()  # (T, head_size)
        x_np = x[0].cpu().numpy()     # (T, n_embd)
    return x_np, v_np, out_np


@torch.no_grad()
def plot_v_before_after_demo_sequences(model, itos, sequences, save_dir=None, arrow_scale=0.15, grid_resolution=60):
    """
    For each demo sequence, create one figure. Each figure has one panel per output token.
    On each panel: 2D embedding-space grid; at each position where the NEXT token equals
    that panel's token, draw an arrow from (position + scale*V_before) to (position + scale*V_after).

    Args:
        model: Trained model (n_embd=2, head_size=2).
        itos: Index-to-string for tokens.
        sequences: List of sequences (each sequence = list of token ids, length <= block_size).
        save_dir: Directory to save figures (one file per sequence, e.g. v_before_after_demo_0.png).
        arrow_scale: Scale factor for V vectors so arrows fit on grid (default 0.15).
        grid_resolution: Grid resolution for background heatmap (default 60).
    """
    model.eval()
    vocab_size = model.token_embedding.weight.shape[0]
    block_size = model.block_size
    n_embd = model.lm_head.in_features
    if n_embd != 2:
        print(f"plot_v_before_after_demo_sequences: n_embd={n_embd}, need 2. Skipping.")
        return

    # Embedding extent and P(token) grid for background (same as lm_head heatmaps)
    with torch.no_grad():
        token_emb = model.token_embedding.weight.detach().cpu().numpy()
        pos_emb = model.position_embedding_table.weight.detach().cpu().numpy()
        combined = token_emb[:, None, :] + pos_emb[None, :, :]
        flat = combined.reshape(-1, 2)
        x_min, x_max = flat[:, 0].min() - 0.5, flat[:, 0].max() + 0.5
        y_min, y_max = flat[:, 1].min() - 0.5, flat[:, 1].max() + 0.5
        W = model.lm_head.weight.detach().cpu().numpy()
        b = model.lm_head.bias.detach().cpu().numpy()

    xs = np.linspace(x_min, x_max, grid_resolution)
    ys = np.linspace(y_min, y_max, grid_resolution)
    xx, yy = np.meshgrid(xs, ys)
    points = np.stack([xx.ravel(), yy.ravel()], axis=1)
    logits = points @ W.T + b
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)  # (N, vocab_size)
    argmax_token = logits.argmax(axis=1)  # (N,) which token has highest logit at each point

    n_data_rows = 3   # V values, Embed with arrows to Final, Final position
    n_cols = vocab_size  # one column per predicted digit
    n_rows = n_data_rows

    arrow_data_list = []  # for big arrow-only figure: (x_np, v_before, v_after, correct, seq, plot_x_min, plot_x_max, plot_y_min, plot_y_max, T)

    for seq_idx, seq in enumerate(sequences):
        if len(seq) < 2:
            continue
        seq = seq[:block_size]
        T = len(seq)
        dev = next(model.parameters()).device
        idx = torch.tensor([seq], dtype=torch.long, device=dev)
        with torch.no_grad():
            logits, _ = model(idx)
            pred_next = logits[0].argmax(dim=-1).cpu().numpy()
        correct = np.array([i < T - 1 and pred_next[i] == seq[i + 1] for i in range(T)])
        with torch.no_grad():
            g_seed = 42 + seq_idx
            torch.manual_seed(g_seed)
            start = torch.tensor([[seq[0]]], dtype=torch.long, device=dev)
            generated = model.generate(start, max_new_tokens=T - 1)[0].tolist()
        x_np, v_before, v_after = _get_v_before_after_for_sequence(model, idx)
        land_x = x_np[:, 0] + v_after[:, 0]
        land_y = x_np[:, 1] + v_after[:, 1]
        margin = 0.6
        # Include V values (v_before) in plot limits
        plot_x_min = min(x_min, x_np[:, 0].min(), v_before[:, 0].min(), v_after[:, 0].min(), land_x.min()) - margin
        plot_x_max = max(x_max, x_np[:, 0].max(), v_before[:, 0].max(), v_after[:, 0].max(), land_x.max()) + margin
        plot_y_min = min(y_min, x_np[:, 1].min(), v_before[:, 1].min(), v_after[:, 1].min(), land_y.min()) - margin
        plot_y_max = max(y_max, x_np[:, 1].max(), v_before[:, 1].max(), v_after[:, 1].max(), land_y.max()) + margin
        arrow_data_list.append((x_np.copy(), v_before.copy(), v_after.copy(), correct.copy(), seq[:], plot_x_min, plot_x_max, plot_y_min, plot_y_max, T))
        xs_plot = np.linspace(plot_x_min, plot_x_max, grid_resolution)
        ys_plot = np.linspace(plot_y_min, plot_y_max, grid_resolution)
        xx_plot, yy_plot = np.meshgrid(xs_plot, ys_plot)
        points_plot = np.stack([xx_plot.ravel(), yy_plot.ravel()], axis=1)
        dev = next(model.parameters()).device
        with torch.no_grad():
            pts = torch.tensor(points_plot, dtype=torch.float32, device=dev)
            h = pts + model.ffwd(pts)
            logits_plot = model.lm_head(h).cpu().numpy()
        probs_plot = np.exp(logits_plot - logits_plot.max(axis=1, keepdims=True))
        probs_plot /= probs_plot.sum(axis=1, keepdims=True)

        # Increase figure size and resolution for better quality
        figsize_mult = 1.5 if seq_idx == 0 else 1.0  # Larger for first demo (plot 17)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols * figsize_mult, 2.5 * n_rows * figsize_mult), sharex=True, sharey=True)
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        # Increase font sizes for seq_idx==0 (plot 15) and seq_idx==2 (plot 17)
        title_fs = 14 if (seq_idx == 0 or seq_idx == 2) else 10
        label_fs = 11 if (seq_idx == 0 or seq_idx == 2) else 9
        fs = 7.0 if (seq_idx == 0 or seq_idx == 2) else 4.5  # Font size for point labels
        
        for r in range(n_rows):
            for c in range(n_cols):
                ax = axes[r, c]
                ax.set_xlim(plot_x_min, plot_x_max)
                ax.set_ylim(plot_y_min, plot_y_max)
                ax.set_aspect('equal')
                Z = probs_plot[:, c].reshape(grid_resolution, grid_resolution)
                ax.pcolormesh(xx_plot, yy_plot, Z, cmap='viridis', vmin=0, vmax=1, shading='auto')
                # Add faint dashed lines to indicate origin (0,0)
                ax.axhline(y=0, color='white', linestyle='--', linewidth=0.5, alpha=0.3, zorder=3)
                ax.axvline(x=0, color='white', linestyle='--', linewidth=0.5, alpha=0.3, zorder=3)
                if c == 0:
                    ax.set_ylabel("dim 1", fontsize=label_fs)
                if r == 0:
                    ax.set_title(f"P(next = {itos[c]})", fontsize=title_fs)
                if r == n_rows - 1:
                    ax.set_xlabel("dim 0", fontsize=label_fs)
                # Increase tick label font sizes for plots 15 and 17
                ax.tick_params(axis='both', which='major', labelsize=label_fs)
        
        # Row 0: V values for the specific sequence
        for i in range(T):
            lbl = _token_pos_label(itos[seq[i]], i)
            v_pos = v_before[i]  # V value for this token-position pair
            for c in range(n_cols):
                ax = axes[0, c]
                ax.text(v_pos[0], v_pos[1], lbl, fontsize=fs, fontweight='bold', ha='center', va='center', color='white', zorder=5,
                        path_effects=[pe.withStroke(linewidth=0.8, foreground='black')])
        
        # Row 1: Embed with arrows to Final (x_np with arrows to x_np + v_after)
        for i in range(T):
            lbl = _token_pos_label(itos[seq[i]], i)
            px0, py0 = x_np[i, 0], x_np[i, 1]  # Embed position
            px_final, py_final = x_np[i, 0] + v_after[i, 0], x_np[i, 1] + v_after[i, 1]  # Final position
            for c in range(n_cols):
                ax = axes[1, c]
                # Draw arrow from embed to final (skip arrows for seq_idx==1, but keep labels)
                if seq_idx != 1:  # Only draw arrows if NOT seq_idx==1 (plot 16)
                    dx = px_final - px0
                    dy = py_final - py0
                    arrow_length = np.sqrt(dx**2 + dy**2)
                    head_size = max(0.1, min(0.2, arrow_length * 0.1))
                    ax.arrow(px0, py0, dx, dy,
                            head_width=head_size, head_length=head_size, fc='white', ec='white', alpha=0.8, length_includes_head=True, width=0.01, zorder=4)
                # Annotate at beginning (embed position) - keep this for all sequences
                ax.text(px0, py0, lbl, fontsize=fs, fontweight='bold', ha='center', va='center', color='white', zorder=5,
                        path_effects=[pe.withStroke(linewidth=0.8, foreground='black')])
        
        # Row 2: Final position; white text, stroke color = correct (green) / wrong (red) - keep this for all sequences
        for i in range(T):
            lbl = _token_pos_label(itos[seq[i]], i)
            px_final, py_final = x_np[i, 0] + v_after[i, 0], x_np[i, 1] + v_after[i, 1]  # Final position
            end_color = '#2E7D32' if correct[i] else '#C62828'
            for c in range(n_cols):
                ax = axes[2, c]
                ax.text(px_final, py_final, lbl, fontsize=fs, fontweight='bold', ha='center', va='center', color='white', zorder=5,
                        path_effects=[pe.withStroke(linewidth=0.8, foreground=end_color)])

        # Row labels
        axes[0, 0].set_ylabel("V values\ndim 1", fontsize=label_fs)
        axes[1, 0].set_ylabel("Embed → Final\ndim 1", fontsize=label_fs)
        axes[2, 0].set_ylabel("Final\ndim 1", fontsize=label_fs)

        seq_str = " ".join(str(itos[t]) for t in seq[:25])
        if len(seq) > 25:
            seq_str += "..."
        gen_str = " ".join(str(itos[generated[i]]) for i in range(min(T, 25)))
        if T > 25:
            gen_str += "..."
        n_correct = correct.sum()
        print(f"Demo {seq_idx}  Sequence: {' '.join(str(itos[t]) for t in seq)}")
        print(f"Demo {seq_idx}  Generated (sampled): {' '.join(str(itos[generated[i]]) for i in range(T))}")
        print(f"Demo {seq_idx}  Pred next (argmax at each pos): {' '.join(str(itos[pred_next[i]]) for i in range(T))}")
        # Only add title for seq_idx != 1 (plot 16 is seq_idx=1, remove title)
        # For seq_idx==0 (plot 15) and seq_idx==2 (plot 17), remove only the "Generated: {gen_str}" part
        title_fontsize = 13 if (seq_idx == 0 or seq_idx == 2) else 9  # Bigger title font for plots 15 and 17
        if seq_idx != 1:
            if seq_idx == 0 or seq_idx == 2:
                # Remove only the generated sequence part
                fig.suptitle(f"Demo {seq_idx}: {seq_str}  |  Correct: {n_correct}/{T-1} (row 2: green=correct, red=wrong)", fontsize=title_fontsize, fontweight='bold', y=1.01)
            else:
                # Keep full title for other sequences (if any)
                fig.suptitle(f"Demo {seq_idx}: {seq_str}  |  Generated: {gen_str}  |  Correct: {n_correct}/{T-1} (row 2: green=correct, red=wrong)", fontsize=title_fontsize, fontweight='bold', y=1.01)
        plt.tight_layout()
        if save_dir:
            path = os.path.join(save_dir, f"v_before_after_demo_{seq_idx}.png")
            # Higher resolution for seq_idx=0 (plot 17)
            dpi_val = 300 if seq_idx == 0 else 150
            plt.savefig(path, bbox_inches='tight', dpi=dpi_val, facecolor='white')
            plt.close()
            print(f"V before/after demo figure saved to {path}")
        else:
            plt.show()

    if save_dir and sequences:
        print(f"Saved {min(len(sequences), len([s for s in sequences if len(s) >= 2]))} demo sequence figures to {save_dir}")
