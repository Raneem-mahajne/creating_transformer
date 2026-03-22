"""Plotting: 2x2 output landscape summary (argmax, entropy, argmax+embeddings, argmax+values)."""
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from scipy import ndimage

import plotting._utils as _u
from plotting._utils import (
    _constrain_figsize, _update_font_scale_for_figure,
    _SUBSCRIPT_DIGITS, _pos_subscript, _token_pos_label,
)


def _build_argmax_cmap(itos, vocab_size):
    """Build a categorical colormap for the argmax plot, one color per token."""
    base_cmap = plt.cm.get_cmap("tab20", vocab_size)
    colors = [base_cmap(i) for i in range(vocab_size)]
    cmap = mcolors.ListedColormap(colors)
    bounds = np.arange(-0.5, vocab_size + 0.5, 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm


def _plot_argmax(ax, xx, yy, probs, itos, cmap, norm, grid_resolution, extent,
                 annotate_regions=False, min_region_pixels=60):
    """Plot the argmax map on the given axes. Optionally annotate large regions with their token."""
    argmax = probs.argmax(axis=1).reshape(grid_resolution, grid_resolution)
    ax.pcolormesh(xx, yy, argmax, cmap=cmap, norm=norm, shading="auto")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_title("Argmax prediction", fontsize=10)

    if annotate_regions:
        vocab_size = len(itos)
        dx = (extent[1] - extent[0]) / grid_resolution
        dy = (extent[3] - extent[2]) / grid_resolution
        for token_id in range(vocab_size):
            mask = (argmax == token_id)
            labeled, n_regions = ndimage.label(mask)
            for region_id in range(1, n_regions + 1):
                region_mask = (labeled == region_id)
                n_pixels = int(region_mask.sum())
                if n_pixels < min_region_pixels:
                    continue
                cy, cx = ndimage.center_of_mass(region_mask)
                x_cent = extent[0] + (cx + 0.5) * dx
                y_cent = extent[2] + (cy + 0.5) * dy
                label = str(itos[token_id])
                ax.text(
                    x_cent, y_cent, label, fontsize=13, ha="center", va="center",
                    color="white", weight="bold", zorder=6,
                    path_effects=[pe.withStroke(linewidth=2, foreground="black")],
                )


def _plot_entropy(ax, xx, yy, probs, grid_resolution, extent):
    """Plot the entropy map on the given axes."""
    eps = 1e-12
    entropy = -np.sum(probs * np.log(probs + eps), axis=1)
    entropy = entropy.reshape(grid_resolution, grid_resolution)
    im = ax.pcolormesh(xx, yy, entropy, cmap="inferno", shading="auto")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_title("Prediction entropy", fontsize=10)
    return im


def _overlay_labels(ax, points, labels, color="white"):
    """Overlay text labels on the axes."""
    fs = 10 if _u._JOURNAL_MODE else 10
    for pt, label in zip(points, labels):
        ax.text(
            pt[0], pt[1], label, fontsize=fs, ha="center", va="center",
            color=color, weight="bold", zorder=6,
            path_effects=[pe.withStroke(linewidth=1.5, foreground="black")],
        )


@torch.no_grad()
def plot_output_landscape_summary(
    model, itos, save_path=None, grid_resolution=80, extent_margin=0.5,
):
    """
    2x2 summary figure:
      (a) argmax prediction map
      (b) entropy map
      (c) argmax map + combined-embedding overlay
      (d) argmax map + value-vector overlay
    """
    model.eval()
    vocab_size = model.token_embedding.weight.shape[0]
    block_size = model.position_embedding_table.weight.shape[0]
    n_embd = model.lm_head.in_features
    if n_embd != 2:
        print(f"plot_output_landscape_summary: n_embd={n_embd}, need 2. Skipping.")
        return

    token_emb = model.token_embedding.weight.detach().cpu().numpy()
    pos_emb = model.position_embedding_table.weight.detach().cpu().numpy()
    head = model.sa_heads.heads[0]
    W_V = head.value.weight.detach().cpu().numpy()

    emb_pts, emb_labels = [], []
    val_pts, val_labels = [], []
    for ti in range(vocab_size):
        for pi in range(block_size):
            e = token_emb[ti] + pos_emb[pi]
            emb_pts.append(e)
            emb_labels.append(_token_pos_label(itos[ti], pi))
            val_pts.append(W_V @ e)
            val_labels.append(_token_pos_label(itos[ti], pi))
    emb_pts = np.array(emb_pts)
    val_pts = np.array(val_pts)

    all_overlay = np.concatenate([emb_pts, val_pts], axis=0)
    combined = token_emb[:, None, :] + pos_emb[None, :, :]
    flat = combined.reshape(-1, 2)
    all_pts = np.concatenate([flat, all_overlay], axis=0)
    x_min = all_pts[:, 0].min() - extent_margin
    x_max = all_pts[:, 0].max() + extent_margin
    y_min = all_pts[:, 1].min() - extent_margin
    y_max = all_pts[:, 1].max() + extent_margin
    x_c, y_c = (x_min + x_max) / 2, (y_min + y_max) / 2
    half = max(x_max - x_min, y_max - y_min) / 2
    x_min, x_max = x_c - half, x_c + half
    y_min, y_max = y_c - half, y_c + half

    xs = np.linspace(x_min, x_max, grid_resolution)
    ys = np.linspace(y_min, y_max, grid_resolution)
    xx, yy = np.meshgrid(xs, ys)
    points = np.stack([xx.ravel(), yy.ravel()], axis=1)
    dev = next(model.parameters()).device
    with torch.no_grad():
        pts_t = torch.tensor(points, dtype=torch.float32, device=dev)
        h = pts_t + model.ffwd(pts_t)
        logits = model.lm_head(h).cpu().numpy()
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    extent = (x_min, x_max, y_min, y_max)

    cmap, norm = _build_argmax_cmap(itos, vocab_size)

    if _u._JOURNAL_MODE:
        fig, axes = plt.subplots(2, 2, figsize=(7.0, 7.0))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # (a) argmax with region annotations for large regions
    _plot_argmax(axes[0, 0], xx, yy, probs, itos, cmap, norm, grid_resolution, extent,
                 annotate_regions=True, min_region_pixels=60)

    # (b) entropy
    ent_im = _plot_entropy(axes[0, 1], xx, yy, probs, grid_resolution, extent)
    cbar = fig.colorbar(ent_im, ax=axes[0, 1], fraction=0.046, pad=0.04)
    cbar.set_label("nats", fontsize=8)

    # (c) argmax + embeddings
    _plot_argmax(axes[1, 0], xx, yy, probs, itos, cmap, norm, grid_resolution, extent)
    axes[1, 0].set_title("Argmax + embeddings $\\mathbf{e}_i$", fontsize=10)
    _overlay_labels(axes[1, 0], emb_pts, emb_labels)

    # (d) argmax + values
    _plot_argmax(axes[1, 1], xx, yy, probs, itos, cmap, norm, grid_resolution, extent)
    axes[1, 1].set_title("Argmax + values $W_V \\mathbf{e}_i$", fontsize=10)
    _overlay_labels(axes[1, 1], val_pts, val_labels)

    # shared legend for argmax colors
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=cmap(norm(i)), label=str(itos[i])) for i in range(vocab_size)
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=min(vocab_size, 6),
        fontsize=8, frameon=True, title="Predicted token",
    )

    for r in range(2):
        for c in range(2):
            axes[r, c].set_xlabel("embedding dim 0", fontsize=9)
            axes[r, c].set_ylabel("embedding dim 1", fontsize=9)

    # Panel labels outside the panels (to the left of each panel)
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]
    for idx, (r, c) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        axes[r, c].text(
            -0.18, 1.0, panel_labels[idx], transform=axes[r, c].transAxes,
            fontsize=12, fontweight="bold", va="top", ha="right",
        )

    plt.tight_layout(rect=[0.04, 0.06, 1, 1])  # extra left margin for panel labels
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150, facecolor="white")
        plt.close()
        print(f"Output landscape summary (2×2) saved to {save_path}")
    else:
        plt.show()

    model.train()
