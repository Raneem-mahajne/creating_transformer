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
import matplotlib.colors as mcolors
from matplotlib.ticker import NullLocator
import seaborn as sns

from data import get_batch_from_sequences

import plotting._utils as _u

_mod_18_final_on_output_grid = __import__(
    "plotting.18_final_on_output_grid", fromlist=["compute_second_residual_next_token_prob_grid"]
)
compute_second_residual_next_token_prob_grid = (
    _mod_18_final_on_output_grid.compute_second_residual_next_token_prob_grid
)
from plotting._utils import (
    _constrain_figsize, _update_font_scale_for_figure,
    set_journal_mode, clear_journal_mode,
    _label_panels, annotate_sequence, sparse_ticks,
    _SUBSCRIPT_DIGITS, _pos_subscript, _token_pos_label, _pos_only_label,
    collect_epoch_stats, get_attention_snapshot_from_X, get_multihead_snapshot_from_X,
    estimate_loss, estimate_rule_error,
)
import matplotlib.patheffects as _pe


_FLYER_BOX = dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='0.25', linewidth=0.5)


def _segments_cross(p1, p2, q1, q2):
    """True if segments p1-p2 and q1-q2 properly intersect (display coordinates)."""
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    d1, d2 = orient(q1, q2, p1), orient(q1, q2, p2)
    d3, d4 = orient(p1, p2, q1), orient(p1, p2, q2)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


def _segment_hits_rect(p1, p2, rect):
    x0, y0, x1, y1 = rect
    for p in (p1, p2):
        if x0 <= p[0] <= x1 and y0 <= p[1] <= y1:
            return True
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return any(_segments_cross(p1, p2, corners[k], corners[(k + 1) % 4]) for k in range(4))


def _point_seg_dist(p, a, b):
    p, a, b = (np.asarray(v, float) for v in (p, a, b))
    ab = b - a
    n2 = float(ab @ ab)
    t = 0.0 if n2 == 0 else float(np.clip((p - a) @ ab / n2, 0, 1))
    return float(np.linalg.norm(p - (a + t * ab)))


def _seg_seg_dist(p1, p2, q1, q2):
    if _segments_cross(p1, p2, q1, q2):
        return 0.0
    return min(_point_seg_dist(p1, q1, q2), _point_seg_dist(p2, q1, q2),
               _point_seg_dist(q1, p1, p2), _point_seg_dist(q2, p1, p2))


def _point_rect_dist(p, rect):
    dx = max(rect[0] - p[0], 0, p[0] - rect[2])
    dy = max(rect[1] - p[1], 0, p[1] - rect[3])
    return float(np.hypot(dx, dy))


def _grow(rect, m):
    return (rect[0] - m, rect[1] - m, rect[2] + m, rect[3] + m)


def _rects_overlap(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _group_coincident(XY, tol=0.15):
    groups, used = [], [False] * len(XY)
    for i in range(len(XY)):
        if used[i]:
            continue
        g = [i]
        used[i] = True
        for j in range(i + 1, len(XY)):
            if not used[j] and np.hypot(*(XY[i] - XY[j])) < tol:
                g.append(j)
                used[j] = True
        groups.append(g)
    return groups


def _draw_clean_arrow(ax, start, end, lw=1.1):
    """Plain black arrow (no outline)."""
    ax.annotate(
        '', xy=end, xytext=start, zorder=5,
        arrowprops=dict(arrowstyle='-|>,head_length=0.4,head_width=0.2', color='black',
                        lw=lw, shrinkA=0, shrinkB=0),
    )


def _place_flyers(ax, anchors, texts, avoid_segments=(), fontsize=8,
                  radii_pt=(5, 8, 12, 17, 24, 34, 46, 60), n_dir=24, length_weight=7.0,
                  soft_scale=1.0, tip_pref=0.0):
    """Call-out labels placed so leaders and boxes avoid the given segments, points and each other.

    Must be called after the figure layout is final. Candidate positions (directions x radii
    around each anchor) are scored; crossings with arrows/labels are heavily penalised and
    shorter leaders are preferred.
    tip_pref: when an anchor is an arrow's tip, extra cost for label directions that deviate
        from the arrow's own direction, so the label sits just past the arrowhead unless that
        spot is blocked.
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_px = ax.transData.transform
    to_data = ax.transData.inverted().transform
    px_per_pt = fig.dpi / 72.0

    A = to_px(np.asarray(anchors, dtype=float))
    segs = [(to_px(np.asarray(s, float)), to_px(np.asarray(e, float))) for s, e in avoid_segments]
    axb = ax.get_window_extent(renderer)
    inner = (axb.x0 + 2, axb.y0 + 2, axb.x1 - 2, axb.y1 - 2)
    pad = 0.25 * fontsize * px_per_pt + 1.5
    clear = 3.0 * px_per_pt
    soft = 12.0 * px_per_pt

    sizes = []
    for t in texts:
        tmp = ax.text(0, 0, t, fontsize=fontsize)
        bb = tmp.get_window_extent(renderer)
        tmp.remove()
        sizes.append((bb.width + 2 * pad, bb.height + 2 * pad))

    placed_rects, placed_leaders = [], []
    order = sorted(range(len(A)), key=lambda k: -sizes[k][0])
    choice = {}
    for k in order:
        ax_, ay_ = A[k]
        w, h = sizes[k]
        if tip_pref:
            # First choice: box just past the arrowhead, on the arrow's own line, with the
            # leader touching the tip. Used unless it hits another arrow, label, point or edge.
            tipdir = None
            for q0, q1 in segs:
                if np.hypot(q1[0] - ax_, q1[1] - ay_) < 6:
                    v = np.array((ax_, ay_)) - np.array(q0)
                    n = np.linalg.norm(v)
                    if n > 0:
                        tipdir = v / n
                        break
            placed_here = False
            if tipdir is not None:
                # Straight past the tip first, then angled away from the arrow's line, then
                # beside the arrowhead; the leader always starts at the tip.
                for off_deg in (0, 30, -30, 60, -60, 90, -90):
                    a = np.radians(off_deg)
                    c = tipdir[0] * np.cos(a) - tipdir[1] * np.sin(a)
                    s = tipdir[0] * np.sin(a) + tipdir[1] * np.cos(a)
                    R = 2 * px_per_pt
                    ex, ey = ax_ + R * c, ay_ + R * s
                    t = min(w / 2 / max(abs(c), 1e-9), h / 2 / max(abs(s), 1e-9))
                    cx, cy = ex + c * t, ey + s * t
                    rect = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
                    ok = not (rect[0] < inner[0] or rect[1] < inner[1]
                              or rect[2] > inner[2] or rect[3] > inner[3])
                    for q0, q1 in segs if ok else ():
                        own = np.hypot(q1[0] - ax_, q1[1] - ay_) < 6
                        if own:
                            if _segment_hits_rect(q0, q1, rect):
                                ok = False
                        elif (_segment_hits_rect(q0, q1, _grow(rect, clear))
                              or _point_rect_dist(q1, rect) < 2 * clear):
                            ok = False
                        if not ok:
                            break
                    if ok and any(_rects_overlap(rect, _grow(r2, 3)) for r2 in placed_rects):
                        ok = False
                    if ok and any(_segment_hits_rect(l0, l1, _grow(rect, 2))
                                  for l0, l1 in placed_leaders):
                        ok = False
                    if ok and any(_point_rect_dist((bx, by), rect) < 4
                                  for j, (bx, by) in enumerate(A) if j != k):
                        ok = False
                    if ok:
                        placed_rects.append(rect)
                        placed_leaders.append(((ax_, ay_), (ex, ey)))
                        choice[k] = (cx, cy)
                        placed_here = True
                        break
            if placed_here:
                continue
        best = None
        for R_pt in radii_pt:
            R = R_pt * px_per_pt
            for d in range(n_dir):
                th = 2 * np.pi * d / n_dir
                c, s = np.cos(th), np.sin(th)
                ex, ey = ax_ + R * c, ay_ + R * s          # leader end = box edge point
                cx = ex + (w / 2) * np.sign(c) * (abs(c) > 0.38)
                cy = ey + (h / 2) * np.sign(s) * (abs(s) > 0.38)
                rect = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
                if rect[0] < inner[0] or rect[1] < inner[1] or rect[2] > inner[2] or rect[3] > inner[3]:
                    continue
                p0 = (ax_ + 4 * c, ay_ + 4 * s)
                p1 = (ex, ey)
                # Short leaders keep each label close to its point/arrow.
                score = length_weight * R
                for q0, q1 in segs:
                    at_tip = np.hypot(q1[0] - ax_, q1[1] - ay_) < 6
                    own = np.hypot(q0[0] - ax_, q0[1] - ay_) < 2 or at_tip
                    if own:
                        # Leader must not run back along an arrow that starts/ends at this anchor.
                        far = q0 if at_tip else q1
                        v = np.array(far) - np.array((ax_, ay_))
                        n = np.linalg.norm(v)
                        if n > 0 and np.dot(v / n, (c, s)) > np.cos(np.radians(35)):
                            score += 500
                        if at_tip and n > 0 and tip_pref:
                            # Prefer continuing straight past the arrowhead (direction -v).
                            score += tip_pref * (1 + np.dot(v / n, (c, s))) / 2
                    elif _seg_seg_dist(p0, p1, q0, q1) < clear:
                        score += 1000
                    if _segment_hits_rect(q0, q1, _grow(rect, clear)):
                        score += 800
                    # Arrowheads are wider than the shaft. A label may sit just beside its own
                    # arrowhead (tighter margin) but must keep clear of other arrows' heads.
                    if _point_rect_dist(q1, rect) < (0.8 * clear if own else 2 * clear):
                        score += 800
                for r2 in placed_rects:
                    if _rects_overlap(rect, _grow(r2, 3)):
                        score += 5000
                    if _segment_hits_rect(p0, p1, _grow(r2, 2)):
                        score += 2000
                for l0, l1 in placed_leaders:
                    if _seg_seg_dist(p0, p1, l0, l1) < clear:
                        score += 2000
                    if _segment_hits_rect(l0, l1, _grow(rect, 2)):
                        score += 2000
                own_d = _point_rect_dist((ax_, ay_), rect)
                for j, (bx, by) in enumerate(A):
                    if j == k:
                        continue
                    d_other = _point_rect_dist((bx, by), rect)
                    if d_other < 4:
                        score += 600
                    # The box must read as belonging to its own point, not a neighbour.
                    if d_other < own_d + 4:
                        score += 400
                    if _point_seg_dist((bx, by), p0, p1) < clear + 2:
                        score += 300
                # Soft terms: prefer open space and labels sitting beside (not above/below) the point.
                score += 25 * soft_scale * abs(s)
                for j, (bx, by) in enumerate(A):
                    if j != k:
                        score += 30 * soft_scale * max(0.0, 1 - _point_rect_dist((bx, by), rect) / soft)
                for q0, q1 in segs:
                    dseg = 0.0 if _segment_hits_rect(q0, q1, rect) else min(
                        [_point_seg_dist(corner, q0, q1)
                         for corner in ((rect[0], rect[1]), (rect[2], rect[1]),
                                        (rect[2], rect[3]), (rect[0], rect[3]))]
                        + [_point_rect_dist(q0, rect), _point_rect_dist(q1, rect)]
                    )
                    score += 20 * soft_scale * max(0.0, 1 - dseg / soft)
                for r2 in placed_rects:
                    gap = max(r2[0] - rect[2], rect[0] - r2[2], r2[1] - rect[3], rect[1] - r2[3], 0)
                    score += 20 * soft_scale * max(0.0, 1 - gap / soft)
                if best is None or score < best[0]:
                    best = (score, rect, p0, p1, (cx, cy))
        if best is None:
            best = (0, (ax_, ay_, ax_, ay_), (ax_, ay_), (ax_, ay_), (ax_ + 20, ay_ + 20))
        placed_rects.append(best[1])
        placed_leaders.append((best[2], best[3]))
        choice[k] = best[4]

    for k in range(len(A)):
        tx, ty = to_data(choice[k])
        ax.annotate(
            texts[k], xy=tuple(anchors[k]), xytext=(tx, ty), textcoords='data',
            ha='center', va='center', fontsize=fontsize, color='black', zorder=7,
            bbox=_FLYER_BOX,
            arrowprops=dict(arrowstyle='-', color='0.2', lw=0.6, shrinkA=0, shrinkB=0),
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
            gs = GridSpec(n_rows_r, num_cols, figure=fig, hspace=0.22, wspace=0.38)
        else:
            fig = plt.figure(figsize=(4 * num_cols, 5 * n_rows_r))
            gs = GridSpec(n_rows_r, num_cols, figure=fig, hspace=0.22, wspace=0.22)
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
    padding = 0.02
    xlim_shared = (x_min_shared - padding * x_range_shared, x_max_shared + padding * x_range_shared)
    ylim_shared = (y_min_shared - padding * y_range_shared, y_max_shared + padding * y_range_shared)

    prob_grid_bg = None
    if use_two_rows_r and all_data and all_data[0]["embeddings"].shape[1] == 2:
        d0 = all_data[0]
        extra_xy = np.vstack([d0["embeddings"], d0["V_transformed"], d0["Sum"]])
        # Background resolution is controlled by grid_resolution: more points -> smoother
        # categorical boundaries in the argmax next-token underlay.
        bg_grid_resolution = 120 if _u._JOURNAL_MODE else 60
        prob_grid_bg = compute_second_residual_next_token_prob_grid(
            model, grid_resolution=bg_grid_resolution, extent_margin=0.5, extra_xy=extra_xy,
        )
        # Match 2D axis limits to the probability grid so the underlay fills the axes (no empty margins).
        if prob_grid_bg is not None:
            xx_bg, yy_bg, _ = prob_grid_bg
            xlim_shared = (float(xx_bg.min()), float(xx_bg.max()))
            ylim_shared = (float(yy_bg.min()), float(yy_bg.max()))

    show_landscape = prob_grid_bg is not None and use_two_rows_r
    landscape_mappable = [None]

    def _draw_output_landscape(ax):
        if not show_landscape:
            return
        xx_g, yy_g, probs_g = prob_grid_bg
        V = probs_g.shape[2]
        # Discrete regions: color by argmax token (one color per winning next-token class).
        Z_cat = np.argmax(probs_g, axis=2)
        tab = plt.cm.get_cmap("tab20", 20)
        colors = [tab(i % 20) for i in range(V)]
        cat_cmap = mcolors.ListedColormap(colors)
        bounds = np.arange(V + 1) - 0.5
        norm = mcolors.BoundaryNorm(bounds, cat_cmap.N)
        pcm = ax.pcolormesh(
            xx_g, yy_g, Z_cat, cmap=cat_cmap, norm=norm,
            shading="auto", zorder=0, rasterized=True,
        )
        if landscape_mappable[0] is None:
            landscape_mappable[0] = pcm

    def _dim_axis_labels(used_pca):
        if used_pca:
            return "PC1", "PC2"
        return "Dim 0", "Dim 1"
    
    # Second pass: create plots
    _resid_axes = []
    _flyer_jobs = []
    ax_row2_last = None
    ax_col_b_top = None  # top-row axis of column (b); sequence title centered over this
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
            _col_abc = ("(a) ", "(b) ", "(c) ")
        else:
            _col_abc = None
            r0_r = r1_r = r2_r = seq_idx
            c_emb_hm, c_v_hm, c_sum_hm, c_emb_sc, c_v_sc, c_arrow = 0, 1, 2, 3, 4, 5
        
        # Embeddings heatmap (row 0)
        ax = fig.add_subplot(gs[r0_r, c_emb_hm])
        _resid_axes.append(ax)
        dim_str = f"(T×d={T}×{embeddings.shape[1]})"
        sns.heatmap(embeddings, cmap="RdBu_r", center=0, vmin=hm_vmin, vmax=hm_vmax,
                   xticklabels=False, yticklabels=tokens, cbar=False, ax=ax)
        heatmap_axes.append(ax)
        ax.set_xlabel("Embedding Dim", fontsize=10)
        ax.set_ylabel("Tokens" if use_two_rows_r else f"Seq {seq_idx+1}\n{seq_str}\n", fontsize=10)
        _p0 = _col_abc[0] if _col_abc else ""
        _hm_fs = 9 if _u._JOURNAL_MODE else 11
        if _col_abc:
            _body0 = r"$\mathbf{E}$ " + dim_str if _u._JOURNAL_MODE else r"Embeddings $\mathbf{E}$ (Token+Pos) " + dim_str
            ax.set_title(f"{_p0.strip()}\n{_body0}", fontsize=_hm_fs)
        else:
            ax.set_title(_p0 + (r"$\mathbf{E}$ " + dim_str if _u._JOURNAL_MODE else r"Embeddings $\mathbf{E}$ (Token+Pos) " + dim_str), fontsize=_hm_fs)
        # V Transformed heatmap
        ax = fig.add_subplot(gs[r0_r, c_v_hm])
        if use_two_rows_r:
            ax_col_b_top = ax
        _resid_axes.append(ax)
        dim_str = f"(T×d={T}×{V_transformed.shape[1]})"
        sns.heatmap(V_transformed, cmap="RdBu_r", center=0, vmin=hm_vmin, vmax=hm_vmax,
                   xticklabels=False, yticklabels=tokens, cbar=False, ax=ax)
        heatmap_axes.append(ax)
        ax.set_xlabel("Embedding Dim", fontsize=10)
        ax.set_ylabel("Tokens" if use_two_rows_r else f"Seq {seq_idx+1}\n{seq_str}\n", fontsize=10)
        _p1 = _col_abc[1] if _col_abc else ""
        _title_attn_e_i = r"$\mathrm{Attn}(\mathbf{E})$" + f" {dim_str}"
        if _col_abc:
            _body1 = _title_attn_e_i
            ax.set_title(f"{_p1.strip()}\n{_body1}", fontsize=_hm_fs)
        else:
            ax.set_title(f"{_p1}{_title_attn_e_i}", fontsize=_hm_fs)
        
        # Final heatmap (col 2, row 0 when single seq)
        if use_two_rows_r:
            ax = fig.add_subplot(gs[r0_r, c_final_hm])
        else:
            ax = fig.add_subplot(gs[r0_r, c_sum_hm])
        dim_str = f"(T×d={T}×{Sum.shape[1]})"
        sns.heatmap(Sum, cmap="RdBu_r", center=0, vmin=hm_vmin, vmax=hm_vmax,
                   xticklabels=False, yticklabels=tokens, cbar=False, ax=ax)
        heatmap_axes.append(ax)
        ax.set_xlabel("Embedding Dim", fontsize=10)
        ax.set_ylabel("Tokens" if use_two_rows_r else f"Seq {seq_idx+1}\n{seq_str}\n", fontsize=10)
        _p2 = _col_abc[2] if _col_abc else ""
        if _col_abc:
            _body2 = r"$\mathbf{Z} = \mathbf{E} + \mathrm{Attn}(\mathbf{E})$ " + dim_str
            ax.set_title(f"{_p2.strip()}\n{_body2}", fontsize=_hm_fs)
        else:
            ax.set_title(_p2 + r"$\mathbf{Z} = \mathbf{E} + \mathrm{Attn}(\mathbf{E})$ " + dim_str, fontsize=_hm_fs)
        
        # Embeddings scatter (row 1, under Embed heatmap)
        ax = fig.add_subplot(gs[r1_r, c_emb_sc])
        embeddings_2d = pca_2d(embeddings)
        _draw_output_landscape(ax)
        # Mark origin with faint dashed lines
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=5)
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=5)
        # Token labels drawn at the embedding points themselves (no markers or call-outs).
        _labels = [_token_pos_label(t, p) for p, t in enumerate(tokens)]
        for (px_, py_), lbl in zip(embeddings_2d, _labels):
            ax.text(px_, py_, lbl, fontsize=9, fontweight='bold', color='black',
                    ha='center', va='center', zorder=7)
        ax.set_xlim(xlim_shared)
        ax.set_ylim(ylim_shared)
        ax.margins(0)
        used_pca = embeddings.shape[1] > 2
        xl, yl = _dim_axis_labels(used_pca)
        ax.set_xlabel(xl, fontsize=10)
        ax.set_ylabel(yl, fontsize=10)
        title_suffix = " (PCA)" if used_pca else ""
        _sc_fs = 9 if _u._JOURNAL_MODE else 11
        ax.set_title(r"$\mathbf{e}_i$" + title_suffix, fontsize=_sc_fs)
        ax.grid(True, alpha=0.35, zorder=4)
        
        # V Transformed scatter (with arrows from origin to each point)
        ax = fig.add_subplot(gs[r1_r, c_v_sc])
        V_transformed_2d = pca_2d(V_transformed)
        _draw_output_landscape(ax)
        # Mark origin with faint dashed lines
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=5)
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=5)
        # Tokens whose attention outputs coincide share one arrow and one call-out.
        _labels = [_token_pos_label(t, p) for p, t in enumerate(tokens)]
        _groups = _group_coincident(V_transformed_2d)
        _tips = np.array([V_transformed_2d[g].mean(axis=0) for g in _groups])
        for tip in _tips:
            _draw_clean_arrow(ax, (0.0, 0.0), tuple(tip))
        _flyer_jobs.append((
            ax, _tips, [", ".join(_labels[i] for i in g) for g in _groups],
            [((0.0, 0.0), tuple(tip)) for tip in _tips],
            # Tight call-outs: short leaders, weak open-space preference (overlaps still forbidden).
            dict(radii_pt=(2, 3, 5, 8, 12, 17, 24, 34), length_weight=12.0, soft_scale=0.3, tip_pref=350.0),
        ))
        ax.set_xlim(xlim_shared)
        ax.set_ylim(ylim_shared)
        ax.margins(0)
        used_pca = V_transformed.shape[1] > 2
        xl, yl = _dim_axis_labels(used_pca)
        ax.set_xlabel(xl, fontsize=10)
        ax.set_ylabel(yl, fontsize=10)
        title_suffix = " (PCA)" if used_pca else ""
        ax.set_title(r"$\mathrm{Attn}(\mathbf{E})_i$" + title_suffix, fontsize=_sc_fs)
        ax.grid(True, alpha=0.35, zorder=4)
        
        # Embeddings → Final arrows (col 2, row 1 when single seq)
        ax = fig.add_subplot(gs[r1_r, c_final_arrow] if use_two_rows_r else gs[r1_r, c_arrow])
        embeddings_2d = pca_2d(embeddings)
        Final_2d = pca_2d(Sum)
        _draw_output_landscape(ax)
        # Mark origin with faint dashed lines
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=5)
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=5)
        # Draw arrows from embeddings to Final points
        _labels = [_token_pos_label(t, p) for p, t in enumerate(tokens)]
        for i in range(len(tokens)):
            _draw_clean_arrow(ax, tuple(embeddings_2d[i]), tuple(Final_2d[i]))
        ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=14, c='black',
                   edgecolors='white', linewidths=0.6, zorder=6)
        # Call-outs label the arrowheads (z_i), with tight placement as in the Attn panel.
        _flyer_jobs.append((
            ax, Final_2d.copy(), _labels,
            [(tuple(embeddings_2d[i]), tuple(Final_2d[i])) for i in range(len(tokens))],
            dict(radii_pt=(2, 3, 5, 8, 12, 17, 24, 34), length_weight=12.0, soft_scale=0.3, tip_pref=350.0),
        ))
        ax_row2_last = ax
        ax.set_xlim(xlim_shared)
        ax.set_ylim(ylim_shared)
        ax.margins(0)
        used_pca = embeddings.shape[1] > 2
        xl, yl = _dim_axis_labels(used_pca)
        ax.set_xlabel(xl, fontsize=10)
        ax.set_ylabel(yl, fontsize=10)
        title_suffix = " (PCA)" if used_pca else ""
        ax.set_title(r"$\mathbf{e}_i \rightarrow \mathbf{z}_i = \mathbf{e}_i + \mathrm{Attn}(\mathbf{E})_i$" + title_suffix, fontsize=_sc_fs)
        ax.grid(True, alpha=0.35, zorder=4)
    
    top_rect = 0.95
    right_rect = 0.86 if show_landscape else 0.90
    plt.tight_layout(rect=[0, 0, right_rect, top_rect], pad=0.4)
    # Sequence title: centered over column (b) — use bbox of V heatmap after layout
    if use_two_rows_r and all_data:
        seq_str_sub = all_data[0]['seq_str']
        if ax_col_b_top is not None:
            _pos_b = ax_col_b_top.get_position()
            _seq_x = _pos_b.x0 + _pos_b.width / 2
        else:
            _seq_x = right_rect / 2
        if show_landscape:
            fig.text(
                _seq_x, 0.998, f"Sequence: {seq_str_sub}",
                ha="center", va="top", fontsize=10, transform=fig.transFigure,
            )
        else:
            fig.suptitle(
                f"Sequence: {seq_str_sub}", fontsize=10, x=_seq_x, y=0.99, ha="center",
            )
    # Colorbar(s) in right margin
    if heatmap_axes and use_two_rows_r:
        mappable = heatmap_axes[0].collections[0]
        # Span the same vertical extent as the row-1 heatmaps so the bar sits beside them.
        _hm_pos = heatmap_axes[-1].get_position()
        ax_cbar = fig.add_axes([0.91, _hm_pos.y0, 0.015, _hm_pos.height])
        fig.colorbar(mappable, cax=ax_cbar)
    if show_landscape and landscape_mappable[0] is not None and prob_grid_bg is not None:
        if use_two_rows_r and ax_row2_last is not None:
            _r2_pos = ax_row2_last.get_position()
            ax_cbar2 = fig.add_axes([0.91, _r2_pos.y0, 0.015, _r2_pos.height])
        else:
            ax_cbar2 = fig.add_axes([0.91, 0.12, 0.015, 0.30])
        V_cb = prob_grid_bg[2].shape[2]
        tick_idx = np.arange(V_cb)
        cb2 = fig.colorbar(landscape_mappable[0], cax=ax_cbar2, ticks=tick_idx)
        cb2.set_ticklabels([str(itos[i]) for i in range(V_cb)])
        cb2.set_label("argmax next token", fontsize=7 if _u._JOURNAL_MODE else 8)
        # Discrete BoundaryNorm colorbars often get minor ticks between classes; hide them.
        cb2.ax.yaxis.set_minor_locator(NullLocator())
        cb2.ax.minorticks_off()
    for _ax, _anchors, _texts, _segs, *_kw in _flyer_jobs:
        _place_flyers(_ax, _anchors, _texts, _segs, **(_kw[0] if _kw else {}))
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
        ax.set_xlabel("Dim 0", fontsize=9)
        ax.grid(True, alpha=0.15)

    ax1.set_ylabel("Dim 1", fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()
        print(f"FFN + second residual arrows saved to {save_path}")
    else:
        plt.show()
    model.train()
