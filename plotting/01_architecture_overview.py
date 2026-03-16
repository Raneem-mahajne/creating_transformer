"""Plotting: plot_architecture_diagram."""
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


def plot_architecture_diagram(config: dict, save_path: str = None, model=None, vocab_size=None, batch_size=None):
    """Generate a professional architecture diagram using matplotlib.

    Produces a clean flow-chart with rounded boxes (FancyBboxPatch),
    consistent spacing, a modern colour palette, and proper arrowheads.
    Saves both PNG and SVG side-by-side.
    """
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
    import matplotlib.patheffects as pe

    # ── extract dimensions ──────────────────────────────────────────────
    has_ln1 = has_ln2 = False
    has_proj = False

    if model is not None:
        has_ln1 = hasattr(model, 'ln1') and 'ln1' in model._modules
        has_ln2 = hasattr(model, 'ln2') and 'ln2' in model._modules
        vocab_size_model = model.token_embedding.num_embeddings
        n_embd = model.token_embedding.embedding_dim
        block_size = model.block_size
        num_heads = len(model.sa_heads.heads)
        head_size = model.sa_heads.heads[0].head_size
        has_proj = hasattr(model, 'proj') and 'proj' in model._modules
        ffwd_net = model.ffwd.net
        ffwd_hidden_dim = ffwd_net[0].out_features
        if vocab_size is None:
            vocab_size = vocab_size_model
    else:
        mc = config['model']
        dc = config['data']
        if vocab_size is None:
            vocab_size = dc['max_value'] - dc['min_value'] + 1
            if dc.get('generator_type') in ['PlusLastEvenRule']:
                vocab_size += 1
        n_embd = mc['n_embd']; block_size = mc['block_size']
        num_heads = mc['num_heads']; head_size = mc['head_size']
        ffwd_hidden_dim = n_embd * 16

    if batch_size is None:
        batch_size = config.get('training', {}).get('batch_size', 4)

    # ── colours ─────────────────────────────────────────────────────────
    C_INPUT   = '#E8F4FD'; C_EMBED = '#DAEAF6'
    C_ATTN    = '#FFF8E1'; C_ATTN_BG = '#FFFDF5'; C_ATTN_BD = '#F9A825'
    C_LINEAR  = '#FADBD8'; C_OUTPUT = '#D5F5E3'
    C_STROKE  = '#34495E'; C_RESID = '#27AE60'
    C_SUB     = '#666666'

    # Check if residuals are used
    use_residual = True
    if model is not None:
        use_residual = getattr(model, 'use_residual', True)

    # ── figure setup (defer xlim until we know total width) ───────────
    if _u._JOURNAL_MODE:
        # Vertical layout for A4 paper: 7" wide, tall for clarity
        fig, ax = plt.subplots(figsize=(7.0, 14.0), dpi=200)
        H_px = 2800   # accommodate LARGER boxes
        W_px = 420    # width (wider for legibility)
        ax.set_ylim(H_px, 0)
        ax.set_xlim(0, W_px)
    else:
        H_px = 500
        fig, ax = plt.subplots(figsize=(20, 6), dpi=160)
        ax.set_ylim(H_px, 0)        # y increases downward (like SVG)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    # ── drawing helpers ─────────────────────────────────────────────────
    _fs = 9 if _u._JOURNAL_MODE else 10.5
    _sub_fs = 8 if _u._JOURNAL_MODE else 8.5
    def draw_box(x, y, w, h, color, label, sub=None, fs=None, sub_fs=None, gap=None, sub_lh=None):
        """Rounded-rectangle box. Text centred with generous internal spacing."""
        if fs is None: fs = _fs
        if sub_fs is None: sub_fs = _sub_fs
        fancy = FancyBboxPatch((x, y), w, h,
                               boxstyle="round,pad=0,rounding_size=10",
                               facecolor=color, edgecolor=C_STROKE,
                               linewidth=1.4, zorder=3)
        ax.add_patch(fancy)
        cx, cy_box = x + w / 2, y + h / 2
        lines = label.split('\n')
        n = len(lines)
        lh = fs * 1.6   # line height for spacing
        if sub:
            sub_lines = sub.split('\n')
            n_sub = len(sub_lines)
            label_block = lh * (n - 1)
            _sub_lh = sub_lh if sub_lh is not None else sub_fs * 1.5
            sub_block = _sub_lh * (n_sub - 1)
            gap_between = gap if gap is not None else 22
            total = label_block + gap_between + sub_block
            label_top = cy_box - total / 2
            for i, ln in enumerate(lines):
                ax.text(cx, label_top + i * lh, ln, ha='center', va='center',
                        fontsize=fs, fontweight='bold', color=C_STROKE, zorder=4,
                        fontfamily='sans-serif')
            sub_top = label_top + label_block + gap_between
            for j, sl in enumerate(sub_lines):
                ax.text(cx, sub_top + j * _sub_lh, sl,
                        ha='center', va='center',
                        fontsize=sub_fs, color=C_STROKE, zorder=4,
                        fontfamily='sans-serif', fontweight='normal')
        else:
            base_y = cy_box - lh * (n - 1) / 2
            for i, ln in enumerate(lines):
                ax.text(cx, base_y + i * lh, ln, ha='center', va='center',
                        fontsize=fs, fontweight='bold', color=C_STROKE, zorder=4,
                        fontfamily='sans-serif')

    _arrow_lw = 1.8 if _u._JOURNAL_MODE else 1.3
    _arrow_style = '-|>' if _u._JOURNAL_MODE else '->'
    def draw_arrow(x1, y1, x2, y2, color=C_STROKE, lw=None, shrinkA=2, shrinkB=2):
        lw = lw if lw is not None else _arrow_lw
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=_arrow_style, color=color, lw=lw,
                                    shrinkA=shrinkA, shrinkB=shrinkB,
                                    mutation_scale=12 if _u._JOURNAL_MODE else 10),
                    zorder=5)

    def draw_arrow_vertical_path(x1, y1, x2, y2, color=C_STROKE, lw=None):
        """Draw path: vertical, horizontal, vertical. Arrow tip at end of final vertical (pointing down)."""
        lw = lw if lw is not None else _arrow_lw
        y_mid = (y1 + y2) / 2
        ax.plot([x1, x1], [y1, y_mid], color=color, lw=lw, zorder=5, solid_capstyle='round')
        ax.plot([x1, x2], [y_mid, y_mid], color=color, lw=lw, zorder=5, solid_capstyle='round')
        ax.plot([x2, x2], [y_mid, y2 - 2], color=color, lw=lw, zorder=5, solid_capstyle='round')
        draw_arrow(x2, y2 - 2, x2, y2, color=color, lw=lw, shrinkA=0, shrinkB=0)

    _plus_fs = 13 if _u._JOURNAL_MODE else 15
    def draw_circle(cx, cy, r, label='+'):
        circ = Circle((cx, cy), r, facecolor='white', edgecolor=C_STROKE,
                       linewidth=1.4 if not _u._JOURNAL_MODE else 1.6, zorder=3)
        ax.add_patch(circ)
        ax.text(cx, cy, label, ha='center', va='center',
                fontsize=_plus_fs, fontweight='bold', color=C_STROKE, zorder=4,
                fontfamily='sans-serif')

    _resid_lw = 1.8 if _u._JOURNAL_MODE else 1.3
    _resid_fs = 9 if _u._JOURNAL_MODE else 8
    def draw_residual(x1, y_start, x2, y_end, label='residual', below=True):
        """Right-angle residual: go down, across, then up to target."""
        drop = 55 if below else -55
        y_mid = y_start + drop
        # down from source
        ax.plot([x1, x1], [y_start, y_mid], color=C_RESID, lw=_resid_lw,
                linestyle='--', zorder=2, clip_on=False)
        # across
        ax.plot([x1, x2], [y_mid, y_mid], color=C_RESID, lw=_resid_lw,
                linestyle='--', zorder=2, clip_on=False)
        # up to target (with arrow)
        draw_arrow(x2, y_mid, x2, y_end, color=C_RESID, lw=_resid_lw)
        # label
        ax.text((x1 + x2) / 2, y_mid + (12 if below else -12), label,
                ha='center', va='center', fontsize=_resid_fs, color=C_RESID,
                fontfamily='sans-serif', fontstyle='italic', zorder=4)

    # ── layout constants ────────────────────────────────────────────────
    cy = 210           # centre-line y
    bh = 80            # standard box height
    gap = 24           # horizontal gap between elements
    r_plus = 18        # radius of + circles
    if _u._JOURNAL_MODE:
        r_plus = 14
        vgap = 130   # vertical gap between sections
        vh = 90     # box height per row (LARGE for clear text)
        vw = 160    # box width

    # ── build diagram ───────────────────────────────────────────────────
    if _u._JOURNAL_MODE:
        cx = W_px / 2
        skip_x = 42          # left-side x for skip connection routing
        y = 55               # top margin so "Input Tokens" isn't cut off

        # ── Input Tokens ────────────────────────────────────────────
        bw, bh_v = 130, vh
        draw_box(cx - bw/2, y, bw, bh_v, C_INPUT, 'Input\nTokens', f'({batch_size},{block_size})', fs=10, sub_fs=8, gap=16)
        y_inp_b = y + bh_v
        y += bh_v + vgap * 1.4

        # ── Token Emb + Position Emb (side by side) ────────────────
        eb_w, eb_h = 95, 72
        te_x = cx - eb_w - 15
        pe_x = cx + 15
        draw_box(te_x, y, eb_w, eb_h, C_EMBED, 'Token Emb', f'({vocab_size},{n_embd})', fs=10, sub_fs=8, gap=16)
        draw_box(pe_x, y, eb_w, eb_h, C_EMBED, 'Position Emb', f'({block_size},{n_embd})', fs=10, sub_fs=8, gap=16)
        draw_arrow_vertical_path(cx, y_inp_b, te_x + eb_w/2, y)
        draw_arrow_vertical_path(cx, y_inp_b, pe_x + eb_w/2, y)
        y_emb_b = y + eb_h
        y += eb_h + vgap * 1.3

        # ── Add embeddings (+) ─────────────────────────────────────
        plus_cx, plus_y = cx, y + r_plus
        draw_circle(plus_cx, plus_y, r_plus)
        draw_arrow_vertical_path(te_x + eb_w/2, y_emb_b, plus_cx - 6, plus_y - r_plus)
        draw_arrow_vertical_path(pe_x + eb_w/2, y_emb_b, plus_cx + 6, plus_y - r_plus)
        ax.text(plus_cx + r_plus + 8, plus_y, f'x: ({batch_size},{block_size},{n_embd})',
                ha='left', va='center', fontsize=7, color='#555', fontfamily='sans-serif')
        y_add_b = plus_y + r_plus
        y += 2 * r_plus + vgap * 1.7

        # ── Self-Attention: Q, K, V (LARGE boxes, clear text) ─────────────────────
        qkv_w, qkv_h = 70, 88
        qkv_gap = 44
        total_qkv_w = 3 * qkv_w + 2 * qkv_gap
        qkv_x0 = cx - total_qkv_w / 2
        qkv_cx = []
        for i, lbl in enumerate(['Q', 'K', 'V']):
            bx = qkv_x0 + i * (qkv_w + qkv_gap)
            draw_box(bx, y, qkv_w, qkv_h, C_ATTN, lbl, f'{n_embd}\u2192{head_size}', fs=11, sub_fs=9, gap=18)
            qkv_cx.append(bx + qkv_w / 2)
        # Vertical trunk from + down, then branch; vertical arrows into each Q/K/V box
        fan_y = y - 55
        ax.plot([cx, cx], [y_add_b, fan_y], color=C_STROKE, lw=_arrow_lw, zorder=5, solid_capstyle='round')
        for i, qc in enumerate(qkv_cx):
            ax.plot([cx, qc], [fan_y, fan_y], color=C_STROKE, lw=_arrow_lw, zorder=5, solid_capstyle='round')
            draw_arrow(qc, fan_y, qc, y, shrinkA=0, shrinkB=0)
        for i, qc in enumerate(qkv_cx):
            ax.text(qc, fan_y + 4, f'$W_{["Q","K","V"][i]}$', ha='center', va='top',
                    fontsize=9, color=C_STROKE, fontweight='bold', fontfamily='sans-serif', zorder=6,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.9))
        y_qkv_b = y + qkv_h
        y += qkv_h + 70

        # ── Attention: vertical flow ─────────────────────────────────
        # Box A: QK^T / √d_k → Mask → Softmax
        score_w, score_h = 220, 62
        score_x = cx - score_w / 2
        draw_box(score_x, y, score_w, score_h, C_ATTN,
                 'QK\u1d40 / \u221Ad\u2096  \u2192  Mask  \u2192  Softmax', fs=10, gap=18)
        score_cx = cx
        # Q → left of score box, K → right of score box (vertical paths, no diagonals)
        draw_arrow_vertical_path(qkv_cx[0], y_qkv_b, score_cx - 30, y)
        draw_arrow_vertical_path(qkv_cx[1], y_qkv_b, score_cx + 30, y)
        y_score_b = y + score_h
        y += score_h + 75   # large gap for "Attention Weights" label

        # Arrow from Box A: vertical line, tip at end pointing down into Weights×V
        att_label_y = y_score_b + 30
        ax.plot([score_cx, score_cx], [y_score_b, y], color=C_STROKE, lw=_arrow_lw, zorder=5, solid_capstyle='round')
        draw_arrow(score_cx, y - 1, score_cx, y, shrinkA=0, shrinkB=0)
        ax.text(score_cx + 10, att_label_y, 'Attention Weights', ha='left', va='center',
                fontsize=8, fontweight='bold', fontstyle='italic', color='#555',
                fontfamily='sans-serif', zorder=6,
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='none', alpha=0.95))

        # Box B: Weights × V
        mulv_w, mulv_h = 140, 65
        mulv_x = cx - mulv_w / 2
        mulv_cx = cx
        draw_box(mulv_x, y, mulv_w, mulv_h, C_ATTN, 'Weights \u00d7 V', fs=11, gap=18)

        # V → Box B: vertical down from V, horizontal to center, vertical arrow down (tip at top of box)
        ax.plot([qkv_cx[2], qkv_cx[2]], [y_qkv_b, y],
                color=C_STROKE, lw=_arrow_lw, zorder=5, solid_capstyle='round')
        ax.plot([qkv_cx[2], cx], [y, y],
                color=C_STROKE, lw=_arrow_lw, zorder=5, solid_capstyle='round')
        draw_arrow(cx, y - 40, cx, y, shrinkA=0, shrinkB=0)

        y_mulv_b = y + mulv_h
        attnv_cx = mulv_cx
        y += mulv_h + vgap * 1.5

        # ── Residual add #1 (+) ────────────────────────────────────
        plus1_cx, plus1_y = cx, y + r_plus
        draw_circle(plus1_cx, plus1_y, r_plus)
        # Arrow from Box B to (+): vertical line, arrow tip at end pointing down
        ax.plot([attnv_cx, attnv_cx], [y_mulv_b, plus1_y - r_plus - 2], color=C_STROKE, lw=_arrow_lw, zorder=5, solid_capstyle='round')
        draw_arrow(attnv_cx, plus1_y - r_plus - 2, attnv_cx, plus1_y - r_plus, shrinkA=0, shrinkB=0)
        out_label_y = y_mulv_b + (plus1_y - r_plus - y_mulv_b) * 0.4
        ax.text(attnv_cx + 12, out_label_y, 'Attention Output', ha='left', va='center',
                fontsize=8, fontweight='bold', fontstyle='italic', color='#555',
                fontfamily='sans-serif', zorder=6,
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='none', alpha=0.95))
        # Skip connection: route down the LEFT side (around attention), then vertical into +
        ax.plot([plus_cx - r_plus, skip_x], [plus_y, plus_y],
                color=C_RESID, lw=_resid_lw, linestyle='--', zorder=2, clip_on=False)
        ax.plot([skip_x, skip_x], [plus_y, plus1_y + r_plus],
                color=C_RESID, lw=_resid_lw, linestyle='--', zorder=2, clip_on=False)
        ax.plot([skip_x, plus1_cx], [plus1_y + r_plus, plus1_y + r_plus],
                color=C_RESID, lw=_resid_lw, linestyle='--', zorder=2, clip_on=False)
        draw_arrow(plus1_cx, plus1_y + r_plus, plus1_cx, plus1_y - r_plus, color=C_RESID, lw=_resid_lw, shrinkA=0, shrinkB=0)
        ax.text(skip_x - 4, (plus_y + plus1_y) / 2, 'skip', ha='right', va='center',
                fontsize=_resid_fs, color=C_RESID, fontfamily='sans-serif', fontstyle='italic', zorder=4)
        y_plus1_b = plus1_y + r_plus
        y += 2 * r_plus + vgap * 1.5

        # ── Feed-Forward ───────────────────────────────────────────
        ff_w, ff_h = 180, vh + 24
        draw_box(cx - ff_w/2, y, ff_w, ff_h, C_LINEAR, 'Feed-Forward',
                 f'Linear({n_embd},{ffwd_hidden_dim})\nReLU\u2192Linear({ffwd_hidden_dim},{n_embd})', fs=10, sub_fs=9, gap=20, sub_lh=16)
        ax.plot([plus1_cx, plus1_cx], [y_plus1_b, y - 2], color=C_STROKE, lw=_arrow_lw, zorder=5, solid_capstyle='round')
        draw_arrow(plus1_cx, y - 2, plus1_cx, y, shrinkA=0, shrinkB=0)
        y_ff_b = y + ff_h
        y += ff_h + vgap * 1.5

        # ── Residual add #2 (+) ────────────────────────────────────
        plus2_cx, plus2_y = cx, y + r_plus
        draw_circle(plus2_cx, plus2_y, r_plus)
        ax.plot([cx, cx], [y_ff_b, plus2_y - r_plus - 2], color=C_STROKE, lw=_arrow_lw, zorder=5, solid_capstyle='round')
        draw_arrow(cx, plus2_y - r_plus - 2, cx, plus2_y - r_plus, shrinkA=0, shrinkB=0)
        # Skip connection: route down the LEFT side (around FFN), then vertical into +
        ax.plot([plus1_cx - r_plus, skip_x], [plus1_y, plus1_y],
                color=C_RESID, lw=_resid_lw, linestyle='--', zorder=2, clip_on=False)
        ax.plot([skip_x, skip_x], [plus1_y, plus2_y + r_plus],
                color=C_RESID, lw=_resid_lw, linestyle='--', zorder=2, clip_on=False)
        ax.plot([skip_x, plus2_cx], [plus2_y + r_plus, plus2_y + r_plus],
                color=C_RESID, lw=_resid_lw, linestyle='--', zorder=2, clip_on=False)
        draw_arrow(plus2_cx, plus2_y + r_plus, plus2_cx, plus2_y - r_plus, color=C_RESID, lw=_resid_lw, shrinkA=0, shrinkB=0)
        ax.text(skip_x - 4, (plus1_y + plus2_y) / 2, 'skip', ha='right', va='center',
                fontsize=_resid_fs, color=C_RESID, fontfamily='sans-serif', fontstyle='italic', zorder=4)
        y_plus2_b = plus2_y + r_plus
        y += 2 * r_plus + vgap * 1.5

        # ── LM Head ───────────────────────────────────────────────
        lm_w, lm_h = 140, vh + 16
        y_lm_top = y
        draw_box(cx - lm_w/2, y, lm_w, lm_h, C_OUTPUT, 'LM Head', f'Linear({n_embd},{vocab_size})', fs=10, sub_fs=9, gap=20)
        ax.plot([plus2_cx, plus2_cx], [y_plus2_b, y - 2], color=C_STROKE, lw=_arrow_lw, zorder=5, solid_capstyle='round')
        draw_arrow(plus2_cx, y - 2, plus2_cx, y, shrinkA=0, shrinkB=0)
        y_lm_b = y + lm_h
        y += lm_h + vgap * 1.3

        # ── Softmax → P(next) ─────────────────────────────────────
        sm_w, sm_h = 85, vh + 8
        draw_box(cx - sm_w - 8, y, sm_w, sm_h, C_OUTPUT, 'Softmax', fs=10)
        draw_box(cx + 8, y, sm_w, sm_h, C_OUTPUT, 'P(next)', f'({batch_size},{block_size},{vocab_size})', fs=10, sub_fs=8, gap=16)
        draw_arrow_vertical_path(cx, y_lm_b, cx - sm_w/2 - 6, y)
        draw_arrow(cx - 6, y + sm_h/2, cx + 6, y + sm_h/2, shrinkA=0, shrinkB=0)

        # ── Notation (top-right) ──────────────────────────────────
        ax.text(W_px - 10, 20, 'Notation', ha='right', va='top', fontsize=8, fontweight='bold', color=C_STROKE, fontfamily='sans-serif')
        for i, ln in enumerate([f'B={batch_size}', f'T={block_size}', f'C={n_embd}', f'd_k={head_size}', f'vocab={vocab_size}']):
            ax.text(W_px - 10, 34 + i * 14, ln, ha='right', va='top', fontsize=7, color='#555', fontfamily='sans-serif')

        # ── Legend (bottom, horizontal) ───────────────────────────
        ly = H_px - 35
        for i, (c, lbl) in enumerate([(C_EMBED, 'Embedding'), (C_ATTN, 'Attention'), (C_LINEAR, 'Feed-Forward'), (C_OUTPUT, 'Output')]):
            ox = 20 + i * 95
            fancy = FancyBboxPatch((ox, ly - 6), 12, 12, boxstyle="round,pad=0,rounding_size=2",
                                   facecolor=c, edgecolor=C_STROKE, linewidth=0.8, zorder=3)
            ax.add_patch(fancy)
            ax.text(ox + 18, ly, lbl, ha='left', va='center', fontsize=8, color=C_STROKE, fontfamily='sans-serif', zorder=4)
    else:
        # Original horizontal layout
        x = 30

        # Input Tokens
        inp_w = 88
        draw_box(x, cy - bh/2, inp_w, bh, C_INPUT, 'Input\nTokens', f'({batch_size}, {block_size})')
        x_inp_r = x + inp_w
        x += inp_w + gap

        # Token + Position Embeddings (stacked)
        emb_w, emb_h = 115, 64
        emb_gap = 14
        emb_top_y = cy - emb_h - emb_gap / 2
        emb_bot_y = cy + emb_gap / 2
        draw_box(x, emb_top_y, emb_w, emb_h, C_EMBED, 'Token Emb', f'({vocab_size}, {n_embd})')
        draw_box(x, emb_bot_y, emb_w, emb_h, C_EMBED, 'Position Emb', f'({block_size}, {n_embd})')
        draw_arrow(x_inp_r, cy - 12, x, emb_top_y + emb_h / 2)
        draw_arrow(x_inp_r, cy + 12, x, emb_bot_y + emb_h / 2)
        x_emb_r = x + emb_w
        x += emb_w + gap

        # Add embeddings (+)
        plus0_cx = x + r_plus
        draw_circle(plus0_cx, cy, r_plus)
        draw_arrow(x_emb_r, emb_top_y + emb_h / 2, plus0_cx - r_plus, cy - 6)
        draw_arrow(x_emb_r, emb_bot_y + emb_h / 2, plus0_cx - r_plus, cy + 6)
        # Shape label above
        ax.text(plus0_cx, emb_top_y - 14, f'x : (B,T,C) = ({batch_size},{block_size},{n_embd})',
                ha='center', va='center', fontsize=8, color='#777', fontfamily='sans-serif')
        x_add0_r = plus0_cx + r_plus
        x = x_add0_r + gap

        # ── Self-Attention block ────────────────────────────────────────────
        attn_x0 = x - 8

        # W_Q, W_K, W_V (stacked)
        qkv_w, qkv_h, qkv_gap = 66, 44, 8
        total_qkv = 3 * qkv_h + 2 * qkv_gap
        qkv_top = cy - total_qkv / 2
        qkv_cy_list = []
        for i, lbl in enumerate(['W_Q', 'W_K', 'W_V']):
            by = qkv_top + i * (qkv_h + qkv_gap)
            draw_box(x, by, qkv_w, qkv_h, C_ATTN, lbl, f'{n_embd}\u2192{head_size}', fs=10, sub_fs=8)
            qkv_cy_list.append(by + qkv_h / 2)
        # Fan-out arrows from + to each W_Q/W_K/W_V — use a short horizontal
        # trunk then individual branches so arrows don't cross through boxes
        fan_x = x_add0_r + (x - x_add0_r) * 0.5  # midpoint of gap
        # Trunk: horizontal line from + to the fan point
        ax.plot([x_add0_r, fan_x], [cy, cy],
                color=C_STROKE, lw=1.3, zorder=5, solid_capstyle='round')
        # Branches: from fan point to each QKV box
        for yc in qkv_cy_list:
            ax.plot([fan_x, fan_x], [cy, yc],
                    color=C_STROKE, lw=1.3, zorder=5, solid_capstyle='round')
            draw_arrow(fan_x, yc, x, yc, lw=1.3)
        x_qkv_r = x + qkv_w
        x += qkv_w + gap

        # QK^T / sqrt(dk)
        dot_w = 80
        x_dot_l = x
        draw_box(x, cy - bh/2, dot_w, bh, C_ATTN, 'QK\u1d40 / \u221Ad\u2096', fs=10)
        # Q → top of QK^T, K → bottom of QK^T (short horizontal arrows)
        draw_arrow(x_qkv_r, qkv_cy_list[0], x, cy - 12)
        draw_arrow(x_qkv_r, qkv_cy_list[1], x, cy + 12)
        x_dot_r = x + dot_w
        x += dot_w + gap

        # Causal Mask + Softmax
        mask_w = 90
        draw_box(x, cy - bh/2, mask_w, bh, C_ATTN, 'Causal Mask\n+ Softmax', fs=9.5)
        draw_arrow(x_dot_r, cy, x, cy)
        x_mask_r = x + mask_w
        x += mask_w + gap

        # Attn × V
        av_w = 78
        x_av_l = x
        draw_box(x, cy - bh/2, av_w, bh, C_ATTN, 'Attn \u00d7 V', fs=10.5)
        # Attention weights → top of Attn×V
        draw_arrow(x_mask_r, cy, x_av_l, cy - 8)
        # V path: route BELOW the QK^T and Mask boxes to avoid crossing
        # W_V right edge → down to below boxes → right → up into Attn×V bottom
        v_route_y = cy + bh / 2 + 14  # below the main-flow boxes
        v_start_x = x_qkv_r
        v_start_y = qkv_cy_list[2]  # W_V center-y
        v_end_x = x_av_l + av_w / 2  # center of Attn×V
        v_end_y = cy + bh / 2       # bottom edge of Attn×V
        # Draw the routed V path: right from W_V → down → across → up into Attn×V
        ax.plot([v_start_x, v_start_x + 10], [v_start_y, v_start_y],
                color=C_STROKE, lw=1.3, zorder=5, solid_capstyle='round')
        ax.plot([v_start_x + 10, v_start_x + 10], [v_start_y, v_route_y],
                color=C_STROKE, lw=1.3, zorder=5, solid_capstyle='round')
        ax.plot([v_start_x + 10, v_end_x], [v_route_y, v_route_y],
                color=C_STROKE, lw=1.3, zorder=5, solid_capstyle='round')
        draw_arrow(v_end_x, v_route_y, v_end_x, v_end_y, lw=1.3)
        # Label the V path
        ax.text((v_start_x + 10 + v_end_x) / 2, v_route_y + 10, 'V',
                ha='center', va='center', fontsize=9, fontweight='bold',
                color=C_STROKE, fontfamily='sans-serif', zorder=6)

        x_av_r = x + av_w
        x += av_w + gap + 6

        attn_x1 = x

        # Attention block outline (dashed) — must enclose V routing path
        attn_pad = 14
        attn_bottom = v_route_y + 22
        attn_rect_top = qkv_top - 30
        attn_rect = FancyBboxPatch(
            (attn_x0 - attn_pad, attn_rect_top), attn_x1 - attn_x0 + 2 * attn_pad, attn_bottom - attn_rect_top,
            boxstyle="round,pad=0,rounding_size=10",
            facecolor=C_ATTN_BG, edgecolor=C_ATTN_BD,
            linewidth=1.4, linestyle='--', zorder=1)
        ax.add_patch(attn_rect)
        ax.text((attn_x0 + attn_x1) / 2, qkv_top - 16,
                f'Self-Attention ({num_heads} head{"s" if num_heads > 1 else ""})',
                ha='center', va='center', fontsize=10, fontweight='bold',
                color=C_ATTN_BD, fontfamily='sans-serif', zorder=2)

        # ── Post-attention ──────────────────────────────────────────────────
        if use_residual:
            # Residual add #1: x + attn_out
            plus1_cx = x + r_plus
            draw_circle(plus1_cx, cy, r_plus)
            draw_arrow(x_av_r, cy, plus1_cx - r_plus, cy)
            # Residual path: from embedding add (+0) down-across-up to +1
            draw_residual(plus0_cx, cy + r_plus, plus1_cx, cy + r_plus,
                          label='x (skip around attention)')
            x_p1_r = plus1_cx + r_plus
            x = x_p1_r + gap
        else:
            x_p1_r = x_av_r
            x = x_av_r + gap

        # Feed-Forward
        ff_w = 125
        draw_box(x, cy - bh/2, ff_w, bh, C_LINEAR, 'Feed-Forward',
                 f'Linear({n_embd},{ffwd_hidden_dim})\nReLU \u2192 Linear({ffwd_hidden_dim},{n_embd})',
                 sub_fs=8)
        draw_arrow(x_p1_r, cy, x, cy)
        x_ff_r = x + ff_w
        x += ff_w + gap

        if use_residual:
            # Residual add #2: x + ffwd(x)
            plus2_cx = x + r_plus
            draw_circle(plus2_cx, cy, r_plus)
            draw_arrow(x_ff_r, cy, plus2_cx - r_plus, cy)
            # Residual path: from +1 down-across-up to +2
            draw_residual(plus1_cx, cy + r_plus, plus2_cx, cy + r_plus,
                          label='x (skip around FFN)')
            x_p2_r = plus2_cx + r_plus
            x = x_p2_r + gap
        else:
            x_p2_r = x_ff_r
            x = x_ff_r + gap

        # LM Head
        lm_w = 88
        draw_box(x, cy - bh/2, lm_w, bh, C_OUTPUT, 'LM Head',
                 f'Linear({n_embd}, {vocab_size})')
        draw_arrow(x_p2_r, cy, x, cy)
        x_lm_r = x + lm_w
        x += lm_w + gap

        # Softmax
        sm_w = 76
        draw_box(x, cy - bh/2, sm_w, bh, C_OUTPUT, 'Softmax')
        draw_arrow(x_lm_r, cy, x, cy)
        x_sm_r = x + sm_w
        x += sm_w + gap

        # Output Probabilities
        out_w = 92
        draw_box(x, cy - bh/2, out_w, bh, C_OUTPUT, 'P(next)',
                 f'({batch_size},{block_size},{vocab_size})')
        draw_arrow(x_sm_r, cy, x, cy)
        x += out_w + 30

        # ── Notation (top-right) ────────────────────────────────────────────
        nx = x - 115
        ny = 22
        ax.text(nx, ny, 'Notation', ha='left', va='top', fontsize=9.5,
                fontweight='bold', color=C_STROKE, fontfamily='sans-serif')
        for i, ln in enumerate([
            f'B = {batch_size}  (batch)',
            f'T = {block_size}  (sequence)',
            f'C = {n_embd}  (n_embd)',
            f'd_k = {head_size}  (head size)',
            f'vocab = {vocab_size}',
        ]):
            ax.text(nx, ny + 18 + i * 15, ln, ha='left', va='top',
                    fontsize=8, color='#555', fontfamily='sans-serif')

        # ── Legend (bottom-left) ────────────────────────────────────────────
        lx, ly = 30, H_px - 26
        for i, (c, lbl) in enumerate([
            (C_EMBED, 'Embedding'), (C_ATTN, 'Attention'),
            (C_LINEAR, 'Feed-Forward'), (C_OUTPUT, 'Output'),
        ]):
            ox = lx + i * 120
            fancy = FancyBboxPatch((ox, ly - 7), 14, 14,
                                   boxstyle="round,pad=0,rounding_size=3",
                                   facecolor=c, edgecolor=C_STROKE, linewidth=0.8, zorder=3)
            ax.add_patch(fancy)
            ax.text(ox + 20, ly, lbl, ha='left', va='center',
                    fontsize=8.5, color=C_STROKE, fontfamily='sans-serif', zorder=4)

        # ── finalise axes limits ────────────────────────────────────────────
        W_px = x + 20
        ax.set_xlim(0, W_px)
        ax.set_aspect('equal')

    # ── save ────────────────────────────────────────────────────────────
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=200, facecolor='white', pad_inches=0.15)
        # Also save SVG
        svg_path = save_path.rsplit('.', 1)[0] + '.svg'
        plt.savefig(svg_path, bbox_inches='tight', facecolor='white', pad_inches=0.15, format='svg')
        plt.close()
        print(f"Architecture diagram saved to {save_path} and {svg_path}")
    else:
        plt.show()

# -----------------------------
# Q/K Embedding Space Visualization
# -----------------------------
