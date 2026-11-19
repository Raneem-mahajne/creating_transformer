"""Options for linking the zoom insets in Figure 5 panel (e) to the region they magnify.

Standalone comparison figure (does not touch 05_token_embeddings.png). Each panel redraws
panel (e) (token + position embeddings) with one way of making the inset source clear.

Usage (from repo root):
    python plotting/05c_inset_link_options.py [checkpoint_dir] [out_path]
"""
import os
import sys
import pickle
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch

SUB = "₀₁₂₃₄₅₆₇₈₉"
def sub(n): return "".join(SUB[int(d)] for d in str(n))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ckpt = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "plus_last_even", "checkpoints")
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "plus_last_even", "plots", "a4",
                                                          "05c_inset_link_options.png")

sd = torch.load(os.path.join(ckpt, "model.pt"), map_location="cpu")
itos = pickle.load(open(os.path.join(ckpt, "itos.pkl"), "rb"))
E = sd["token_embedding.weight"].numpy()
P = sd["position_embedding_table.weight"].numpy()
V, T = E.shape[0], P.shape[0]
X = E[:, None, :] + P[None, :, :]                       # (V, T, 2)
labels = [str(itos[i]) for i in range(V)]
odd = [i for i in range(V) if labels[i].isdigit() and int(labels[i]) % 2 == 1]
ZOOM_POS = [4, 3]                                        # upper inset, lower inset
COLORS = {4: "#1f77b4", 3: "#d95f02"}
TINTS = {4: "#d6ecff", 3: "#ffe8c2"}

FS_MAIN, FS_INS = 10, 7
# Inset rects (axes fraction), as in Figure 5 panel (e)
W, H, X0, TOP, GAP = 0.105, 0.18, 0.53, 0.56, 0.04
RECTS = {p: [X0, TOP - (k + 1) * H - k * GAP, W, H] for k, p in enumerate(ZOOM_POS)}

pts_all = X.reshape(-1, 2)
XLIM = (pts_all[:, 0].min() - 2.0, pts_all[:, 0].max() + 1.9)
YLIM = (pts_all[:, 1].min() - 0.6, pts_all[:, 1].max() + 0.5)


def draw_main(ax, odd_style="text", color_rows=False, fade_others=False):
    """Main scatter of token+position labels. odd_style: 'text' or 'dots'."""
    ax.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.5, zorder=0)
    ax.axvline(0, color="gray", ls="--", lw=0.8, alpha=0.5, zorder=0)
    ax.grid(alpha=0.3)
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
    ax.set_xlabel("Dim 0"); ax.set_ylabel("Dim 1")
    for i in range(V):
        for t in range(T):
            x, y = X[i, t]
            is_odd = i in odd
            color, alpha = "black", 1.0
            if is_odd and color_rows and t in COLORS:
                color = COLORS[t]
            if is_odd and fade_others and t not in ZOOM_POS:
                color, alpha = "0.6", 0.35
            if is_odd and odd_style == "dots":
                ax.scatter(x, y, s=6, color=color, alpha=alpha, zorder=3)
            else:
                ax.text(x, y, f"{labels[i]}{sub(t)}", fontsize=FS_MAIN, fontweight="bold",
                        ha="center", va="center", color=color, alpha=alpha, zorder=3)


def add_inset(ax, p, color="black", box="tight", link="two", tint=None, tag=False):
    """Text-only zoom inset of the odd tokens at position p.

    box:  'tight' (zoom extent, as now), 'visible' (padded, drawn on top), or None
    link: 'two' (two connector lines), 'arrow' (one line with arrowhead), or None
    """
    pts = X[odd, p]
    c = pts.mean(0)
    R = np.linalg.norm(pts - c, axis=1).max()
    half = 2.0 * R
    ins = ax.inset_axes(RECTS[p])
    # Near-identical points (5 and 7) are nudged apart sideways.
    nudge = np.zeros(len(odd))
    done = set()
    for a in range(len(odd)):
        if a in done:
            continue
        dup = [b for b in range(len(odd)) if b not in done and np.linalg.norm(pts[b] - pts[a]) < 0.1 * R]
        for k, b in enumerate(dup):
            nudge[b] = 0.95 * FS_INS * (k - (len(dup) - 1) / 2)
        done.update(dup)
    for (x, y), i, dx in zip(pts, odd, nudge):
        ins.annotate(f"{labels[i]}{sub(p)}", (x, y), xytext=(dx, 0), textcoords="offset points",
                     fontsize=FS_INS, fontweight="bold", ha="center", va="center",
                     color=color if color != "black" else "black")
    ins.set_xlim(c[0] - half, c[0] + half); ins.set_ylim(c[1] - half, c[1] + half)
    ins.set_aspect("equal", adjustable="box")
    ins.set_xticks([]); ins.set_yticks([])
    for s in ins.spines.values():
        s.set_edgecolor(color); s.set_linewidth(1.6 if color != "black" else 1.0)
    if tint:
        ins.set_facecolor(tint)
    if tag:
        ins.text(0.05, 0.93, f"P{p}", transform=ins.transAxes, fontsize=FS_INS, fontweight="bold",
                 color=color, va="top")
        ax.text(c[0] - 0.32, c[1], f"P{p}", fontsize=FS_INS + 1, fontweight="bold", color=color,
                ha="right", va="center", zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color, lw=0.8))

    # Source region in the main axes
    if box == "visible" or tint:
        hw, hh = 0.2, 0.1
    else:
        hw = hh = half
    if tint:
        ax.add_patch(Rectangle((c[0] - hw, c[1] - hh), 2 * hw, 2 * hh, fc=tint, ec="none",
                               alpha=0.9, zorder=2))
    if box == "tight":
        ax.add_patch(Rectangle((c[0] - hw, c[1] - hh), 2 * hw, 2 * hh, fill=False, ec=color,
                               lw=1.0, alpha=0.8, zorder=5))
    elif box == "visible":
        ax.add_patch(Rectangle((c[0] - hw, c[1] - hh), 2 * hw, 2 * hh, fill=False, ec=color,
                               lw=1.8, zorder=6))

    lc = dict(color=color, lw=1.0 if color == "black" else 1.3, alpha=0.8)
    if link == "two":
        for ya, yb in [(0, c[1] - hh), (1, c[1] + hh)]:
            ax.figure.add_artist(ConnectionPatch(xyA=(0, ya), coordsA=ins.transAxes,
                                                 xyB=(c[0] + hw, yb), coordsB=ax.transData, **lc))
    elif link == "arrow":
        ax.figure.add_artist(ConnectionPatch(xyA=(0, 0.5), coordsA=ins.transAxes,
                                             xyB=(c[0] + hw, c[1]), coordsB=ax.transData,
                                             arrowstyle="-|>", mutation_scale=10, **lc))
    return ins


OPTIONS = [
    ("Current", dict(), dict()),
    ("1. Visible source box drawn on top", dict(), dict(box="visible")),
    ("2. One colour per inset (border, lines, box)", dict(), dict(colored=True, box="visible")),
    ("3. Colour + matching labels coloured in main plot", dict(color_rows=True), dict(colored=True, box="visible")),
    ("4. Fade odd labels at other positions", dict(fade_others=True), dict(box="visible")),
    ("5. Tinted source patch + matching inset background (no lines)", dict(), dict(tinted=True, box=None, link=None)),
    ("6. Single connector with arrowhead", dict(), dict(box="visible", link="arrow")),
    ("7. Position tags instead of lines", dict(), dict(box=None, link=None, tag=True, colored=True)),
    ("8. Odd tokens as dots in main plot", dict(odd_style="dots"), dict(box="visible")),
    ("Recommended: 2 + 3 + 4", dict(color_rows=True, fade_others=True), dict(colored=True, box="visible")),
]

fig, axes = plt.subplots(5, 2, figsize=(16, 30))
plt.subplots_adjust(hspace=0.28, wspace=0.12, top=0.975, bottom=0.02, left=0.05, right=0.98)
for ax, (title, main_kw, ins_kw) in zip(axes.flat, OPTIONS):
    draw_main(ax, **main_kw)
    ins_kw = dict(ins_kw)
    colored = ins_kw.pop("colored", False)
    tinted = ins_kw.pop("tinted", False)
    for p in ZOOM_POS:
        add_inset(ax, p, color=COLORS[p] if colored else "black",
                  tint=TINTS[p] if tinted else None, **ins_kw)
    ax.set_title(title, fontsize=12, fontweight="bold")

fig.suptitle("Ways to show where the panel (e) insets point", fontsize=15, fontweight="bold", y=0.993)
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
