"""Ten ways to label the arrow tips in Figure 15, panel e_i -> z_i (bottom right),
drawn at the panel's real A4 size and font size.

Every option puts each label at its arrow tip when that spot is free. If the label would overlap
an arrow, a point, another label or the panel edge, it moves to the nearest free spot around the
tip and a thin leader line runs back to the tip. Nothing is forced to overlap.

Standalone comparison figure; does not touch 17_residuals.png.

Usage (from repo root):
    python plotting/17b_tip_label_options.py [out_path]
"""
import os
import sys
import pickle
import importlib
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from model import BigramLanguageModel  # noqa: E402
_grid_mod = importlib.import_module("plotting.18_final_on_output_grid")

CKPT = os.path.join(ROOT, "plus_last_even", "checkpoints")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    ROOT, "plus_last_even", "plots", "a4", "17b_tip_label_options.png")

SUB = "₀₁₂₃₄₅₆₇₈₉"
def lab(tok, p): return f"{tok}{SUB[p]}"

# ---------------------------------------------------------------- data (same as Figure 15)
stoi = pickle.load(open(os.path.join(CKPT, "stoi.pkl"), "rb"))
itos = pickle.load(open(os.path.join(CKPT, "itos.pkl"), "rb"))
model = BigramLanguageModel(len(itos), 2, 8, 1, 2, use_residual=True)
model.load_state_dict(torch.load(os.path.join(CKPT, "model.pt"), map_location="cpu"))
model.eval()
SEQ = [4, 1, "+", 4, 6, 9, 5, "+"]
X = torch.tensor([[stoi[str(t)] for t in SEQ]])
with torch.no_grad():
    E = (model.token_embedding(X) + model.position_embedding_table(torch.arange(8)))
    A_out, _ = model.sa_heads(E)
E = E[0].numpy(); ATT = A_out[0].numpy(); Z = E + ATT
LABELS = [lab(t, p) for p, t in enumerate(SEQ)]
XX, YY, PROBS = _grid_mod.compute_second_residual_next_token_prob_grid(
    model, grid_resolution=120, extent_margin=0.5, extra_xy=np.vstack([E, ATT, Z]))
XLIM = (float(XX.min()), float(XX.max())); YLIM = (float(YY.min()), float(YY.max()))


def group_coincident(P, tol=0.15):
    used = [False] * len(P); groups = []
    for i in range(len(P)):
        if used[i]:
            continue
        g = [i]; used[i] = True
        for j in range(i + 1, len(P)):
            if not used[j] and np.hypot(*(P[i] - P[j])) < tol:
                g.append(j); used[j] = True
        groups.append(g)
    return groups


# Panel b: arrows origin -> Attn(E)_i (coincident outputs share one arrow/label).
B_ARROWS = [((0.0, 0.0), tuple(ATT[g].mean(0)), ", ".join(LABELS[i] for i in g))
            for g in group_coincident(ATT)]
# Panel c: arrows e_i -> z_i, labelled at the head z_i.
C_ARROWS = [(tuple(E[i]), tuple(Z[i]), LABELS[i]) for i in range(8)]
TOK_COLORS = plt.cm.tab10(np.linspace(0, 1, 10))


# ---------------------------------------------------------------- drawing helpers
def background(ax):
    V = PROBS.shape[2]
    tab = plt.get_cmap("tab20", 20)
    cmap = mcolors.ListedColormap([tab(i % 20) for i in range(V)])
    norm = mcolors.BoundaryNorm(np.arange(V + 1) - 0.5, cmap.N)
    ax.pcolormesh(XX, YY, np.argmax(PROBS, axis=2), cmap=cmap, norm=norm,
                  shading="auto", zorder=0, rasterized=True)
    ax.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.5, zorder=1)
    ax.axvline(0, color="gray", ls="--", lw=0.8, alpha=0.5, zorder=1)
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
    ax.tick_params(labelsize=5)


def arrow(ax, a, b, color="black", lw=1.1):
    ax.annotate("", xy=b, xytext=a, zorder=5,
                arrowprops=dict(arrowstyle="-|>,head_length=0.4,head_width=0.2", color=color,
                                lw=lw, shrinkA=0, shrinkB=0))


def unit(v):
    n = np.hypot(*v)
    return np.asarray(v) / n if n > 1e-9 else np.array([1.0, 0.0])


def align(d):
    """ha/va so text offset along d sits on the far side of its anchor."""
    ha = "left" if d[0] > 0.35 else "right" if d[0] < -0.35 else "center"
    va = "bottom" if d[1] > 0.35 else "top" if d[1] < -0.35 else "center"
    return ha, va


def _cross(p1, p2, q1, q2):
    def o(a, b, c): return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return (o(p1, p2, q1) * o(p1, p2, q2) < 0) and (o(q1, q2, p1) * o(q1, q2, p2) < 0)


def seg_hits_box(p, q, bb):
    x0, y0, x1, y1 = bb
    if any(x0 <= r[0] <= x1 and y0 <= r[1] <= y1 for r in (p, q)):
        return True
    cs = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return any(_cross(p, q, cs[k], cs[(k + 1) % 4]) for k in range(4))


def grow(bb, m): return (bb[0] - m, bb[1] - m, bb[2] + m, bb[3] + m)


def boxes_overlap(a, b): return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


BOX = dict(boxstyle="round,pad=0.25", fc="white", ec="0.25", lw=0.5)
# Sizes match Figure 15 (A4 version): bottom-right panel is 1.443 x 1.908 in, and its
# label text renders at 5.6 pt (fontsize 8 scaled by the journal-mode factor 0.7).
PANEL_W, PANEL_H = 1.443, 1.908
FS = 5.6
ANGLES = (0, 15, -15, 30, -30, 45, -45, 60, -60, 75, -75, 90, -90,
          105, -105, 120, -120, 135, -135, 150, -150, 165, -165, 180)
DISTS = (2, 5, 9, 14, 20, 28, 38)


def place_labels(ax, arrows, texts, style, extra_points=()):
    """Place each label at its arrow tip if free, else at the nearest free spot (with a leader).

    arrows: list of (tail, tip) in data coords; texts: label per arrow.
    style(k) -> dict of annotate kwargs (fontsize, bbox, colour, rotation, ...).
    """
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    T = ax.transData.transform
    segs = [(T(a), T(b)) for a, b in arrows]
    pts = [T(b) for _, b in arrows] + [T(p) for p in extra_points]
    axb = ax.get_window_extent(rend)
    inner = (axb.x0 + 1, axb.y0 + 1, axb.x1 - 1, axb.y1 - 1)
    ppt = fig.dpi / 72.0
    placed = []
    # Longest labels first: they are hardest to fit.
    for k in sorted(range(len(arrows)), key=lambda k: -len(texts[k])):
        a_px, b_px = segs[k]
        d0 = unit(b_px - a_px)
        chosen = None
        for dist in DISTS:
            for ang in ANGLES:
                r = np.radians(ang)
                d = np.array([d0[0] * np.cos(r) - d0[1] * np.sin(r),
                              d0[0] * np.sin(r) + d0[1] * np.cos(r)])
                ha, va = align(d)
                kw = dict(style(k))
                if kw.pop("rotate_along", False):
                    ang_deg = np.degrees(np.arctan2(d[1], d[0]))
                    flip = abs(ang_deg) > 90
                    kw.update(rotation=ang_deg + 180 if flip else ang_deg, rotation_mode="anchor",
                              ha="right" if flip else "left", va="center")
                else:
                    kw.update(ha=ha, va=va)
                ann = ax.annotate(texts[k], xy=arrows[k][1], xytext=tuple(dist * d),
                                  textcoords="offset points", zorder=8, **kw)
                # Text extent, grown by the box padding when the label has a box (the box patch
                # itself is only positioned at draw time, so its extent is not reliable here).
                tb = ann.get_window_extent(rend)
                bb = (tb.x0, tb.y0, tb.x1, tb.y1)
                if "bbox" in kw:
                    padpt = 0.3 * kw.get("fontsize", FS) * ppt + 1
                    bb = grow(bb, padpt)
                ok = bb[0] >= inner[0] and bb[1] >= inner[1] and bb[2] <= inner[2] and bb[3] <= inner[3]
                for j, (p, q) in enumerate(segs):
                    if not ok:
                        break
                    if j == k:
                        # Own arrow: only its shaft (not the last few px at the tip) must stay clear.
                        q_short = q - 4 * ppt * unit(q - p)
                        ok = not seg_hits_box(p, q_short, bb)
                    else:
                        ok = not seg_hits_box(p, q, grow(bb, 2)) and not (
                            bb[0] - 3 * ppt <= q[0] <= bb[2] + 3 * ppt and bb[1] - 3 * ppt <= q[1] <= bb[3] + 3 * ppt)
                if ok:
                    ok = not any(boxes_overlap(grow(bb, 2), o) for o in placed)
                if ok:
                    ok = not any(grow(bb, 2)[0] <= p[0] <= grow(bb, 2)[2] and grow(bb, 2)[1] <= p[1] <= grow(bb, 2)[3]
                                 for i, p in enumerate(pts) if i != k)
                if ok and dist > DISTS[1]:
                    # A moved label gets a leader from the tip to the nearest box point; that
                    # leader must not cross other arrows or already-placed labels.
                    lead_end = (min(max(b_px[0], bb[0]), bb[2]), min(max(b_px[1], bb[1]), bb[3]))
                    ok = not any(_cross(b_px, lead_end, p, q) for j, (p, q) in enumerate(segs) if j != k)
                    if ok:
                        ok = not any(seg_hits_box(b_px, lead_end, grow(o, 1)) for o in placed)
                if ok:
                    chosen = (ann, bb, dist, kw)
                    break
                ann.remove()
            if chosen:
                break
        if chosen is None:   # nothing free: fall back to straight past the tip
            ha, va = align(d0)
            kw = dict(style(k)); kw.pop("rotate_along", None)
            kw.update(ha=ha, va=va)
            ann = ax.annotate(texts[k], xy=arrows[k][1], xytext=tuple(4 * d0),
                              textcoords="offset points", zorder=8, **kw)
            chosen = (ann, (0, 0, 0, 0), 4, kw)
        ann, bb, dist, kw = chosen
        if dist > DISTS[1]:
            # Moved away from the tip: redraw with a thin leader that touches the tip.
            xyann = ann.xyann
            ann.remove()
            ax.annotate(texts[k], xy=arrows[k][1], xytext=xyann, textcoords="offset points",
                        zorder=8, arrowprops=dict(arrowstyle="-", color="0.2", lw=0.6,
                                                  shrinkA=0, shrinkB=0), **kw)
        placed.append(bb)


def S(**kw):
    return lambda k: dict(fontsize=FS, fontweight="bold", **kw)


# ---------------------------------------------------------------- the ten options
def run(ax, arrows, style, texts=None, arrow_colors=None, extra_points=()):
    for k, (a, b, _) in enumerate(arrows):
        arrow(ax, a, b, color=arrow_colors[k] if arrow_colors else "black",
              lw=1.4 if arrow_colors else 1.1)
    place_labels(ax, [(a, b) for a, b, _ in arrows], texts or [s for _, _, s in arrows], style,
                 extra_points)


def opt1(ax, arrows, ep):   # boxed label at the tip
    run(ax, arrows, S(bbox=BOX), extra_points=ep)

def opt2(ax, arrows, ep):   # plain text at the tip
    run(ax, arrows, S(), extra_points=ep)

def opt3(ax, arrows, ep):   # text with white halo at the tip
    run(ax, arrows, S(path_effects=[pe.withStroke(linewidth=2.5, foreground="white")]),
        extra_points=ep)

def opt4(ax, arrows, ep):   # smaller, non-bold boxed label
    run(ax, arrows, lambda k: dict(fontsize=0.8 * FS, bbox=dict(BOX, pad=0.15)), extra_points=ep)

def opt5(ax, arrows, ep):   # label in a box with a thin coloured edge per token
    run(ax, arrows, lambda k: dict(fontsize=FS, fontweight="bold",
                                   bbox=dict(BOX, ec=TOK_COLORS[k % 10], lw=1.2)), extra_points=ep)

def opt6(ax, arrows, ep):   # one colour per token: arrow + label
    cols = [TOK_COLORS[k % 10] * [0.75, 0.75, 0.75, 1] for k in range(len(arrows))]
    run(ax, arrows, lambda k: dict(fontsize=FS, fontweight="bold", color=cols[k],
                                   path_effects=[pe.withStroke(linewidth=2.5, foreground="white")]),
        arrow_colors=cols, extra_points=ep)

def opt7(ax, arrows, ep):   # numbered tags at the tips + key
    tags = [str(k + 1) for k in range(len(arrows))]
    run(ax, arrows, lambda k: dict(fontsize=0.8 * FS, fontweight="bold", color="white",
                                   bbox=dict(boxstyle="circle,pad=0.2", fc="black", ec="white", lw=0.5)),
        texts=tags, extra_points=ep)
    ax.text(0.98, 0.02, "\n".join(f"{k + 1}: {s}" for k, (_, _, s) in enumerate(arrows)),
            transform=ax.transAxes, fontsize=0.75 * FS, ha="right", va="bottom", zorder=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5", lw=0.5))

def opt8(ax, arrows, ep):   # nearby tips share one stacked label
    tips = np.array([b for _, b, _ in arrows])
    groups = group_coincident(tips, tol=0.9)
    for a, b, _ in arrows:
        arrow(ax, a, b)
    merged = [(arrows[g[0]][0], tuple(tips[g].mean(0))) for g in groups]
    place_labels(ax, merged, ["\n".join(arrows[i][2] for i in g) for g in groups], S(bbox=BOX),
                 extra_points=list(ep) + [tuple(t) for t in tips])

def opt9(ax, arrows, ep):   # text rotated along the arrow
    run(ax, arrows, S(rotate_along=True,
                      path_effects=[pe.withStroke(linewidth=2.5, foreground="white")]),
        extra_points=ep)

def opt10(ax, arrows, ep):  # boxed label, arrows drawn thinner and grey
    for a, b, _ in arrows:
        arrow(ax, a, b, color="0.15", lw=0.8)
    place_labels(ax, [(a, b) for a, b, _ in arrows], [s for _, _, s in arrows], S(bbox=BOX),
                 extra_points=ep)


OPTIONS = [
    ("1. Boxed label at the tip", opt1),
    ("2. Plain text at the tip", opt2),
    ("3. Text with white halo at the tip", opt3),
    ("4. Smaller boxed label at the tip", opt4),
    ("5. Box edge coloured per token", opt5),
    ("6. One colour per token (arrow + label)", opt6),
    ("7. Numbered tags at the tips + key", opt7),
    ("8. Nearby tips share one stacked label", opt8),
    ("9. Text rotated along the arrow", opt9),
    ("10. Boxed label, thinner arrows", opt10),
]

# Panel (c) only, each drawn at Figure 15's real panel size.
NCOL, NROW = 5, 2
GAP_W, GAP_H, LEFT, TOP = 0.55, 0.75, 0.45, 0.55
FIG_W = LEFT + NCOL * (PANEL_W + GAP_W)
FIG_H = TOP + NROW * (PANEL_H + GAP_H)
fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=300)
for idx, (title, fn) in enumerate(OPTIONS):
    r, c = divmod(idx, NCOL)
    x0 = LEFT + c * (PANEL_W + GAP_W)
    y0 = FIG_H - TOP - (r + 1) * (PANEL_H + GAP_H) + GAP_H * 0.35
    ax = fig.add_axes([x0 / FIG_W, y0 / FIG_H, PANEL_W / FIG_W, PANEL_H / FIG_H])
    background(ax)
    ax.scatter(E[:, 0], E[:, 1], s=14, c="black", edgecolors="white", linewidths=0.6, zorder=6)
    fn(ax, C_ARROWS, [tuple(p) for p in E])
    ax.set_title(title, fontsize=6.3, fontweight="bold", loc="left")

fig.suptitle(r"Figure 15 panel $\mathbf{e}_i \rightarrow \mathbf{z}_i$ at its real size: "
             "labels at the arrow tips, moved only when blocked (10 options)",
             fontsize=8, fontweight="bold", y=0.995)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print("saved", OUT)
