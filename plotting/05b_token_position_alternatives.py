"""Ten alternative views of the token+position embedding sum (panel (e) of 05_token_embeddings).

Standalone: loads the trained checkpoint and writes a new figure; does not touch 05_token_embeddings.png.

Usage (from repo root):
    python plotting/05b_token_position_alternatives.py [checkpoint_dir] [out_path]
"""
import os
import sys
import pickle
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse, Polygon
from scipy.spatial import ConvexHull
from sklearn.svm import SVC

SUB = "₀₁₂₃₄₅₆₇₈₉"
def sub(n): return "".join(SUB[int(d)] for d in str(n))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ckpt = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "plus_last_even", "checkpoints")
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "plus_last_even", "plots", "a4",
                                                          "05b_token_position_alternatives.png")

sd = torch.load(os.path.join(ckpt, "model.pt"), map_location="cpu")
itos = pickle.load(open(os.path.join(ckpt, "itos.pkl"), "rb"))
E = sd["token_embedding.weight"].numpy()           # (V, 2)
P = sd["position_embedding_table.weight"].numpy()  # (T, 2)
V, T = E.shape[0], P.shape[0]
X = E[:, None, :] + P[None, :, :]                   # (V, T, 2)
labels = [itos[i] for i in range(V)]

def kind(s):
    if s == "+": return "plus"
    return "odd" if int(s) % 2 else "even"
odd = [i for i in range(V) if kind(labels[i]) == "odd"]
even = [i for i in range(V) if kind(labels[i]) == "even"]
plus = [i for i in range(V) if kind(labels[i]) == "plus"]

# Colors: evens = blues/greens (by value), odds = reds, '+' = black
even_cmap, odd_cmap = plt.get_cmap("viridis"), plt.get_cmap("Reds")
COL = {}
for k, i in enumerate(even): COL[i] = even_cmap(0.05 + 0.8 * k / max(len(even) - 1, 1))
for k, i in enumerate(odd):  COL[i] = odd_cmap(0.45 + 0.5 * k / max(len(odd) - 1, 1))
for i in plus: COL[i] = "black"
ODD_C = "#c0392b"

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
                     "axes.labelsize": 9})
BOLD_EP = r"$\mathbf{E}+\mathbf{P}$"

def base(ax, title):
    ax.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.5, zorder=0)
    ax.axvline(0, color="gray", ls="--", lw=0.8, alpha=0.5, zorder=0)
    ax.grid(alpha=0.25)
    ax.set_title(title)
    ax.set_xlabel("Dim 0"); ax.set_ylabel("Dim 1")

def full_limits(ax, pad=0.4):
    pts = X.reshape(-1, 2)
    ax.set_xlim(pts[:, 0].min() - pad, pts[:, 0].max() + pad)
    ax.set_ylim(pts[:, 1].min() - pad, pts[:, 1].max() + pad)

fig = plt.figure(figsize=(14, 31))
gs = GridSpec(5, 2, figure=fig, hspace=0.38, wspace=0.22, top=0.975, bottom=0.02)
panel_axes = []

# ---------- 1. Merge odd tokens into one label ----------
ax = fig.add_subplot(gs[0, 0]); panel_axes.append(ax)
base(ax, "1. Odd tokens merged into one label")
oddX = X[odd].mean(0)
spread = np.max(np.linalg.norm(X[odd] - oddX[None], axis=-1))
for i in even + plus:
    for t in range(T):
        ax.text(*X[i, t], f"{labels[i]}{sub(t)}", ha="center", va="center", fontsize=8,
                fontweight="bold", color="black")
for t in range(T):
    ax.text(*oddX[t], f"odd{sub(t)}", ha="center", va="center", fontsize=8,
            fontweight="bold", color=ODD_C)
ax.text(0.98, 0.04, f"odd = {{1,3,5,7,9}}\nmax spread around mean = {spread:.3f}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=ODD_C,
        bbox=dict(fc="white", ec=ODD_C, alpha=0.9))
full_limits(ax)

# ---------- 2. Zoom inset on odd cluster ----------
ax = fig.add_subplot(gs[0, 1]); panel_axes.append(ax)
base(ax, "2. Dots + zoom inset on the odd cluster")
for i in range(V):
    ax.scatter(X[i, :, 0], X[i, :, 1], s=18, color=COL[i], zorder=3)
for i in even + plus:
    ax.annotate(labels[i], X[i, -1], textcoords="offset points", xytext=(0, 6),
                ha="center", fontsize=8, fontweight="bold")
full_limits(ax)
t_zoom = 3
c = X[odd, t_zoom].mean(0)
half = 0.035
ax.add_patch(plt.Rectangle(c - half, 2 * half, 2 * half, fill=False, ec=ODD_C, lw=1.2, zorder=4))
ins = ax.inset_axes([0.56, 0.06, 0.4, 0.4])
for k, i in enumerate(odd):
    ins.scatter(*X[i, t_zoom], s=50, color=COL[i], edgecolor="k", lw=0.5, zorder=3)
    ins.annotate(labels[i], X[i, t_zoom], textcoords="offset points", xytext=[(6, 3), (6, -8), (-12, 4), (6, -10), (6, 3)][k],
                 fontsize=8, fontweight="bold", color=COL[i])
ins.set_xlim(c[0] - half, c[0] + half); ins.set_ylim(c[1] - half, c[1] + half)
ins.tick_params(labelsize=6); ins.grid(alpha=0.3)
ins.set_title(f"odd tokens at P{t_zoom} (box = {2*half:.2f} wide)", fontsize=7, fontweight="normal")
ax.indicate_inset_zoom(ins, edgecolor=ODD_C)

# ---------- 3. Dots + position trajectories ----------
ax = fig.add_subplot(gs[1, 0]); panel_axes.append(ax)
base(ax, "3. Trajectories P0 → P7 (size/opacity = position)")
sizes = np.linspace(8, 60, T); alphas = np.linspace(0.35, 1, T)
for i in range(V):
    ax.plot(X[i, :, 0], X[i, :, 1], color=COL[i], lw=1.2 if i not in odd else 3, alpha=0.6, zorder=2)
    for t in range(T):
        ax.scatter(*X[i, t], s=sizes[t], color=COL[i], alpha=alphas[t], edgecolor="none", zorder=3)
    if i in odd:
        continue
    ax.annotate(labels[i], X[i, -1], textcoords="offset points", xytext=(0, 7), ha="center",
                fontsize=8, fontweight="bold")
ax.annotate("odd (5 overlapping lines)", X[odd, -1].mean(0), textcoords="offset points",
            xytext=(8, 4), fontsize=8, fontweight="bold", color=ODD_C)
full_limits(ax)

# ---------- 4. Vector addition: E points + one shared P track ----------
ax = fig.add_subplot(gs[1, 1]); panel_axes.append(ax)
base(ax, r"4. Vector addition: $\mathbf{E}$ + shared $\mathbf{P}$ offset")
for i in range(V):
    ax.vlines(E[i, 0] + P[:, 0].mean(), X[i, :, 1].min(), X[i, :, 1].max(), color=COL[i],
              alpha=0.25, lw=6, zorder=1)
    ax.scatter(*E[i], s=40, color=COL[i], edgecolor="k", lw=0.5, zorder=3)
for i in even + plus:
    ax.annotate(labels[i], E[i], textcoords="offset points", xytext=(6, 2), fontsize=8, fontweight="bold")
ax.annotate("odd", E[odd].mean(0), textcoords="offset points", xytext=(6, 2), fontsize=8,
            fontweight="bold", color=ODD_C)
anchor = np.array([2.6, -3.0])
track = anchor + P - P[0]
for t in range(T - 1):
    ax.annotate("", track[t + 1], track[t], arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=1.4))
for t in range(T):
    ax.text(track[t, 0] + 0.08, track[t, 1], f"P{t}", fontsize=7, va="center", color="#2c3e50")
ax.text(anchor[0], anchor[1] - 0.35, r"$\mathbf{P}_0\to\mathbf{P}_7$" + "\n(same for every token)",
        ha="center", va="top", fontsize=8, color="#2c3e50")
ax.text(0.02, 0.98, "dot = token embedding E\nshaded bar = range covered by E+P",
        transform=ax.transAxes, va="top", fontsize=7.5)
ax.set_xlim(-2, 4); ax.set_ylim(-5, 3.2)

# ---------- 5. Small multiples by position ----------
sg = gs[2, 0].subgridspec(2, 4, wspace=0.08, hspace=0.25)
pts = X.reshape(-1, 2)
lims = (pts[:, 0].min() - 0.4, pts[:, 0].max() + 0.4), (pts[:, 1].min() - 0.4, pts[:, 1].max() + 0.4)
sm_axes = []
for t in range(T):
    a = fig.add_subplot(sg[t // 4, t % 4]); sm_axes.append(a)
    a.axhline(0, color="gray", ls="--", lw=0.6, alpha=0.5)
    for i in even + plus:
        a.text(*X[i, t], labels[i], ha="center", va="center", fontsize=7, fontweight="bold")
    a.text(*X[odd, t].mean(0), "odd", ha="center", va="center", fontsize=7, fontweight="bold", color=ODD_C)
    a.set_xlim(*lims[0]); a.set_ylim(*lims[1])
    a.set_title(f"P{t}", fontsize=8, fontweight="normal", pad=2)
    a.tick_params(labelsize=6)
    if t % 4: a.set_yticklabels([])
    if t < 4: a.set_xticklabels([])
sm_axes[0].text(0, 1.28, "5. Small multiples: one panel per position", transform=sm_axes[0].transAxes,
                fontsize=10, fontweight="bold")
panel_axes.append(sm_axes[0])

# ---------- 6. Split by token type ----------
sg = gs[2, 1].subgridspec(1, 3, wspace=0.45)
titles = ["even", "odd (zoomed x)", "+"]
groups = [even, odd, plus]
split_axes = []
for k, (g, ttl) in enumerate(zip(groups, titles)):
    a = fig.add_subplot(sg[0, k]); split_axes.append(a)
    for i in g:
        a.plot(X[i, :, 0], X[i, :, 1], "-o", ms=3, color=COL[i], lw=1)
        a.annotate(labels[i], X[i, -1], textcoords="offset points", xytext=(0, 5), ha="center",
                   fontsize=8, fontweight="bold", color=COL[i] if g is odd else "black")
    a.set_title(ttl, fontsize=9, fontweight="normal")
    a.grid(alpha=0.25); a.tick_params(labelsize=7); a.set_xlabel("Dim 0")
    if k == 0: a.set_ylabel("Dim 1")
split_axes[0].text(0, 1.1, "6. Split by token type (own axes each)", transform=split_axes[0].transAxes,
                   fontsize=10, fontweight="bold")
panel_axes.append(split_axes[0])

# ---------- 7. Dim 1 vs position ----------
ax = fig.add_subplot(gs[3, 0]); panel_axes.append(ax)
ax.set_title("7. Dim 1 of E+P vs position (Dim 0 ≈ constant per token)")
pos = np.arange(T)
for i in even + plus:
    ax.plot(pos, X[i, :, 1], "-o", ms=3, color=COL[i], lw=1.3)
    ax.text(T - 0.85, X[i, -1, 1], labels[i], va="center", fontsize=8, fontweight="bold")
lo, hi = X[odd, :, 1].min(0), X[odd, :, 1].max(0)
ax.fill_between(pos, lo - 0.03, hi + 0.03, color=ODD_C, alpha=0.35, lw=0)
ax.plot(pos, X[odd, :, 1].mean(0), color=ODD_C, lw=1.5)
ax.text(T - 0.85, X[odd, -1, 1].mean(), "odd\n(band)", va="center", fontsize=8, fontweight="bold", color=ODD_C)
ax.axhline(0, color="gray", ls="--", lw=0.8)
ax.set_xticks(pos); ax.set_xticklabels([f"P{t}" for t in pos])
ax.set_xlim(-0.3, T + 0.2); ax.set_xlabel("Position"); ax.set_ylabel("Dim 1"); ax.grid(alpha=0.25)

# ---------- 8. Heatmaps tokens x positions ----------
sg = gs[3, 1].subgridspec(1, 2, wspace=0.35)
order = even + odd + plus
hm_axes = []
for d in range(2):
    a = fig.add_subplot(sg[0, d]); hm_axes.append(a)
    M = X[order, :, d]
    vmax = np.abs(M).max()
    im = a.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    for r in range(M.shape[0]):
        for cc in range(T):
            a.text(cc, r, f"{M[r, cc]:.1f}", ha="center", va="center", fontsize=5.5,
                   color="white" if abs(M[r, cc]) > 0.6 * vmax else "black")
    a.set_yticks(range(len(order))); a.set_yticklabels([labels[i] for i in order], fontsize=7)
    for tl, i in zip(a.get_yticklabels(), order):
        if i in odd: tl.set_color(ODD_C)
    a.set_xticks(range(T)); a.set_xticklabels([f"P{t}" for t in range(T)], fontsize=6)
    a.axhline(len(even) - 0.5, color="k", lw=1); a.axhline(len(even) + len(odd) - 0.5, color="k", lw=1)
    a.set_title(f"Dim {d}", fontsize=9, fontweight="normal")
    fig.colorbar(im, ax=a, fraction=0.05, pad=0.03).ax.tick_params(labelsize=6)
hm_axes[0].text(0, 1.1, "8. Heatmaps of " + BOLD_EP + " (token × position)",
                transform=hm_axes[0].transAxes, fontsize=10, fontweight="bold")
panel_axes.append(hm_axes[0])

# ---------- 9. Signed distance to odd/even boundary ----------
ax = fig.add_subplot(gs[4, 0]); panel_axes.append(ax)
ax.set_title("9. Signed distance to the odd | even boundary (linear SVM)")
Xtr = np.concatenate([X[even].reshape(-1, 2), X[odd].reshape(-1, 2)])
ytr = np.array([1] * (len(even) * T) + [0] * (len(odd) * T))
svm = SVC(kernel="linear", C=1e6).fit(Xtr, ytr)  # hard-margin
w, b = svm.coef_[0], svm.intercept_[0]
dist = (X @ w + b) / np.linalg.norm(w)              # (V, T), >0 = even side
for i in even + plus:
    ax.plot(pos, dist[i], "-o", ms=3, color=COL[i], lw=1.3)
    ax.text(T - 0.85, dist[i, -1], labels[i], va="center", fontsize=8, fontweight="bold")
ax.fill_between(pos, dist[odd].min(0), dist[odd].max(0), color=ODD_C, alpha=0.35, lw=0)
ax.plot(pos, dist[odd].mean(0), color=ODD_C, lw=1.5)
ax.text(T - 0.85, dist[odd, -1].mean(), "odd", va="center", fontsize=8, fontweight="bold", color=ODD_C)
ax.axhline(0, color="k", lw=1.2)
yl = ax.get_ylim()
ax.axhspan(0, yl[1], color="tab:blue", alpha=0.05)
ax.axhspan(yl[0], 0, color=ODD_C, alpha=0.05)
ax.set_ylim(*yl)
ax.text(0.01, 0.97, "even side", transform=ax.transAxes, va="top", fontsize=8, color="tab:blue")
ax.text(0.01, 0.03, "odd side", transform=ax.transAxes, va="bottom", fontsize=8, color=ODD_C)
ax.set_xticks(pos); ax.set_xticklabels([f"P{t}" for t in pos])
ax.set_xlim(-0.3, T + 0.2); ax.set_xlabel("Position"); ax.set_ylabel("Signed distance"); ax.grid(alpha=0.25)
margin = min(dist[even].min(), -dist[odd].max())
ax.text(0.99, 0.5, f"min margin = {margin:.2f}", transform=ax.transAxes, ha="right", fontsize=8,
        bbox=dict(fc="white", ec="gray"))

# ---------- 10. Hulls / ellipses per group ----------
ax = fig.add_subplot(gs[4, 1]); panel_axes.append(ax)
base(ax, "10. One shape per token (odd tokens pooled)")
def hull(ax, pts, color, alpha=0.3):
    if len(pts) < 3 or np.ptp(pts[:, 0]) < 1e-3:
        m = pts.mean(0); s = pts.std(0) * 2 + 0.05
        ax.add_patch(Ellipse(m, s[0] * 2, s[1] * 2, fc=color, ec=color, alpha=alpha))
        return
    h = ConvexHull(pts)
    ax.add_patch(Polygon(pts[h.vertices], closed=True, fc=color, ec=color, alpha=alpha, lw=1.5,
                         joinstyle="round"))
for i in even + plus:
    hull(ax, X[i], COL[i])
    ax.text(X[i, -1, 0], X[i, -1, 1] + 0.15, labels[i], ha="center", va="bottom", fontsize=10, fontweight="bold")
hull(ax, X[odd].reshape(-1, 2), ODD_C, alpha=0.35)
ax.text(X[odd, :, 0].max() + 0.12, X[odd].reshape(-1, 2).mean(0)[1], "odd {1,3,5,7,9}", ha="left",
        va="center", fontsize=9, fontweight="bold", color=ODD_C)
full_limits(ax)


fig.suptitle("Token + position embeddings (" + BOLD_EP + "): ten alternative views",
             fontsize=14, fontweight="bold", y=0.99)
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=200, bbox_inches="tight")
print("saved", out)
