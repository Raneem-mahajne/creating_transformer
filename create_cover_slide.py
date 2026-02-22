"""Create an artistic cover slide using actual model geometry as visual art."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
from matplotlib.collections import LineCollection
import matplotlib.patheffects as pe
import numpy as np
import torch

from checkpoint import load_checkpoint
from config_loader import load_config

config = load_config("plus_last_even")
checkpoint_data = load_checkpoint(config["name"], step=None)
model = checkpoint_data["model"]
itos = checkpoint_data["itos"]
model.eval()

vocab_size = model.token_embedding.weight.shape[0]
block_size = model.position_embedding_table.weight.shape[0]
token_emb = model.token_embedding.weight.detach().cpu().numpy()
pos_emb = model.position_embedding_table.weight.detach().cpu().numpy()
head = model.sa_heads.heads[0]
W_Q = head.query.weight.detach().cpu().numpy()
W_K = head.key.weight.detach().cpu().numpy()
head_size = W_Q.shape[0]

Q_all, K_all, labels_all = [], [], []
for t in range(vocab_size):
    for p in range(block_size):
        combined = token_emb[t] + pos_emb[p]
        Q_all.append(W_Q @ combined)
        K_all.append(W_K @ combined)
        labels_all.append(itos[t])
Q_all = np.array(Q_all)
K_all = np.array(K_all)
attn_raw = (Q_all @ K_all.T) / np.sqrt(head_size)
softmax_attn = np.exp(attn_raw - attn_raw.max(axis=1, keepdims=True))
softmax_attn = softmax_attn / softmax_attn.sum(axis=1, keepdims=True)

def to_canvas(pts, xr, yr):
    p = pts.copy()
    for d in range(2):
        mn, mx = p[:, d].min(), p[:, d].max()
        if mx > mn:
            p[:, d] = (p[:, d] - mn) / (mx - mn)
    p[:, 0] = p[:, 0] * (xr[1] - xr[0]) + xr[0]
    p[:, 1] = p[:, 1] * (yr[1] - yr[0]) + yr[0]
    return p

bg = '#0d1117'
even_col = '#4fc3f7'
odd_col = '#ffb74d'
plus_col = '#69f0ae'

fig = plt.figure(figsize=(16, 9), facecolor=bg)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')
ax.set_facecolor(bg)

# --- Background radial glow ---
for cx, cy, r, col in [(2, 3, 3.5, plus_col), (12, 6, 4, even_col), (8, 4.5, 5, '#6a1b9a')]:
    for k in range(15):
        frac = k / 15
        ax.add_patch(Circle((cx, cy), r * (1 - frac * 0.6),
                     facecolor=col, alpha=0.008 * (1 - frac), edgecolor='none'))

# --- Attention web ---
Q_c = to_canvas(Q_all, (0.5, 15.5), (1.8, 8.2))
K_c = to_canvas(K_all, (0.5, 15.5), (1.8, 8.2))

lines, line_colors = [], []
for i in range(len(Q_all)):
    top_keys = np.argsort(softmax_attn[i])[-2:]
    for j in top_keys:
        w = softmax_attn[i, j]
        if w > 0.08:
            lines.append([Q_c[i], K_c[j]])
            if labels_all[i] == '+':
                line_colors.append((0.41, 0.94, 0.68, w * 0.6))
            else:
                line_colors.append((0.39, 0.71, 0.96, w * 0.2))

ax.add_collection(LineCollection(lines, colors=line_colors, linewidths=0.4))

# --- Q/K scatter constellation ---
for i in range(len(Q_all)):
    a = 0.06
    ax.plot(Q_c[i, 0], Q_c[i, 1], '.', color=even_col, markersize=1.5, alpha=a)
    ax.plot(K_c[i, 0], K_c[i, 1], '.', color='#ef5350', markersize=1.5, alpha=a)

# --- Token embedding orbs ---
tok_c = to_canvas(token_emb, (1.5, 14.5), (2.5, 7.5))

for t in range(vocab_size):
    label = itos[t]
    cx, cy = tok_c[t]
    is_even = label != '+' and isinstance(label, str) and label.lstrip('-').isdigit() and int(label) % 2 == 0
    is_plus = label == '+'

    if is_plus:
        color, glow, tsz = plus_col, 0.8, 24
    elif is_even:
        color, glow, tsz = even_col, 0.55, 18
    else:
        color, glow, tsz = odd_col, 0.4, 15

    # Multi-layer glow
    for k in range(10):
        frac = k / 10
        ax.add_patch(Circle((cx, cy), glow * (1 - frac * 0.7),
                     facecolor=color, alpha=0.03 * (1 - frac), edgecolor='none', zorder=4))

    # Core
    ax.add_patch(Circle((cx, cy), 0.18, facecolor=color, alpha=0.85,
                        edgecolor='white', linewidth=0.8, zorder=5))

    # Label
    ax.text(cx, cy, label, fontsize=tsz, fontweight='bold',
            ha='center', va='center', color='white', zorder=6,
            path_effects=[pe.withStroke(linewidth=3, foreground=color)])

# --- Bottom rule strip ---
# Dark overlay
ax.add_patch(Rectangle((0, 0), 16, 1.55, facecolor=bg, alpha=0.92, edgecolor='none', zorder=7))
ax.plot([0.5, 15.5], [1.55, 1.55], color='#222222', linewidth=0.6, zorder=7.5)

strip_y = 0.7
example = [
    ('5', '#666666', '#161b22'), ('3', '#666666', '#161b22'),
    ('8', '#666666', '#161b22'), ('7', '#666666', '#161b22'),
    ('+', plus_col, '#0d2818'),  ('8', even_col, '#0d1926'),
    ('10', '#666666', '#161b22'), ('2', '#666666', '#161b22'),
    ('4', '#666666', '#161b22'),
    ('+', plus_col, '#0d2818'),  ('4', even_col, '#0d1926'),
]
ew, egap = 0.56, 0.04
total_ew = len(example) * (ew + egap) - egap
ex0 = 8 - total_ew / 2

for i, (tok, tcol, bgcol) in enumerate(example):
    x = ex0 + i * (ew + egap) + ew / 2
    ax.add_patch(FancyBboxPatch((x - ew/2, strip_y - 0.24), ew, 0.48,
                 boxstyle="round,pad=0.05", facecolor=bgcol,
                 edgecolor='#333333', linewidth=0.8, zorder=8))
    ax.text(x, strip_y, tok, fontsize=13, fontweight='bold',
            ha='center', va='center', color=tcol, zorder=9)

def _tc(idx):
    return ex0 + idx * (ew + egap) + ew / 2

for src, dst, txt in [(2, 5, 'last even = 8'), (8, 10, 'last even = 4')]:
    ax.annotate('', xy=(_tc(dst), strip_y + 0.29), xytext=(_tc(src), strip_y + 0.29),
                arrowprops=dict(arrowstyle='->', lw=1.2, color=plus_col,
                               connectionstyle='arc3,rad=-0.28', alpha=0.65), zorder=10)
    ax.text((_tc(src) + _tc(dst)) / 2, strip_y + 0.62, txt,
            fontsize=9, ha='center', va='bottom', color=plus_col,
            fontstyle='italic', alpha=0.7, zorder=10)

ax.text(ex0 - 0.4, strip_y, 'Rule:', fontsize=11, fontweight='bold',
        ha='right', va='center', color='#777777', zorder=9)

plt.savefig('plus_last_even/plots/seq_1/00_cover_slide.png',
            dpi=250, bbox_inches='tight', facecolor=bg, edgecolor='none',
            pad_inches=0)
print("Cover slide saved.")
plt.close()
