"""Plots for the character-level statistical-learning experiment.

Renders are matplotlib-only (no Graphviz). Trie and DFA share a prefix-tree node
layout; the DFA additionally draws curved restart edges. Comparison variants
colour each edge by whether it is present in both the ground-truth and the
Transformer-derived structure, only in the ground truth (missing), or only in
the samples (extra / invalid).
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch

from statistical_learning.charlm.automata import PrefixDFA, Trie

# Edge colour scheme for comparisons.
_LABEL_STYLE = {
    "both": ("#2e7d32", "-", 2.2),      # present in both: green solid
    "missing": ("#9e9e9e", "--", 1.6),  # in ground truth only: gray dashed
    "extra": ("#d32f2f", "-", 2.2),     # in samples only: red solid
    "plain": ("#666666", "-", 1.6),     # single-structure rendering
}


# --------------------------------------------------------------------------- #
# Shared prefix-tree layout
# --------------------------------------------------------------------------- #
def _layout(nodes: set[str], word_edges: dict[tuple[str, str], str]) -> dict[str, tuple[float, float]]:
    children: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for (parent, ch), child in word_edges.items():
        children[parent].append((ch, child))

    pos: dict[str, tuple[float, float]] = {}
    leaf = [0.0]

    def assign(node: str) -> float:
        kids = sorted(children.get(node, []))
        if not kids:
            x = leaf[0]
            leaf[0] += 1.0
        else:
            xs = [assign(child) for _, child in kids]
            x = sum(xs) / len(xs)
        pos[node] = (x, -float(len(node)))
        return x

    # Cover every node even if disconnected from "".
    assign("")
    for n in sorted(nodes, key=lambda s: (len(s), s)):
        if n not in pos:
            pos[n] = (leaf[0], -float(len(n)))
            leaf[0] += 1.0
    return pos


def _node_face(node: str, terminals: set[str], root: str = "") -> str:
    if node == root:
        return "#d3d3d3"
    if node in terminals:
        return "#90EE90"
    return "white"


# --------------------------------------------------------------------------- #
# Trie
# --------------------------------------------------------------------------- #
def plot_trie(trie: Trie, save_path: Path, title: str = "Word trie") -> None:
    edge_labels = {e: "plain" for e in trie.edges}
    _draw_trie(trie.nodes, trie.edges, edge_labels, trie.terminals, save_path, title)


def plot_trie_comparison(
    true_trie: Trie,
    gen_trie: Trie,
    save_path: Path,
    title: str = "Trie comparison (true vs Transformer samples)",
) -> None:
    from statistical_learning.charlm.automata import classify_edges

    labels = classify_edges(set(true_trie.edges), set(gen_trie.edges))
    nodes = true_trie.nodes | gen_trie.nodes
    all_edges: dict[tuple[str, str], str] = {**true_trie.edges, **gen_trie.edges}
    terminals = true_trie.terminals | gen_trie.terminals
    _draw_trie(nodes, all_edges, labels, terminals, save_path, title, comparison=True)


def _draw_trie(
    nodes: set[str],
    edges: dict[tuple[str, str], str],
    edge_labels: dict[tuple[str, str], str],
    terminals: set[str],
    save_path: Path,
    title: str,
    comparison: bool = False,
) -> None:
    pos = _layout(nodes, edges)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    width = max(7.0, (max(xs) - min(xs) + 1) * 1.5)
    height = max(3.5, (max(ys) - min(ys) + 1) * 1.5)
    fig, ax = plt.subplots(figsize=(width, height))

    for (parent, ch), child in edges.items():
        color, ls, lw = _LABEL_STYLE[edge_labels.get((parent, ch), "plain")]
        x0, y0 = pos[parent]
        x1, y1 = pos[child]
        ax.plot([x0, x1], [y0, y1], color=color, lw=lw, ls=ls, zorder=1)
        ax.text(
            (x0 + x1) / 2, (y0 + y1) / 2, ch,
            fontsize=12, fontweight="bold", color="#1f4e79",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"), zorder=3,
        )

    for node, (x, y) in pos.items():
        face = _node_face(node, terminals)
        ax.scatter([x], [y], s=480, facecolors=face, edgecolors="#333333",
                   linewidths=1.5, zorder=2)
        if node == "":
            ax.text(x, y, "ε", fontsize=10, ha="center", va="center", zorder=4)
        elif node in terminals:
            ax.text(x, y - 0.30, node, fontsize=9, fontweight="bold",
                    color="#1b5e20", ha="center", va="top", zorder=4)

    ax.set_title(title)
    ax.axis("off")
    ax.margins(0.12)
    if comparison:
        ax.legend(handles=_comparison_legend(), loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# --------------------------------------------------------------------------- #
# DFA / prefix automaton
# --------------------------------------------------------------------------- #
def _split_dfa_edges(dfa: PrefixDFA):
    """Separate advancing word edges from restart edges."""
    word_edges: dict[tuple[str, str], str] = {}
    restart_edges: dict[tuple[str, str], str] = {}
    for (s, ch), t in dfa.transitions.items():
        if t == s + ch and len(t) == len(s) + 1:
            word_edges[(s, ch)] = t
        else:
            restart_edges[(s, ch)] = t
    return word_edges, restart_edges


def plot_dfa(dfa: PrefixDFA, save_path: Path, title: str = "Prefix automaton (DFA)") -> None:
    word_edges, restart_edges = _split_dfa_edges(dfa)
    labels = {e: "plain" for e in {**word_edges, **restart_edges}}
    _draw_dfa(set(dfa.states), word_edges, restart_edges, labels, set(dfa.accepting),
              save_path, title)


def plot_dfa_comparison(
    true_dfa: PrefixDFA,
    gen_dfa: PrefixDFA,
    save_path: Path,
    title: str = "DFA comparison (true vs Transformer samples)",
) -> None:
    from statistical_learning.charlm.automata import classify_edges

    labels = classify_edges(set(true_dfa.transitions), set(gen_dfa.transitions))
    merged = {**true_dfa.transitions, **gen_dfa.transitions}
    word_edges: dict[tuple[str, str], str] = {}
    restart_edges: dict[tuple[str, str], str] = {}
    for (s, ch), t in merged.items():
        if t == s + ch and len(t) == len(s) + 1:
            word_edges[(s, ch)] = t
        else:
            restart_edges[(s, ch)] = t
    nodes = set(true_dfa.states) | set(gen_dfa.states)
    accepting = set(true_dfa.accepting) | set(gen_dfa.accepting)
    _draw_dfa(nodes, word_edges, restart_edges, labels, accepting, save_path, title,
              comparison=True)


def _draw_dfa(
    nodes: set[str],
    word_edges: dict[tuple[str, str], str],
    restart_edges: dict[tuple[str, str], str],
    edge_labels: dict[tuple[str, str], str],
    accepting: set[str],
    save_path: Path,
    title: str,
    comparison: bool = False,
) -> None:
    pos = _layout(nodes, word_edges)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    width = max(8.0, (max(xs) - min(xs) + 1) * 1.7)
    height = max(4.0, (max(ys) - min(ys) + 1) * 1.7)
    fig, ax = plt.subplots(figsize=(width, height))

    for (s, ch), t in word_edges.items():
        color, ls, lw = _LABEL_STYLE[edge_labels.get((s, ch), "plain")]
        x0, y0 = pos[s]
        x1, y1 = pos[t]
        ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls,
                            shrinkA=12, shrinkB=12), zorder=1,
        )
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, ch, fontsize=12, fontweight="bold",
                color="#1f4e79", ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"), zorder=3)

    # Restart edges (terminal -> start / first char): curved to reduce clutter.
    for (s, ch), t in restart_edges.items():
        color, ls, lw = _LABEL_STYLE[edge_labels.get((s, ch), "plain")]
        x0, y0 = pos[s]
        x1, y1 = pos[t]
        rad = 0.35 if x1 >= x0 else -0.35
        arrow = FancyArrowPatch(
            (x0, y0), (x1, y1), connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>", color=color, lw=lw * 0.8, ls=ls,
            alpha=0.55, mutation_scale=12, shrinkA=12, shrinkB=12, zorder=0,
        )
        ax.add_patch(arrow)
        label = "␣" if ch == " " else ch
        ax.text(x0 + (x1 - x0) * 0.5, y0 + (y1 - y0) * 0.5 - 0.25 * np.sign(rad),
                label, fontsize=8, color=color, ha="center", va="center", alpha=0.8,
                zorder=2)

    for node, (x, y) in pos.items():
        face = "#d3d3d3" if node == "" else ("#90EE90" if node in accepting else "white")
        ax.scatter([x], [y], s=520, facecolors=face, edgecolors="#333333",
                   linewidths=1.5, zorder=2)
        label = "ε" if node == "" else node
        ax.text(x, y + 0.0, label, fontsize=9, ha="center", va="center",
                fontweight="bold" if node in accepting else "normal", zorder=4)

    ax.set_title(title)
    ax.axis("off")
    ax.margins(0.14)
    handles = _comparison_legend() if comparison else [
        Patch(facecolor="#90EE90", edgecolor="#333", label="accepting (full word)"),
        Patch(facecolor="#d3d3d3", edgecolor="#333", label="start (ε)"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def _comparison_legend():
    return [
        Line2D([0], [0], color="#2e7d32", lw=2.2, label="present in both"),
        Line2D([0], [0], color="#9e9e9e", lw=1.6, ls="--", label="missing from samples"),
        Line2D([0], [0], color="#d32f2f", lw=2.2, label="extra / invalid in samples"),
    ]


# --------------------------------------------------------------------------- #
# Attention heatmaps
# --------------------------------------------------------------------------- #
def plot_attention(
    chars: list[str],
    attn_heads: list[np.ndarray],
    save_path: Path,
    title: str = "Self-attention (last layer)",
) -> None:
    """attn_heads: list of (T, T) matrices, one per head."""
    n = len(attn_heads)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.2 * rows), squeeze=False)
    T = len(chars)
    ticks = list(range(T))
    for h, A in enumerate(attn_heads):
        ax = axes[h // cols][h % cols]
        im = ax.imshow(A, cmap="viridis", vmin=0, vmax=1, aspect="equal")
        ax.set_title(f"head {h}", fontsize=10)
        ax.set_xticks(ticks)
        ax.set_xticklabels([c if c != " " else "␣" for c in chars], fontsize=6)
        ax.set_yticks(ticks)
        ax.set_yticklabels([c if c != " " else "␣" for c in chars], fontsize=6)
        ax.set_xlabel("attended-to (key)", fontsize=8)
        ax.set_ylabel("query", fontsize=8)
    for k in range(n, rows * cols):
        axes[k // cols][k % cols].axis("off")
    fig.suptitle(title, fontsize=12)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="attention weight")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# --------------------------------------------------------------------------- #
# Hidden states (raw 2-D, no dimensionality reduction)
# --------------------------------------------------------------------------- #
def _scatter_by_label(ax, coords: np.ndarray, labels: list[str], subtitle: str) -> None:
    categories = sorted(set(labels))
    cmap = plt.get_cmap("tab20" if len(categories) > 10 else "tab10")
    cat_to_color = {c: cmap(i % cmap.N) for i, c in enumerate(categories)}
    for c in categories:
        idx = [i for i, l in enumerate(labels) if l == c]
        disp = "ε" if c == "" else ("␣" if c == " " else c)
        ax.scatter(coords[idx, 0], coords[idx, 1], s=14, alpha=0.7,
                   color=cat_to_color[c], label=disp)
    ax.set_title(subtitle, fontsize=11)
    ax.set_xlabel("dim 0")
    ax.set_ylabel("dim 1")
    ncol = 2 if len(categories) > 12 else 1
    ax.legend(loc="best", fontsize=7, ncol=ncol, framealpha=0.8)


def plot_hidden_2d(
    coords_in: np.ndarray,
    coords_out: np.ndarray,
    current_chars: list[str],
    next_chars: list[str],
    states: list[str],
    save_path: Path,
    title: str = "Hidden states (2-D)",
) -> None:
    """Plot the raw 2-D representations directly (n_embd == 2, so no PCA).

    Top row: input embeddings (token + position) fed into the transformer.
    Bottom row: output hidden states fed to the lm_head.
    """
    panels = [
        ("colored by current character", current_chars),
        ("colored by next target character", next_chars),
        ("colored by prefix / DFA state", states),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    for ax, (subtitle, labels) in zip(axes[0], panels):
        _scatter_by_label(ax, coords_in, labels, "INPUT — " + subtitle)
    for ax, (subtitle, labels) in zip(axes[1], panels):
        _scatter_by_label(ax, coords_out, labels, "OUTPUT — " + subtitle)
    fig.suptitle(title, fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# --------------------------------------------------------------------------- #
# Word frequencies
# --------------------------------------------------------------------------- #
def plot_word_frequencies(evaluation: dict, save_path: Path, max_invalid: int = 20) -> None:
    valid = evaluation["valid_frequencies"]
    invalid = list(evaluation["invalid_frequencies"].items())[:max_invalid]

    labels = list(valid.keys()) + [w for w, _ in invalid]
    counts = list(valid.values()) + [c for _, c in invalid]
    colors = ["#90EE90"] * len(valid) + ["#ff6b6b"] * len(invalid)

    fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(labels) + 2), 4.8))
    x = np.arange(len(labels))
    bars = ax.bar(x, counts, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Occurrences in Transformer samples")
    rate = 100.0 * evaluation["validity_rate"]
    ax.set_title(
        f"Generated word frequencies  (valid {evaluation['valid_count']}/"
        f"{evaluation['total_words']} = {rate:.1f}%; invalid types in red)"
    )
    ax.grid(True, axis="y", alpha=0.3)
    for bar, c in zip(bars, counts):
        if c > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(c),
                    ha="center", va="bottom", fontsize=8)
    ax.legend(handles=[
        Patch(facecolor="#90EE90", label="valid (in vocabulary)"),
        Patch(facecolor="#ff6b6b", label="invalid (spurious)"),
    ], loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
