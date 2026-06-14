"""Render a word trie to a PNG using matplotlib (no Graphviz dependency)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from statistical_learning.trie import TrieNode


def _layout(root: TrieNode):
    """Assign (x, y) positions: x by leaf order, y by negative depth."""
    positions: dict[int, tuple[float, float]] = {}
    terminal: dict[int, bool] = {}
    word_at: dict[int, str] = {}
    edges: list[tuple[int, int, str]] = []
    leaf_counter = [0]

    def assign(node: TrieNode, depth: int) -> float:
        nid = id(node)
        terminal[nid] = node.is_terminal
        word_at[nid] = node.word if (node.is_terminal and node.word) else ""
        children = sorted(node.children.items())
        if not children:
            x = float(leaf_counter[0])
            leaf_counter[0] += 1
        else:
            xs = []
            for ch, child in children:
                edges.append((nid, id(child), ch))
                xs.append(assign(child, depth + 1))
            x = sum(xs) / len(xs)
        positions[nid] = (x, -float(depth))
        return x

    assign(root, 0)
    return positions, terminal, word_at, edges, id(root)


def plot_trie(root: TrieNode, save_path: Path, title: str = "Word trie") -> None:
    positions, terminal, word_at, edges, root_id = _layout(root)

    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    width = max(6.0, (max(xs) - min(xs) + 1) * 1.4)
    height = max(3.0, (max(ys) - min(ys) + 1) * 1.4)
    fig, ax = plt.subplots(figsize=(width, height))

    for parent_id, child_id, ch in edges:
        x0, y0 = positions[parent_id]
        x1, y1 = positions[child_id]
        ax.plot([x0, x1], [y0, y1], color="#888888", lw=1.5, zorder=1)
        ax.text(
            (x0 + x1) / 2,
            (y0 + y1) / 2,
            ch,
            fontsize=13,
            fontweight="bold",
            color="#1f4e79",
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"),
            zorder=3,
        )

    for nid, (x, y) in positions.items():
        is_root = nid == root_id
        is_term = terminal[nid]
        if is_root:
            fc = "#d3d3d3"
        elif is_term:
            fc = "#90EE90"
        else:
            fc = "white"
        ax.scatter([x], [y], s=520, facecolors=fc, edgecolors="#333333", linewidths=1.5, zorder=2)
        if is_root:
            ax.text(x, y, "start", fontsize=8, ha="center", va="center", zorder=4)
        if is_term and word_at[nid]:
            ax.text(x, y - 0.28, word_at[nid], fontsize=10, fontweight="bold",
                    color="#1b5e20", ha="center", va="top", zorder=4)

    ax.set_title(title)
    ax.axis("off")
    ax.margins(0.1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
