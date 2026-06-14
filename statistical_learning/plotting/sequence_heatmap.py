"""Color generated character sequences by DFA correctness."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from statistical_learning.generator import WordCorpusGenerator


def plot_generated_sequences_heatmap(
    sequences: list[list[str]],
    generator: WordCorpusGenerator,
    save_path: Path,
    num_sequences: int = 5,
    max_length: int = 40,
    title: str = "Generated sequences (green = legal DFA step, red = violation, gray = unconstrained)",
) -> dict:
    seqs = sequences[:num_sequences]
    max_len = min(max_length, max(len(s) for s in seqs))

    data: list[list[str | None]] = []
    correctness: list[list[float]] = []
    constrained: list[list[float]] = []

    for seq in seqs:
        s = list(seq[:max_len])
        c, _ = generator.verify_sequence(s)
        v = generator.valence_mask(s)
        while len(s) < max_len:
            s.append(None)
            c.append(np.nan)
            v.append(np.nan)
        data.append(s)
        correctness.append([float(x) if x is not None else np.nan for x in c])
        constrained.append([float(x) if x is not None else np.nan for x in v])

    arr_data = np.array(data, dtype=object)
    arr_c = np.array(correctness, dtype=float)
    arr_v = np.array(constrained, dtype=float)

    cmap = ListedColormap(["#ff6b6b", "#90EE90", "#d3d3d3"])
    cmap.set_bad(color=(1, 1, 1, 0))
    color_idx = np.where(
        np.isnan(arr_c),
        np.nan,
        np.where(arr_v == 0.0, 2.0, arr_c),
    )
    masked = np.ma.masked_invalid(color_idx)

    fig_h = max(2.5, num_sequences * 0.7 + 1.5)
    fig_w = max(10, max_len * 0.45)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(masked, cmap=cmap, aspect="auto", vmin=0, vmax=2)

    for i in range(len(seqs)):
        for j in range(max_len):
            v = arr_data[i, j]
            if v is not None:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=11, fontweight="bold")

    ax.set_xlabel("Position in sequence")
    ax.set_ylabel("Sequence")
    ax.set_title(title)
    ax.set_xticks(range(0, max_len, max(1, max_len // 20)))
    ax.set_yticks(range(len(seqs)))
    ax.set_yticklabels([f"#{i}" for i in range(len(seqs))])

    legend = [
        Patch(facecolor="#90EE90", label="Legal"),
        Patch(facecolor="#ff6b6b", label="Illegal"),
        Patch(facecolor="#d3d3d3", label="Unconstrained (>1 legal char)"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(bottom=0.18)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    total_correct = int(np.sum((arr_c == 1) & (arr_v == 1)))
    total_incorrect = int(np.sum((arr_c == 0) & (arr_v == 1)))
    total = total_correct + total_incorrect
    return {
        "constrained_total": total,
        "constrained_correct": total_correct,
        "constrained_incorrect": total_incorrect,
        "constrained_accuracy": (total_correct / total) if total > 0 else float("nan"),
    }
