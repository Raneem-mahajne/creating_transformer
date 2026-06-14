"""Bar chart of valid (ground-truth) and invalid words observed in generation."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_word_frequencies(
    freqs: dict,
    save_path: Path,
    max_invalid: int = 15,
) -> None:
    valid = freqs["valid"]
    invalid = freqs["invalid"]
    total = freqs["total_words"] or 1

    valid_items = list(valid.items())
    invalid_items = sorted(invalid.items(), key=lambda kv: -kv[1])[:max_invalid]

    labels = [w for w, _ in valid_items] + [w for w, _ in invalid_items]
    counts = [c for _, c in valid_items] + [c for _, c in invalid_items]
    colors = ["#90EE90"] * len(valid_items) + ["#ff6b6b"] * len(invalid_items)

    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(labels) + 2), 4.5))
    x = np.arange(len(labels))
    bars = ax.bar(x, counts, color=colors, edgecolor="black", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Occurrences across generated sequences")
    invalid_total = sum(invalid.values())
    valid_total = sum(valid.values())
    pct_valid = 100.0 * valid_total / total
    ax.set_title(
        f"Word frequencies in generation  "
        f"(valid={valid_total}/{total} = {pct_valid:.1f}%, invalid types shown ≤{max_invalid})"
    )
    ax.grid(True, axis="y", alpha=0.3)
    for bar, c in zip(bars, counts):
        if c > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                str(c),
                ha="center",
                va="bottom",
                fontsize=9,
            )

    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor="#90EE90", label="Ground-truth word"),
            Patch(facecolor="#ff6b6b", label="Invalid (spurious) word"),
        ],
        loc="best",
        fontsize=9,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
