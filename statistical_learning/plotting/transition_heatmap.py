"""Empirical char->char transition heatmap vs ground-truth DFA edges."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from statistical_learning.dfa import DFA


def _ground_truth_allowed(dfa: DFA, alphabet: list[str]) -> set[tuple[str, str]]:
    """Return the set of (prev_char, next_char) pairs allowed by the DFA.

    A pair is allowed if there exists some DFA state s such that
    delta(s, prev_char) is defined AND delta(delta(s, prev_char), next_char)
    is defined. That captures: 'after seeing prev_char, can next_char legally
    follow?'.
    """
    allowed: set[tuple[str, str]] = set()
    for (s, ch1), s1 in dfa.transitions.items():
        for ch2 in alphabet:
            if (s1, ch2) in dfa.transitions:
                allowed.add((ch1, ch2))
    return allowed


def plot_transition_heatmap(
    transitions: dict,
    dfa: DFA,
    save_path: Path,
) -> None:
    alphabet: list[str] = list(transitions["alphabet"])
    counts = np.array(transitions["counts"], dtype=float)
    row_sums = counts.sum(axis=1, keepdims=True)
    probs = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums > 0)

    allowed = _ground_truth_allowed(dfa, alphabet)

    fig, ax = plt.subplots(figsize=(max(5, 0.55 * len(alphabet) + 2), max(4, 0.55 * len(alphabet) + 2)))
    im = ax.imshow(probs, cmap="Blues", vmin=0, vmax=1, aspect="equal")

    for i, ch1 in enumerate(alphabet):
        for j, ch2 in enumerate(alphabet):
            p = probs[i, j]
            if p > 0.01:
                ax.text(
                    j,
                    i,
                    f"{p:.2f}",
                    ha="center",
                    va="center",
                    color="white" if p > 0.55 else "black",
                    fontsize=9,
                )
            if (ch1, ch2) in allowed:
                ax.add_patch(
                    Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="#1f9d55",
                        linewidth=2.0,
                    )
                )

    ax.set_xticks(range(len(alphabet)))
    ax.set_xticklabels(alphabet)
    ax.set_yticks(range(len(alphabet)))
    ax.set_yticklabels(alphabet)
    ax.set_xlabel("Next character")
    ax.set_ylabel("Previous character")
    ax.set_title("Empirical P(next | prev) from generated text\n(green outline = allowed by ground-truth DFA)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Probability")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
