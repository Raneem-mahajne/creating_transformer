"""Plot training/validation loss and rule error from checkpoint metadata."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_learning_curve(
    steps: list[int],
    train_losses: list[float],
    val_losses: list[float],
    rule_error_history: list[float] | None,
    save_path: Path,
    eval_interval: int | None = None,
) -> None:
    fig, ax1 = plt.subplots(figsize=(11, 5))

    ax1.set_xlabel(
        f"Training step{f' (eval every {eval_interval})' if eval_interval else ''}",
        fontsize=11,
    )
    ax1.set_ylabel("Cross-entropy loss", color="tab:blue", fontsize=11)
    l1 = ax1.plot(steps, train_losses, label="Train loss", color="tab:blue", linewidth=2)
    l2 = ax1.plot(steps, val_losses, label="Val loss", color="tab:orange", linewidth=2)
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True, alpha=0.3)

    lines = list(l1) + list(l2)
    title = "Learning curve: loss"
    if rule_error_history:
        ax2 = ax1.twinx()
        ax2.set_ylabel(
            "Rule error (fraction of constrained positions wrong)",
            color="tab:red",
            fontsize=11,
        )
        l3 = ax2.plot(
            steps,
            rule_error_history,
            label="Rule error",
            color="tab:red",
            linewidth=2,
            linestyle="--",
        )
        ax2.tick_params(axis="y", labelcolor="tab:red")
        ax2.set_ylim(0, max(1.05, max(rule_error_history) * 1.1))
        lines += list(l3)
        title = "Learning curve: loss and rule error"

    ax1.legend(lines, [l.get_label() for l in lines], loc="best")
    plt.title(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
