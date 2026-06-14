"""Generate character sequences from a trained statistical-learning checkpoint."""
from __future__ import annotations

import random
from typing import List

import numpy as np
import torch

from statistical_learning.checkpoint import load_checkpoint


def generate_sequences(
    config_name: str,
    num_sequences: int = 5,
    seq_length: int = 60,
    seed: int | None = 18,
    step: int | None = None,
    start_chars: list[str] | None = None,
) -> tuple[list[list[str]], dict]:
    """Load checkpoint and sample `num_sequences` character sequences from the model.

    Returns (sequences, checkpoint_data). Each sequence is a list of single-char strings.
    If `start_chars` is given (e.g. the first characters of vocabulary words), each
    sequence is conditioned on a random one so generation begins at a word boundary.
    """
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    checkpoint_data = load_checkpoint(config_name, step=step)
    if checkpoint_data is None:
        raise FileNotFoundError(
            f"No checkpoint found for {config_name}"
            + (f" at step {step}" if step is not None else "")
        )

    model = checkpoint_data["model"]
    decode = checkpoint_data["decode"]
    stoi = checkpoint_data["stoi"]
    vocab_size = checkpoint_data["vocab_size"]

    valid_starts = [stoi[c] for c in (start_chars or []) if c in stoi]

    sequences: List[List[str]] = []
    for _ in range(num_sequences):
        if valid_starts:
            start_id = random.choice(valid_starts)
        else:
            start_id = random.randint(0, vocab_size - 1)
        start = torch.tensor([[start_id]], dtype=torch.long)
        with torch.no_grad():
            out = model.generate(start, max_new_tokens=seq_length - 1)[0].tolist()
        sequences.append(decode(out))

    return sequences, checkpoint_data


def write_samples_txt(sequences: list[list[str]], save_path) -> None:
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"# Generated sequences ({len(sequences)} samples)\n")
        f.write("# Each line is one sequence; characters joined verbatim.\n\n")
        for i, seq in enumerate(sequences):
            f.write(f"[{i:02d}] {''.join(seq)}\n")
