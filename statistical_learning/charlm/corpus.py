"""Corpus generation for the character-level statistical-learning task.

A corpus is one long character stream produced by repeatedly sampling a word
uniformly at random from a regime and appending its characters. With
``word_space=True`` a single space separates consecutive words.
"""
from __future__ import annotations

import random
from pathlib import Path


def generate_corpus(
    words: list[str],
    num_chars: int,
    word_space: bool = False,
    seed: int | None = None,
) -> str:
    """Build a character stream of (at least) ``num_chars`` characters.

    Words are sampled uniformly and concatenated. With ``word_space`` a single
    space is appended after every word. Generation stops once the requested
    length is reached (the stream is then trimmed to exactly ``num_chars``).
    """
    if seed is not None:
        random.seed(seed)
    out: list[str] = []
    while len(out) < num_chars:
        w = random.choice(words)
        out.extend(w)
        if word_space:
            out.append(" ")
    return "".join(out[:num_chars])


def build_char_vocab(corpus: str) -> tuple[list[str], dict[str, int], dict[int, str]]:
    """Character vocabulary = sorted set of characters appearing in the corpus."""
    alphabet = sorted(set(corpus))
    stoi = {ch: i for i, ch in enumerate(alphabet)}
    itos = {i: ch for i, ch in enumerate(alphabet)}
    return alphabet, stoi, itos


def save_corpus(corpus: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(corpus, encoding="utf-8")


def encode(corpus: str, stoi: dict[str, int]) -> list[int]:
    return [stoi[ch] for ch in corpus]


def decode(ids: list[int], itos: dict[int, str]) -> str:
    return "".join(itos[int(i)] for i in ids)
