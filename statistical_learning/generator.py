"""Corpus generator: repeated words separated by a delimiter."""
from __future__ import annotations

import random

from statistical_learning.dfa import DFA, build_dfa


class WordCorpusGenerator:
    def __init__(self, words: list[str], delimiter: str = "|", dfa: DFA | None = None):
        self.words = list(words)
        self.delimiter = delimiter
        self.dfa = dfa if dfa is not None else build_dfa(words, delimiter)

    def _legal_chars(self, state: int) -> list[str]:
        return sorted(ch for (s, ch) in self.dfa.transitions if s == state)

    def _transition(self, state: int, ch: str) -> int | None:
        return self.dfa.transitions.get((state, ch))

    def generate_sequence(self, length: int) -> list[str]:
        if length == 0:
            return []
        out: list[str] = []
        while len(out) < length:
            w = random.choice(self.words)
            for ch in w:
                if len(out) >= length:
                    return out
                out.append(ch)
            if len(out) < length:
                out.append(self.delimiter)
        return out

    def generate_dataset(
        self,
        num_sequences: int,
        min_length: int = 40,
        max_length: int = 120,
    ) -> list[list[str]]:
        sequences = []
        print(f"Generating {num_sequences} sequences...", end="", flush=True)
        for i in range(num_sequences):
            length = random.randint(min_length, max_length)
            sequences.append(self.generate_sequence(length))
            if (i + 1) % 1000 == 0:
                print(f" {i + 1}...", end="", flush=True)
        print(" done")
        return sequences

    def verify_sequence(self, sequence: list[str]) -> tuple[list[int], bool]:
        if len(sequence) == 0:
            return [], True
        correctness: list[int] = []
        state = self.dfa.start
        for ch in sequence:
            nxt = self._transition(state, ch)
            if nxt is None:
                correctness.append(0)
                state = self.dfa.start
            else:
                correctness.append(1)
                state = nxt
        return correctness, all(c == 1 for c in correctness)

    def valence_mask(self, sequence: list[str]) -> list[bool]:
        """True where the token at that position was the only legal choice given the prefix."""
        if len(sequence) == 0:
            return []
        mask: list[bool] = []
        state = self.dfa.start
        for ch in sequence:
            legal = self._legal_chars(state)
            mask.append(len(legal) == 1)
            nxt = self._transition(state, ch)
            state = nxt if nxt is not None else self.dfa.start
        return mask
