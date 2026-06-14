"""Prefix trie and prefix-automaton (DFA) over a regime vocabulary.

Both structures are keyed by *prefix strings* so the ground-truth structure
(from the regime) and the empirical structure (from Transformer samples) can be
compared edge-by-edge.

State = a prefix of some word. The start state is the empty prefix "". Reading a
character moves to the next prefix. A full-word (terminal) state restarts a new
word: directly (no separator) it re-enters the first-character states; with
``word_space=True`` it instead takes a space edge back to start.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
# Trie
# --------------------------------------------------------------------------- #
@dataclass
class Trie:
    """Prefix trie keyed by prefix strings."""

    words: list[str]
    nodes: set[str] = field(default_factory=set)            # all prefixes incl. ""
    terminals: set[str] = field(default_factory=set)        # full words
    edges: dict[tuple[str, str], str] = field(default_factory=dict)  # (prefix, ch) -> prefix

    @classmethod
    def from_words(cls, words: list[str]) -> "Trie":
        trie = cls(words=list(words))
        trie.nodes.add("")
        for w in words:
            for i in range(len(w)):
                parent = w[:i]
                child = w[: i + 1]
                trie.nodes.add(parent)
                trie.nodes.add(child)
                trie.edges[(parent, w[i])] = child
            trie.terminals.add(w)
        return trie


# --------------------------------------------------------------------------- #
# DFA / prefix automaton
# --------------------------------------------------------------------------- #
@dataclass
class PrefixDFA:
    words: list[str]
    word_space: bool
    states: list[str] = field(default_factory=list)          # prefixes (start = "")
    start: str = ""
    accepting: set[str] = field(default_factory=set)          # full-word states
    transitions: dict[tuple[str, str], str] = field(default_factory=dict)

    @classmethod
    def from_words(cls, words: list[str], word_space: bool = False) -> "PrefixDFA":
        trie = Trie.from_words(words)
        dfa = cls(words=list(words), word_space=word_space)
        dfa.states = sorted(trie.nodes, key=lambda s: (len(s), s))
        dfa.accepting = set(trie.terminals)
        # In-word transitions follow the trie.
        dfa.transitions.update(trie.edges)
        # Restart transitions out of terminal (full-word) states.
        first_chars = sorted({w[0] for w in words})
        for w in words:
            if word_space:
                dfa.transitions[(w, " ")] = ""
            else:
                for ch in first_chars:
                    dfa.transitions[(w, ch)] = ch
        return dfa

    def legal_next(self, state: str) -> list[str]:
        return sorted(ch for (s, ch) in self.transitions if s == state)

    def step(self, state: str, ch: str) -> str | None:
        return self.transitions.get((state, ch))

    def run(self, stream: str) -> tuple[list[int], list[str]]:
        """Replay a stream. Returns (per-char legality 1/0, list of visited states)."""
        legality: list[int] = []
        visited: list[str] = []
        state = self.start
        for ch in stream:
            visited.append(state)
            nxt = self.step(state, ch)
            if nxt is None:
                legality.append(0)
                state = self.start
            else:
                legality.append(1)
                state = nxt
        return legality, visited

    def prefix_state_sequence(self, stream: str) -> list[str]:
        """State occupied *before* reading each character (for coloring tokens)."""
        _, visited = self.run(stream)
        return visited


# --------------------------------------------------------------------------- #
# Segmentation of generated text into words
# --------------------------------------------------------------------------- #
def segment_words(text: str, word_space: bool, word_len: int) -> list[str]:
    """Split a generated character stream into candidate words."""
    if word_space:
        return [tok for tok in text.split(" ") if tok != ""]
    return [text[i : i + word_len] for i in range(0, len(text) - word_len + 1, word_len)]


def evaluate_words(
    generated_words: list[str], vocabulary: list[str]
) -> dict:
    """Count valid vs invalid generated words and per-word frequencies."""
    vocab_set = set(vocabulary)
    counts = Counter(generated_words)
    valid = {w: counts.get(w, 0) for w in vocabulary}
    invalid = {w: c for w, c in counts.items() if w not in vocab_set}
    n_valid = sum(c for w, c in counts.items() if w in vocab_set)
    n_invalid = sum(c for w, c in counts.items() if w not in vocab_set)
    total = n_valid + n_invalid
    return {
        "total_words": total,
        "valid_count": n_valid,
        "invalid_count": n_invalid,
        "validity_rate": (n_valid / total) if total else 0.0,
        "valid_frequencies": valid,
        "invalid_frequencies": dict(sorted(invalid.items(), key=lambda kv: -kv[1])),
    }


# --------------------------------------------------------------------------- #
# Comparison: true vs generated structure
# --------------------------------------------------------------------------- #
def classify_edges(
    true_edges: set[tuple[str, str]],
    gen_edges: set[tuple[str, str]],
) -> dict[tuple[str, str], str]:
    """Label every edge as 'both', 'missing' (true only) or 'extra' (gen only)."""
    labels: dict[tuple[str, str], str] = {}
    for e in true_edges | gen_edges:
        in_true = e in true_edges
        in_gen = e in gen_edges
        if in_true and in_gen:
            labels[e] = "both"
        elif in_true:
            labels[e] = "missing"
        else:
            labels[e] = "extra"
    return labels
