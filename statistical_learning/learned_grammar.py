"""Build the *learned* trie/DFA/transition graph from generated sequences.

Mirrors the ground-truth artifacts in `artifacts/` (trie.json, dfa.json) but
computed from what the transformer produced rather than the rule definition.

There is no separator, so the raw character stream is segmented with a
prefix-free max-munch parse against the ground-truth trie: walk from the root,
and on reaching a terminal emit the completed word and restart. A character
that cannot extend the current prefix is an off-grammar deviation: the
accumulated run (including that character) is recorded as an invalid fragment
and the parser resyncs at the root. A trailing run that never reaches a
terminal is an incomplete fragment and is dropped.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from statistical_learning.dfa import DFA, dfa_from_trie, dfa_to_dict
from statistical_learning.plotting.export_dot import export_dfa_dot, export_trie_dot
from statistical_learning.trie import TrieNode, build_trie, trie_to_dict


def parse_stream(stream: list[str], root: TrieNode) -> tuple[list[str], list[str]]:
    """Max-munch parse a separator-free stream into (valid_words, invalid_fragments)."""
    valid: list[str] = []
    invalid: list[str] = []
    node = root
    buf: list[str] = []
    for ch in stream:
        if ch in node.children:
            node = node.children[ch]
            buf.append(ch)
            if node.is_terminal:
                valid.append("".join(buf))
                buf = []
                node = root
        else:
            buf.append(ch)
            invalid.append("".join(buf))
            buf = []
            node = root
    # Leftover buf is an incomplete trailing fragment (truncated by length): dropped.
    return valid, invalid


def parse_sequences(
    sequences: list[list[str]], root: TrieNode
) -> tuple[list[str], list[str]]:
    all_valid: list[str] = []
    all_invalid: list[str] = []
    for seq in sequences:
        v, inv = parse_stream(seq, root)
        all_valid.extend(v)
        all_invalid.extend(inv)
    return all_valid, all_invalid


def word_frequencies(
    valid_words: list[str], invalid_fragments: list[str], ground_truth_words: list[str]
) -> dict:
    """Return per-word counts split into valid (in ground truth) vs invalid."""
    valid_counts = Counter(valid_words)
    invalid_counts = Counter(invalid_fragments)
    valid = {w: valid_counts.get(w, 0) for w in ground_truth_words}
    invalid = dict(invalid_counts)
    total = sum(valid_counts.values()) + sum(invalid_counts.values())
    return {
        "total_words": total,
        "valid": valid,
        "invalid": invalid,
    }


def char_transition_counts(
    sequences: list[list[str]], alphabet: list[str]
) -> dict:
    """First-order char->char transition counts across all generated sequences."""
    idx = {ch: i for i, ch in enumerate(alphabet)}
    n = len(alphabet)
    counts = [[0] * n for _ in range(n)]
    for seq in sequences:
        for prev, nxt in zip(seq, seq[1:]):
            if prev in idx and nxt in idx:
                counts[idx[prev]][idx[nxt]] += 1
    return {"alphabet": alphabet, "counts": counts}


def build_learned_trie_and_dfa(
    learned_words: list[str],
) -> tuple[TrieNode, DFA]:
    """Build a trie from the *unique* observed words and derive a DFA."""
    root = build_trie(learned_words)
    dfa = dfa_from_trie(root, learned_words)
    return root, dfa


def save_learned_grammar_artifacts(
    sequences: list[list[str]],
    ground_truth_words: list[str],
    alphabet: list[str],
    out_dir: Path,
) -> dict:
    """Write learned_trie.{json,dot}, learned_dfa.{json,dot}, transitions.json, summary.json."""
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_root = build_trie(ground_truth_words)
    valid_words, invalid_fragments = parse_sequences(sequences, gt_root)
    learned_words = sorted(set(valid_words) | set(invalid_fragments))

    root, dfa = build_learned_trie_and_dfa(learned_words)
    with open(out_dir / "learned_trie.json", "w", encoding="utf-8") as f:
        json.dump(trie_to_dict(root, learned_words), f, indent=2)
    with open(out_dir / "learned_dfa.json", "w", encoding="utf-8") as f:
        json.dump(dfa_to_dict(dfa, learned_words), f, indent=2)
    export_trie_dot(root, out_dir / "learned_trie.dot")
    export_dfa_dot(dfa, out_dir / "learned_dfa.dot")

    freqs = word_frequencies(valid_words, invalid_fragments, ground_truth_words)
    transitions = char_transition_counts(sequences, alphabet)
    with open(out_dir / "word_frequencies.json", "w", encoding="utf-8") as f:
        json.dump(freqs, f, indent=2)
    with open(out_dir / "transitions.json", "w", encoding="utf-8") as f:
        json.dump(transitions, f, indent=2)

    gt_set = set(ground_truth_words)
    learned_set = set(learned_words)
    summary = {
        "num_sequences": len(sequences),
        "ground_truth_words": ground_truth_words,
        "learned_words": learned_words,
        "missing_words": sorted(gt_set - learned_set),
        "spurious_words": sorted(learned_set - gt_set),
        "total_words_generated": freqs["total_words"],
        "valid_word_count": sum(freqs["valid"].values()),
        "invalid_word_count": sum(freqs["invalid"].values()),
    }
    with open(out_dir / "learned_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary
