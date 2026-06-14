"""Character-level statistical-learning experiment on overlapping word regimes.

Trains the shared Transformer (model.py) on a long character stream produced by
uniformly sampling words from a fixed regime vocabulary, then probes whether the
model recovers the finite-state (trie / DFA) structure of that language.
"""
