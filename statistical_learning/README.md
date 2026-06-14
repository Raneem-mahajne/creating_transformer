# Statistical learning (isolated module)

One general, vocabulary-driven character transformer. You give it a **vocabulary**
(a set of words) and it learns to emit a raw character stream that is a clean
concatenation of those words, e.g. `cathatmap...` — **no separators**. The token
vocabulary is just the set of characters in the words.

The old "complexity levels" are simply different example vocabularies fed through
the same code (still one trained model per vocabulary):

| Config file | Example vocabulary | Note |
|-------------|--------------------|------|
| `one_word.yaml` | `cat` | single word |
| `disjoint_letters.yaml` | `cat`, `mop`, `red` | words share no letters |
| `shared_letters.yaml` | `cat`, `hat`, `map` | words share letters |

`vocabulary_type` in a config is just a descriptive label; it has no behavioral effect.

## Prefix-free requirement

Because there is no separator, the vocabulary must be **prefix-free**: no word may be
a prefix of another (equal-length words always qualify). This guarantees every stream
has exactly one segmentation into words, so "is this output correct?" is well-defined.
Non-prefix-free vocabularies are rejected at config time.

## Commands

From the repository root:

```bash
# Build trie.json, dfa.json, vocab.json, metadata.json (+ .dot) for a named config
python -m statistical_learning.main shared_letters --artifacts-only

# Train a named config (reuses model.py / training.py from the parent package).
# After training it auto-runs --visualize to dump plots and the learned grammar.
python -m statistical_learning.main shared_letters --force-retrain

# Train on an ARBITRARY (prefix-free) vocabulary with no YAML needed:
python -m statistical_learning.main --words cat,hat,map --name my_vocab --force-retrain

# Cap training steps (smoke test)
python -m statistical_learning.main shared_letters --force-retrain --max-steps 400

# Only run the post-training analysis from an existing checkpoint
python -m statistical_learning.main shared_letters --visualize
python -m statistical_learning.main shared_letters --visualize --step 1000
```

## Outputs

All outputs live under `statistical_learning/runs/{name}/`:

```
runs/stat_shared_letters/
  artifacts/                          # ground-truth grammar from the vocabulary
    trie.json / trie.dot
    dfa.json  / dfa.dot               # cyclic concatenation DFA (restarts at terminals)
    vocab.json                        # alphabet, words, vocabulary_type
    metadata.json
  checkpoints/                        # per-step model weights + metadata
  plots/                              # produced by --visualize
    trie.png                           # ground-truth vocabulary trie diagram
    learned_trie.png                   # trie over the words the model produced
    generated_sequences.txt            # raw sampled text (e.g. cathatmap...)
    generated_sequences_heatmap.png    # per-token DFA legality (green/red/gray)
    learning_curve.png                 # train+val loss + rule error vs step
    word_frequencies.png               # valid vs invalid words generated
    char_transitions.png               # P(next|prev) with ground-truth edges outlined
    learned_trie.json / .dot
    learned_dfa.json  / .dot
    word_frequencies.json
    transitions.json
    learned_summary.json
    summary.json                       # missing / spurious words + heatmap stats
```

The trie diagrams (`trie.png`, `learned_trie.png`) are rendered directly with
matplotlib — no Graphviz needed. The `.dot` files can still be rendered with
Graphviz if installed, e.g. `dot -Tpng learned_dfa.dot -o learned_dfa.png`.

## How correctness is measured

The raw, separator-free stream is segmented with a prefix-free **max-munch** parse
against the ground-truth trie: walk from the root and emit a word on reaching a
terminal. A character that cannot extend the current prefix is an off-grammar
deviation (recorded as an invalid fragment); a truncated trailing run is dropped.
`summary.json` reports `missing_words`, `spurious_words`, and constrained-position
accuracy. A correct model yields `missing_words: []`, `spurious_words: []`.

## Model

The model (`model.py`, shared with the parent integer task) supports optional
`n_layer` and `use_layernorm`. The statistical-learning configs use a small pre-LN
stack (`n_embd=16, num_heads=4, head_size=4, n_layer=2, use_layernorm=true`), which
is enough to reproduce the vocabulary exactly even when words share letters. The
defaults (`n_layer=1, use_layernorm=false`) preserve the original single-layer
behavior for the parent task.

## Dependencies

Same as the parent project (`torch`, `pyyaml`, `matplotlib`, etc. in `requirements.txt`).
