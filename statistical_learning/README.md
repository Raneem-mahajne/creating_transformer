# Statistical learning (isolated module)

Self-contained word-corpus experiments with **trie** and **DFA** ground-truth artifacts. Does not modify the integer-task pipeline (`main.py`, `configs/`, `plus_last_even/`, etc.).

## Complexity regimes

| Config file | `complexity` | Word set rule | Example |
|-------------|--------------|---------------|---------|
| `one_word.yaml` | `one_word` | Single word | `cat` |
| `disjoint_letters.yaml` | `disjoint_letters` | No letter appears in more than one word | `cat`, `mop`, `red` |
| `shared_letters.yaml` | `shared_letters` | At least one letter shared between some pair | `cat`, `hat`, `map` |

Sequences are character-level: sample a word uniformly, emit its characters, append delimiter `|`, repeat.

## Commands

From the repository root:

```bash
# Build trie.json, dfa.json, vocab.json, metadata.json (+ .dot)
python -m statistical_learning.main one_word --artifacts-only
python -m statistical_learning.main disjoint_letters --artifacts-only
python -m statistical_learning.main shared_letters --artifacts-only

# Train (reuses model.py / training.py from parent package)
python -m statistical_learning.main disjoint_letters --force-retrain

# Optional: cap training steps (e.g. smoke test)
python -m statistical_learning.main disjoint_letters --force-retrain --max-steps 400
```

## Outputs

All outputs live under `statistical_learning/runs/{config_name}/`:

```
runs/stat_disjoint_letters/
  artifacts/
    trie.json
    dfa.json
    vocab.json
    metadata.json
    trie.dot
    dfa.dot
  checkpoints/
```

## Artifact schemas

**trie.json** — `words` list plus nested `root.children` keyed by character; terminal nodes include `word` and `word_id`.

**dfa.json** — `states`, `start`, `accepting`, `alphabet`, `transitions` as `"state,char": next_state`, `delimiter`, and `state_metadata`.

**vocab.json** — `alphabet`, `delimiter`, `words`, `complexity`.

## Dependencies

Same as the parent project (`torch`, `pyyaml`, etc. in `requirements.txt`).
