"""Build and persist trie / DFA / vocab artifacts."""
from __future__ import annotations

import json
from pathlib import Path

from statistical_learning.dfa import build_dfa, dfa_to_dict, dict_to_dfa
from statistical_learning.encoder import build_alphabet, build_char_encoder
from statistical_learning.trie import build_trie, trie_to_dict
from statistical_learning.word_sets import describe_vocabulary, validate_vocabulary

PACKAGE_DIR = Path(__file__).resolve().parent


def get_runs_dir() -> Path:
    return PACKAGE_DIR / "runs"


def get_artifacts_dir(config_name: str) -> Path:
    return get_runs_dir() / config_name / "artifacts"


def build_and_save_all(config: dict) -> Path:
    name = config["name"]
    words = list(config["words"])

    validate_vocabulary(words)
    vocabulary_type = (
        config.get("vocabulary_type") or config.get("complexity") or describe_vocabulary(words)
    )

    root = build_trie(words)
    dfa = build_dfa(words)
    alphabet = build_alphabet(words)

    out_dir = get_artifacts_dir(name)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "trie.json", "w", encoding="utf-8") as f:
        json.dump(trie_to_dict(root, words), f, indent=2)

    with open(out_dir / "dfa.json", "w", encoding="utf-8") as f:
        json.dump(dfa_to_dict(dfa, words), f, indent=2)

    with open(out_dir / "vocab.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "alphabet": alphabet,
                "words": words,
                "vocabulary_type": vocabulary_type,
            },
            f,
            indent=2,
        )

    metadata = {
        "name": name,
        "vocabulary_type": vocabulary_type,
        "words": words,
        "alphabet": alphabet,
        "num_states": len(dfa.states),
        "num_accepting": len(dfa.accepting),
    }
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Artifacts saved to {out_dir}")
    return out_dir


def load_artifacts(config_name: str) -> tuple[dict, dict, dict]:
    out_dir = get_artifacts_dir(config_name)
    with open(out_dir / "trie.json", encoding="utf-8") as f:
        trie_data = json.load(f)
    with open(out_dir / "dfa.json", encoding="utf-8") as f:
        dfa_data = json.load(f)
    with open(out_dir / "vocab.json", encoding="utf-8") as f:
        vocab_data = json.load(f)
    return trie_data, dfa_data, vocab_data


def load_dfa_from_artifacts(config_name: str):
    _, dfa_data, _ = load_artifacts(config_name)
    dfa, words = dict_to_dfa(dfa_data)
    return dfa, words
