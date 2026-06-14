"""Generate plots + learned-grammar artifacts from a trained checkpoint."""
from __future__ import annotations

import json
from pathlib import Path

from statistical_learning.artifacts import (
    get_artifacts_dir,
    get_runs_dir,
    load_dfa_from_artifacts,
)
from statistical_learning.checkpoint import get_checkpoint_dir
from statistical_learning.encoder import build_alphabet
from statistical_learning.generator import WordCorpusGenerator
from statistical_learning.learned_grammar import save_learned_grammar_artifacts
from statistical_learning.plotting.learning_curve import plot_learning_curve
from statistical_learning.plotting.sequence_heatmap import (
    plot_generated_sequences_heatmap,
)
from statistical_learning.plotting.transition_heatmap import plot_transition_heatmap
from statistical_learning.plotting.trie_diagram import plot_trie
from statistical_learning.plotting.word_frequencies import plot_word_frequencies
from statistical_learning.sample import generate_sequences, write_samples_txt
from statistical_learning.trie import build_trie


def get_plots_dir(config_name: str, step: int | None = None) -> Path:
    base = get_runs_dir() / config_name / "plots"
    return base / f"step_{step:06d}" if step is not None else base


def visualize_from_checkpoint(
    config: dict,
    step: int | None = None,
    num_sequences: int = 8,
    seq_length: int = 60,
    seed: int = 18,
) -> dict:
    config_name = config["name"]
    words = list(config["words"])
    alphabet = build_alphabet(words)
    start_chars = sorted({w[0] for w in words})

    plots_dir = get_plots_dir(config_name, step=step)
    plots_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing plots and learned artifacts to {plots_dir}")

    sequences, checkpoint_data = generate_sequences(
        config_name,
        num_sequences=num_sequences,
        seq_length=seq_length,
        seed=seed,
        step=step,
        start_chars=start_chars,
    )
    write_samples_txt(sequences, plots_dir / "generated_sequences.txt")

    metadata_path = get_checkpoint_dir(config_name, step) / "metadata.json"
    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)
    plot_learning_curve(
        steps=metadata["steps_for_plot"],
        train_losses=metadata["train_loss_history"],
        val_losses=metadata["val_loss_history"],
        rule_error_history=metadata.get("rule_error_history"),
        save_path=plots_dir / "learning_curve.png",
        eval_interval=metadata.get("eval_interval"),
    )

    dfa, _ = load_dfa_from_artifacts(config_name)
    generator = WordCorpusGenerator(words, dfa=dfa)
    heatmap_stats = plot_generated_sequences_heatmap(
        sequences,
        generator,
        save_path=plots_dir / "generated_sequences_heatmap.png",
        num_sequences=num_sequences,
        max_length=min(seq_length, 50),
    )

    summary = save_learned_grammar_artifacts(
        sequences,
        ground_truth_words=words,
        alphabet=alphabet,
        out_dir=plots_dir,
    )

    plot_trie(
        build_trie(words),
        plots_dir / "trie.png",
        title=f"Ground-truth vocabulary trie: {words}",
    )
    plot_trie(
        build_trie(summary["learned_words"]),
        plots_dir / "learned_trie.png",
        title="Learned trie (words the model produced)",
    )
    freqs_path = plots_dir / "word_frequencies.json"
    with open(freqs_path, encoding="utf-8") as f:
        freqs = json.load(f)
    plot_word_frequencies(freqs, plots_dir / "word_frequencies.png")

    transitions_path = plots_dir / "transitions.json"
    with open(transitions_path, encoding="utf-8") as f:
        transitions = json.load(f)
    plot_transition_heatmap(transitions, dfa, plots_dir / "char_transitions.png")

    full_summary = {**summary, "heatmap_stats": heatmap_stats, "config_name": config_name}
    with open(plots_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(full_summary, f, indent=2)
    _print_summary(full_summary)
    return full_summary


def _print_summary(s: dict) -> None:
    print("\n========== visualization summary ==========")
    print(f"config:                {s['config_name']}")
    print(f"sequences generated:   {s['num_sequences']}")
    print(f"total words observed:  {s['total_words_generated']}")
    print(
        f"valid / invalid:       {s['valid_word_count']} / {s['invalid_word_count']}"
    )
    print(f"missing ground-truth:  {s['missing_words']}")
    print(f"spurious words:        {s['spurious_words']}")
    hs = s["heatmap_stats"]
    print(
        f"constrained positions: {hs['constrained_correct']} correct / "
        f"{hs['constrained_incorrect']} wrong "
        f"(accuracy={hs['constrained_accuracy']:.3f})"
    )
    print("============================================\n")
