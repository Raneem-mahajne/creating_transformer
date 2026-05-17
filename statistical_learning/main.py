"""Entry point for statistical-learning experiments."""
from __future__ import annotations

import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

PACKAGE_DIR = Path(__file__).resolve().parent
ROOT = PACKAGE_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import get_batch_from_sequences, split_train_val_sequences
from model import BigramLanguageModel
from training import estimate_loss, estimate_rule_error

from statistical_learning.artifacts import build_and_save_all, load_dfa_from_artifacts
from statistical_learning.checkpoint import get_checkpoint_dir, load_checkpoint, save_checkpoint
from statistical_learning.config_loader import load_config
from statistical_learning.dfa import build_dfa
from statistical_learning.encoder import build_alphabet, build_char_encoder
from statistical_learning.generator import WordCorpusGenerator
from statistical_learning.plotting.export_dot import export_dfa_dot, export_trie_dot
from statistical_learning.trie import build_trie


def train(config: dict, force_retrain: bool = False) -> None:
    config_name = config["name"]
    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    words = list(config["words"])
    delimiter = config.get("delimiter", "|")

    checkpoint_data = None
    if not force_retrain:
        checkpoint_data = load_checkpoint(config_name, step=None)

    if checkpoint_data is not None:
        print("Using existing checkpoint. Pass --force-retrain to retrain.")
        return

    checkpoint_base = get_checkpoint_dir(config_name)
    if checkpoint_base.exists() and force_retrain:
        print(f"Deleting old checkpoints from {checkpoint_base}...")
        shutil.rmtree(checkpoint_base)

    artifacts_dir = build_and_save_all(config)
    dfa, _ = load_dfa_from_artifacts(config_name)

    root = build_trie(words)
    export_trie_dot(root, artifacts_dir / "trie.dot")
    export_dfa_dot(dfa, artifacts_dir / "dfa.dot")

    generator = WordCorpusGenerator(words, delimiter, dfa=dfa)
    sequences = generator.generate_dataset(
        num_sequences=data_config["num_sequences"],
        min_length=data_config["min_length"],
        max_length=data_config["max_length"],
    )
    print(f"Generated {len(sequences)} sequences")

    alphabet = build_alphabet(words, delimiter)
    encode, decode, vocab_size, itos, stoi = build_char_encoder(alphabet)
    print("Vocabulary size:", vocab_size)
    print("Alphabet:", alphabet)

    encoded_sequences = [encode(seq) for seq in sequences]
    train_sequences, val_sequences = split_train_val_sequences(encoded_sequences, train_ratio=0.9)
    print(f"Train: {len(train_sequences)} sequences, Val: {len(val_sequences)} sequences")

    n_embd = model_config["n_embd"]
    block_size = model_config["block_size"]
    num_heads = model_config["num_heads"]
    head_size = model_config["head_size"]
    use_residual = model_config.get("use_residual", True)

    model = BigramLanguageModel(
        vocab_size, n_embd, block_size, num_heads, head_size, use_residual=use_residual
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config["learning_rate"])

    batch_size = training_config["batch_size"]
    max_steps = training_config["max_steps"]
    eval_interval = training_config["eval_interval"]
    eval_iterations = training_config["eval_iterations"]
    checkpoint_interval = training_config.get("checkpoint_interval", 100)

    steps_for_plot = []
    train_loss_history = []
    val_loss_history = []
    rule_error_history = []

    seed = 18
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    for step in range(max_steps):
        if step % eval_interval == 0:
            losses = estimate_loss(
                model, train_sequences, val_sequences, block_size, batch_size, eval_iterations
            )
            rule_err = estimate_rule_error(
                model, generator, decode, block_size, num_samples=20, seq_length=30
            )
            steps_for_plot.append(step)
            train_loss_history.append(losses["train"])
            val_loss_history.append(losses["validation"])
            rule_error_history.append(rule_err)
            print(
                f"step {step}: train loss {losses['train']:.4f}, "
                f"val loss {losses['validation']:.4f}, rule err {rule_err:.4f}",
                flush=True,
            )

        if checkpoint_interval > 0 and step > 0 and step % checkpoint_interval == 0:
            save_checkpoint(
                config_name,
                model,
                train_sequences,
                val_sequences,
                itos,
                stoi,
                vocab_size,
                steps_for_plot,
                train_loss_history,
                val_loss_history,
                rule_error_history,
                model_config,
                eval_interval,
                step=step,
            )

        X, Y = get_batch_from_sequences(train_sequences, block_size, batch_size)
        _, loss = model(X, Y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print("Final loss:", loss.item(), flush=True)
    if rule_error_history:
        print(f"Final rule error: {rule_error_history[-1]:.4f}", flush=True)

    save_checkpoint(
        config_name,
        model,
        train_sequences,
        val_sequences,
        itos,
        stoi,
        vocab_size,
        steps_for_plot,
        train_loss_history,
        val_loss_history,
        rule_error_history,
        model_config,
        eval_interval,
        step=None,
    )


def artifacts_only(config: dict) -> None:
    artifacts_dir = build_and_save_all(config)
    words = list(config["words"])
    delimiter = config.get("delimiter", "|")
    root = build_trie(words)
    dfa = build_dfa(words, delimiter)
    export_trie_dot(root, artifacts_dir / "trie.dot")
    export_dfa_dot(dfa, artifacts_dir / "dfa.dot")
    print(f"DOT files: {artifacts_dir / 'trie.dot'}, {artifacts_dir / 'dfa.dot'}")


def main() -> None:
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print(
            "Usage: python -m statistical_learning.main <config_name> [--artifacts-only] [--force-retrain]\n"
            "  config_name: one_word | disjoint_letters | shared_letters"
        )
        sys.exit(0 if argv and argv[0] in ("-h", "--help") else 1)

    config_name = argv[0]
    artifacts_only_flag = "--artifacts-only" in argv
    force_retrain = "--force-retrain" in argv

    config = load_config(config_name)
    if "--max-steps" in argv:
        idx = argv.index("--max-steps")
        if idx + 1 < len(argv):
            config["training"]["max_steps"] = int(argv[idx + 1])
    print(f"Loaded config: {config['name']} (complexity={config['complexity']})")

    if artifacts_only_flag:
        artifacts_only(config)
    else:
        train(config, force_retrain=force_retrain)


if __name__ == "__main__":
    main()
