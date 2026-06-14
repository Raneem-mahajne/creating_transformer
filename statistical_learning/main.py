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
from statistical_learning.word_sets import describe_vocabulary, validate_vocabulary
from statistical_learning.generator import WordCorpusGenerator
from statistical_learning.plotting.export_dot import export_dfa_dot, export_trie_dot
from statistical_learning.trie import build_trie
from statistical_learning.visualize import visualize_from_checkpoint


def train(config: dict, force_retrain: bool = False, run_visualize: bool = True) -> None:
    config_name = config["name"]
    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    words = list(config["words"])

    checkpoint_data = None
    if not force_retrain:
        checkpoint_data = load_checkpoint(config_name, step=None)

    if checkpoint_data is not None:
        print("Using existing checkpoint. Pass --force-retrain to retrain.")
        if run_visualize:
            visualize_from_checkpoint(config)
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

    generator = WordCorpusGenerator(words, dfa=dfa)
    sequences = generator.generate_dataset(
        num_sequences=data_config["num_sequences"],
        min_length=data_config["min_length"],
        max_length=data_config["max_length"],
    )
    print(f"Generated {len(sequences)} sequences")

    alphabet = build_alphabet(words)
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
    n_layer = model_config.get("n_layer", 1)
    use_layernorm = model_config.get("use_layernorm", False)

    model = BigramLanguageModel(
        vocab_size, n_embd, block_size, num_heads, head_size,
        use_residual=use_residual, n_layer=n_layer, use_layernorm=use_layernorm,
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

    if run_visualize:
        visualize_from_checkpoint(config)


def artifacts_only(config: dict) -> None:
    artifacts_dir = build_and_save_all(config)
    words = list(config["words"])
    root = build_trie(words)
    dfa = build_dfa(words)
    export_trie_dot(root, artifacts_dir / "trie.dot")
    export_dfa_dot(dfa, artifacts_dir / "dfa.dot")
    print(f"DOT files: {artifacts_dir / 'trie.dot'}, {artifacts_dir / 'dfa.dot'}")


DEFAULT_DATA = {"num_sequences": 2000, "min_length": 40, "max_length": 120}
DEFAULT_MODEL = {
    "n_embd": 16,
    "block_size": 16,
    "num_heads": 4,
    "head_size": 4,
    "n_layer": 2,
    "use_layernorm": True,
    "use_residual": True,
}
DEFAULT_TRAINING = {
    "max_steps": 10000,
    "batch_size": 8,
    "learning_rate": 0.001,
    "eval_interval": 200,
    "eval_iterations": 50,
    "checkpoint_interval": 100,
}


def build_inline_config(name: str, words: list[str]) -> dict:
    """Build a config for an arbitrary vocabulary passed on the CLI (no YAML needed)."""
    validate_vocabulary(words)
    return {
        "name": name,
        "vocabulary_type": describe_vocabulary(words),
        "words": words,
        "data": dict(DEFAULT_DATA),
        "model": dict(DEFAULT_MODEL),
        "training": dict(DEFAULT_TRAINING),
    }


def _arg_value(argv: list[str], flag: str) -> str | None:
    if flag in argv:
        idx = argv.index(flag)
        if idx + 1 < len(argv):
            return argv[idx + 1]
    return None


def main() -> None:
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print(
            "Usage: python -m statistical_learning.main <config_name> [options]\n"
            "  config_name: one_word | disjoint_letters | shared_letters | any saved config\n"
            "Options:\n"
            "  --words a,b,c            train on an arbitrary (prefix-free) vocabulary, no YAML needed\n"
            "  --name NAME              run/output name when using --words (default: stat_custom)\n"
            "  --artifacts-only         only build trie/DFA/vocab artifacts, no training\n"
            "  --force-retrain          delete existing checkpoints and train from scratch\n"
            "  --visualize              load a trained checkpoint and produce plots only\n"
            "  --step N                 with --visualize, use checkpoint at step N (default: final)\n"
            "  --no-visualize           after training, skip the visualization pass\n"
            "  --max-steps N            cap training steps (smoke test)"
        )
        sys.exit(0 if argv and argv[0] in ("-h", "--help") else 1)

    artifacts_only_flag = "--artifacts-only" in argv
    force_retrain = "--force-retrain" in argv
    visualize_only = "--visualize" in argv
    run_visualize = "--no-visualize" not in argv

    step: int | None = None
    step_arg = _arg_value(argv, "--step")
    if step_arg is not None:
        step = int(step_arg)

    words_arg = _arg_value(argv, "--words")
    if words_arg is not None:
        words = [w.strip() for w in words_arg.split(",") if w.strip()]
        name = _arg_value(argv, "--name") or "stat_custom"
        config = build_inline_config(name, words)
    else:
        config = load_config(argv[0])

    if "--max-steps" in argv:
        max_arg = _arg_value(argv, "--max-steps")
        if max_arg is not None:
            config["training"]["max_steps"] = int(max_arg)
    print(
        f"Loaded config: {config['name']} "
        f"(vocabulary_type={config.get('vocabulary_type') or describe_vocabulary(config['words'])}, "
        f"words={config['words']})"
    )

    if artifacts_only_flag:
        artifacts_only(config)
    elif visualize_only:
        visualize_from_checkpoint(config, step=step)
    else:
        train(config, force_retrain=force_retrain, run_visualize=run_visualize)


if __name__ == "__main__":
    main()
