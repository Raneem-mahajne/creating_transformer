"""Checkpoint save/load under statistical_learning/runs/."""
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import torch

from statistical_learning.artifacts import get_runs_dir

PACKAGE_DIR = Path(__file__).resolve().parent


def _repo_root() -> Path:
    return PACKAGE_DIR.parent


def get_run_dir(config_name: str) -> Path:
    return get_runs_dir() / config_name


def get_checkpoint_dir(config_name: str, step: int | None = None) -> Path:
    base = get_run_dir(config_name) / "checkpoints"
    return base / f"step_{step:06d}" if step is not None else base


def _to_python(obj):
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [_to_python(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    return obj


def create_decode_from_itos(itos):
    def decode(token_indices):
        return [itos[int(idx)] for idx in token_indices]

    return decode


def save_checkpoint(
    config_name: str,
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
    eval_interval=None,
    step=None,
):
    checkpoint_dir = get_checkpoint_dir(config_name, step)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_dir / "model.pt")

    main_dir = get_checkpoint_dir(config_name)
    if not (main_dir / "train_sequences.pkl").exists():
        main_dir.mkdir(parents=True, exist_ok=True)
        for name, data in [
            ("train_sequences.pkl", train_sequences),
            ("val_sequences.pkl", val_sequences),
            ("itos.pkl", itos),
            ("stoi.pkl", stoi),
        ]:
            with open(main_dir / name, "wb") as f:
                pickle.dump(data, f)

    if step is not None:
        for name in ["train_sequences.pkl", "val_sequences.pkl", "itos.pkl", "stoi.pkl"]:
            if (main_dir / name).exists() and not (checkpoint_dir / name).exists():
                shutil.copy2(main_dir / name, checkpoint_dir / name)

    metadata = {
        "vocab_size": vocab_size,
        "step": step,
        "steps_for_plot": _to_python(steps_for_plot),
        "train_loss_history": _to_python(train_loss_history),
        "val_loss_history": _to_python(val_loss_history),
        "rule_error_history": _to_python(rule_error_history),
        "model_config": model_config,
        "eval_interval": eval_interval,
    }
    with open(checkpoint_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Checkpoint saved to {checkpoint_dir}")


def load_checkpoint(config_name: str, step: int | None = None) -> dict | None:
    import sys

    root = str(_repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    from model import BigramLanguageModel

    checkpoint_dir = get_checkpoint_dir(config_name, step)
    meta_path = checkpoint_dir / "metadata.json"
    if not meta_path.exists():
        return None

    with open(meta_path, encoding="utf-8") as f:
        metadata = json.load(f)

    main_dir = get_checkpoint_dir(config_name)
    for name in ["train_sequences.pkl", "val_sequences.pkl", "itos.pkl", "stoi.pkl"]:
        if not (checkpoint_dir / name).exists() and (main_dir / name).exists():
            shutil.copy2(main_dir / name, checkpoint_dir / name)

    with open(checkpoint_dir / "train_sequences.pkl", "rb") as f:
        train_sequences = pickle.load(f)
    with open(checkpoint_dir / "val_sequences.pkl", "rb") as f:
        val_sequences = pickle.load(f)
    with open(checkpoint_dir / "itos.pkl", "rb") as f:
        itos = pickle.load(f)
    with open(checkpoint_dir / "stoi.pkl", "rb") as f:
        stoi = pickle.load(f)

    decode = create_decode_from_itos(itos)
    mc = metadata["model_config"]
    model = BigramLanguageModel(
        vocab_size=metadata["vocab_size"],
        n_embd=mc["n_embd"],
        block_size=mc["block_size"],
        num_heads=mc["num_heads"],
        head_size=mc["head_size"],
        use_residual=mc.get("use_residual", True),
    )
    state = torch.load(checkpoint_dir / "model.pt", weights_only=True)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"Checkpoint loaded from {checkpoint_dir}")
    return {
        "model": model,
        "train_sequences": train_sequences,
        "val_sequences": val_sequences,
        "itos": itos,
        "stoi": stoi,
        "decode": decode,
        "vocab_size": metadata["vocab_size"],
        "model_config": mc,
    }
