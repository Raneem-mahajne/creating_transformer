"""Checkpoint save/load and path helpers."""
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import torch

from model import BigramLanguageModel


def get_checkpoint_dir(config_name_actual: str, step: int | None = None) -> Path:
    base = Path(config_name_actual) / "checkpoints"
    return base / f"step_{step:06d}" if step is not None else base


def get_plots_dir(config_name_actual: str, step: int | None = None, subfolder: str | None = None) -> Path:
    base = Path(config_name_actual) / "plots"
    if step is not None:
        base = base / f"step_{step:06d}"
    if subfolder is not None:
        base = base / subfolder
    return base


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
        out = []
        for idx in token_indices:
            s = itos[idx]
            try:
                out.append(int(s))
            except ValueError:
                out.append(s)
        return out

    return decode


def save_checkpoint(
    config_name_actual: str,
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
    generated_sequences_e0=None,
):
    checkpoint_dir = get_checkpoint_dir(config_name_actual, step)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_dir / "model.pt")

    main_dir = get_checkpoint_dir(config_name_actual)
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
        "generated_sequences_e0": _to_python(generated_sequences_e0) if generated_sequences_e0 is not None else None,
    }
    with open(checkpoint_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Checkpoint saved to {checkpoint_dir}")


def load_checkpoint(config_name_actual: str, step: int | None = None) -> dict | None:
    checkpoint_dir = get_checkpoint_dir(config_name_actual, step)
    if not checkpoint_dir.exists():
        return None
    with open(checkpoint_dir / "metadata.json", "r") as f:
        metadata = json.load(f)
    main_dir = get_checkpoint_dir(config_name_actual)
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
    load_result = model.load_state_dict(state, strict=False)
    if load_result.missing_keys:
        print(f"Warning: checkpoint missing keys (e.g. ln1/ln2); retrain for LayerNorm. Missing: {load_result.missing_keys}")
    model.eval()
    step_loaded = metadata.get("step")
    print(f"Checkpoint loaded from {checkpoint_dir} (step: {step_loaded})")
    return {
        "model": model,
        "train_sequences": train_sequences,
        "val_sequences": val_sequences,
        "itos": itos,
        "stoi": stoi,
        "decode": decode,
        "vocab_size": metadata["vocab_size"],
        "step": step_loaded,
        "steps_for_plot": metadata["steps_for_plot"],
        "train_loss_history": metadata["train_loss_history"],
        "val_loss_history": metadata["val_loss_history"],
        "rule_error_history": metadata.get("rule_error_history", []),
        "generated_sequences_e0": metadata.get("generated_sequences_e0"),
        "model_config": mc,
        "eval_interval": metadata.get("eval_interval"),
    }


def list_available_checkpoints(config_name_actual: str) -> list[int]:
    base = Path(config_name_actual) / "checkpoints"
    if not base.exists():
        return []
    steps = []
    for item in base.iterdir():
        if item.is_dir() and item.name.startswith("step_"):
            try:
                steps.append(int(item.name.split("_")[1]))
            except (ValueError, IndexError):
                continue
    return sorted(steps)
