"""Run the Q dot product gradients plot in an interactive pyplot window.
   Use this to tweak layout/parameters and see the figure update.

   Command (from repo root):
     python scripts/run_q_dot_product_gradients_interactive.py
"""
import sys
from pathlib import Path

# Run from repo root so imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from config_loader import load_config
from checkpoint import load_checkpoint
from plotting import plot_q_dot_product_gradients

# Same sequence used by visualize for consistency
FIXED_SEQUENCE_DECODED = [4, 9, "+", 4, 10, 7, "+", 10]


def main():
    config = load_config("plus_last_even")
    checkpoint_data = load_checkpoint(config["name"], step=None)
    if checkpoint_data is None:
        raise SystemExit("No checkpoint found for plus_last_even. Train or ensure checkpoints exist.")

    model = checkpoint_data["model"]
    itos = checkpoint_data["itos"]
    stoi = checkpoint_data["stoi"]
    block_size = checkpoint_data["model_config"]["block_size"]

    # Encode fixed sequence to token ids
    consistent_sequence = [stoi[str(t)] for t in FIXED_SEQUENCE_DECODED]
    X = torch.tensor(consistent_sequence[:block_size], dtype=torch.long).unsqueeze(0)
    X_list = [X]

    # save_path=None => figure pops up; you can resize and inspect
    plot_q_dot_product_gradients(model, X_list, itos, save_path=None, num_sequences=1)


if __name__ == "__main__":
    main()
