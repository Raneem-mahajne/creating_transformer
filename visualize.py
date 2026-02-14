"""Visualization from checkpoints."""
import csv
import os
import random
import torch

from config_loader import get_generator_from_config
from checkpoint import (
    load_checkpoint,
    get_plots_dir,
    create_decode_from_itos,
    list_available_checkpoints,
)
from data import get_batch_from_sequences
from plotting import (
    plot_training_data_heatmap,
    plot_learning_curve,
    plot_generated_sequences_heatmap_before_after,
    plot_generated_sequences_heatmap,
    plot_architecture_diagram,
    plot_weights_qkv_two_sequences,
    plot_embeddings_pca,
    plot_embeddings_scatterplots_only,
    plot_embedding_qkv_comprehensive,
    plot_qkv_transformations,
    plot_token_position_embedding_space,
    plot_attention_matrix,
    plot_qk_embedding_space,
    plot_qk_embedding_space_focused_query,
    plot_sequence_embeddings,
    plot_qk_full_attention_heatmap,
    plot_lm_head_probability_heatmaps,
    plot_v_before_after_demo_sequences,
    plot_final_on_output_heatmap_grid,
    plot_residuals,
    plot_probability_heatmap_with_embeddings,
    plot_probability_heatmap_with_values,
)


def visualize_from_checkpoint(
    config_name_actual: str, checkpoint_data: dict, config: dict, step: int = None
):
    """Generate all visualizations from checkpoint data."""
    model = checkpoint_data["model"]
    train_sequences = checkpoint_data["train_sequences"]
    itos = checkpoint_data["itos"]
    vocab_size = checkpoint_data["vocab_size"]
    steps_for_plot = checkpoint_data["steps_for_plot"]
    train_loss_history = checkpoint_data["train_loss_history"]
    val_loss_history = checkpoint_data["val_loss_history"]
    rule_error_history = checkpoint_data.get("rule_error_history", [])
    model_config = checkpoint_data["model_config"]
    eval_interval = checkpoint_data.get("eval_interval", 200)

    if "decode" in checkpoint_data:
        decode = checkpoint_data["decode"]
    else:
        decode = create_decode_from_itos(itos)

    block_size = model_config["block_size"]
    data_config = config["data"]
    training_config = config.get("training", {})

    plots_dir = get_plots_dir(config_name_actual, step)
    plots_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = str(plots_dir)

    generator = get_generator_from_config(config)

    decoded_train_sequences = [decode(seq) for seq in train_sequences[:6]]
    manifest = _load_plots_manifest()

    def _plot_path(default_name: str, base_dir: str = plots_dir) -> str:
        mapped = _lookup_manifest_filename(manifest, config_name_actual, default_name)
        return os.path.join(base_dir, mapped)
    with open(os.path.join(plots_dir, "training_data_samples.txt"), "w", encoding="utf-8") as f:
        f.write(f"# Training Data Samples for: {config_name_actual}\n")
        f.write(f"# {len(decoded_train_sequences)} sample sequences (decoded from train_sequences)\n")
        f.write("# Format: space-separated tokens, one sequence per line\n\n")
        for seq in decoded_train_sequences:
            f.write(" ".join(str(i) for i in seq) + "\n")

    num_sequences_to_generate = 5
    generated_sequences = []
    for _ in range(num_sequences_to_generate):
        seq_length = random.randint(data_config["min_length"], data_config["max_length"])
        start_token = random.randint(0, vocab_size - 1)
        start = torch.tensor([[start_token]], dtype=torch.long)
        sample = model.generate(start, max_new_tokens=seq_length - 1)[0].tolist()
        generated_sequences.append(decode(sample))

    generated_sequences_e0 = checkpoint_data.get("generated_sequences_e0") or []
    with open(os.path.join(plots_dir, "generated_integer_sequence.txt"), "w", encoding="utf-8") as f:
        if generated_sequences_e0:
            f.write("E0\n")
            for seq in generated_sequences_e0[:5]:
                f.write(" ".join(str(i) for i in seq) + "\n")
            f.write("\n")
        f.write("Final\n")
        for seq in generated_sequences:
            f.write(" ".join(str(i) for i in seq) + "\n")
    print(f"Generated {num_sequences_to_generate} sequences for step {step}")

    plot_learning_curve(
        steps_for_plot,
        train_loss_history,
        val_loss_history,
        rule_error_history=rule_error_history,
        save_path=_plot_path("learning_curve.png"),
        eval_interval=eval_interval,
    )

    _train_acc, _train_correct, _train_incorrect = plot_training_data_heatmap(
        decoded_train_sequences,
        generator,
        save_path=_plot_path("training_data_heatmap.png"),
        num_sequences=min(4, len(decoded_train_sequences)),
        max_length=20,
    )

    if generated_sequences_e0:
        (acc0, c0, i0), (accf, cf, inf) = plot_generated_sequences_heatmap_before_after(
            generated_sequences_e0,
            generated_sequences,
            generator,
            save_path=_plot_path("generated_sequences_heatmap.png"),
            num_sequences=3,
            max_length=20,
        )
        print(f"Generated sequences heatmap (E0): {c0} correct, {i0} incorrect positions ({acc0:.1%} accuracy)")
        print(f"Generated sequences heatmap (Final): {cf} correct, {inf} incorrect positions ({accf:.1%} accuracy)")
    else:
        heatmap_accuracy, correct_count, incorrect_count = plot_generated_sequences_heatmap(
            generated_sequences,
            generator,
            save_path=_plot_path("generated_sequences_heatmap.png"),
            num_sequences=min(3, len(generated_sequences)),
            max_length=20,
        )
        print(f"Generated sequences heatmap: {correct_count} correct, {incorrect_count} incorrect positions ({heatmap_accuracy:.1%} accuracy)")

    # Select a single consistent sequence for all sequence-dependent plots
    # Use a fixed index to always select the same sequence for reproducibility
    random.seed(43)  # Changed seed to get a different sequence
    
    # Find sequences that are long enough (at least 2 tokens)
    valid_sequences = [seq for seq in train_sequences if len(seq) >= 2]
    
    if valid_sequences:
        # Use a different index (1) to get a different sequence
        consistent_sequence = valid_sequences[1] if len(valid_sequences) > 1 else valid_sequences[0]
    elif train_sequences:
        # Fallback: use first sequence even if short
        consistent_sequence = train_sequences[0]
    else:
        consistent_sequence = []
    
    # Print which sequence is being used for consistency
    decoded_consistent = decode(consistent_sequence)
    print(f"Using consistent sequence for all sequence-dependent plots: {' '.join(str(t) for t in decoded_consistent)}")
    
    # Convert to tensor format for model input
    # Pad or truncate to block_size
    seq_tensor = torch.tensor(consistent_sequence[:block_size], dtype=torch.long).unsqueeze(0)
    
    # Use the same sequence for all sequence-dependent plots
    X_consistent = seq_tensor
    X_list = [X_consistent, X_consistent, X_consistent]  # Use same sequence 3 times for multi-sequence plots

    base_plots_dir = str(get_plots_dir(config_name_actual))
    arch_path = _plot_path("architecture.png", base_dir=base_plots_dir)
    if not os.path.exists(arch_path):
        plot_architecture_diagram(config, save_path=arch_path, model=model, vocab_size=vocab_size, batch_size=training_config.get('batch_size', 4))

    plot_weights_qkv_two_sequences(
        model, X_list, itos, save_path=_plot_path("qkv_query_key_attention.png"), num_sequences=1
    )
    plot_residuals(
        model, X_list, itos, save_path=_plot_path("residuals.png"), num_sequences=1
    )
    plot_embeddings_pca(model, itos, save_path=_plot_path("embeddings.png"))
    # plot_embeddings_scatterplots_only removed — redundant with bottom row of embeddings.png (Figure 05)
    plot_embedding_qkv_comprehensive(model, itos, save_path=_plot_path("embedding_qkv_comprehensive.png"))
    plot_qkv_transformations(model, itos, save_path=_plot_path("qkv_transformations.png"))
    plot_token_position_embedding_space(
        model, itos, save_path=_plot_path("token_position_embedding_space.png")
    )
    # plot_attention_matrix moved to supplementary
    # plot_attention_matrix(
    #     model, X_list, itos, save_path=_plot_path("attention_matrix.png"), num_sequences=3
    # )
    plot_qk_embedding_space(model, itos, save_path=_plot_path("qk_embedding_space.png"))
    plot_qk_embedding_space_focused_query(
        model, itos, token_str="+", position=5,
        save_path=_plot_path("qk_embedding_space_plus5_focus.png"),
    )
    # Plot embeddings for a specific sequence (using consistent sequence)
    plot_sequence_embeddings(
        model, X_consistent, itos, save_path=_plot_path("sequence_embeddings.png")
    )
    plot_qk_full_attention_heatmap(
        model, itos, save_path=_plot_path("qk_full_attention_heatmap.png")
    )
    plot_lm_head_probability_heatmaps(
        model, itos, save_path=_plot_path("lm_head_probability_heatmaps.png")
    )
    plot_probability_heatmap_with_embeddings(
        model, itos, save_path=_plot_path("probability_heatmap_with_embeddings.png")
    )
    plot_probability_heatmap_with_values(
        model, itos, save_path=_plot_path("probability_heatmap_with_values.png")
    )
    # Final-on-output heatmap grid (one figure, same sequence, output units in grid)
    if consistent_sequence:
        plot_final_on_output_heatmap_grid(
            model, itos, consistent_sequence, save_path=_plot_path("final_on_output_heatmap_grid.png")
        )

    print(f"All visualizations saved to {plots_dir}")


def visualize_all_checkpoints(config_name_actual: str, config: dict):
    """Visualize all available checkpoints for a config."""
    steps = list_available_checkpoints(config_name_actual)
    if not steps:
        print(f"No step checkpoints found for {config_name_actual}")
        return
    print(f"Found {len(steps)} checkpoint steps: {steps}")
    for step in steps:
        print(f"Visualizing checkpoint at step {step}...")
        checkpoint_data = load_checkpoint(config_name_actual, step=step)
        if checkpoint_data:
            visualize_from_checkpoint(config_name_actual, checkpoint_data, config, step=step)
        else:
            print(f"Warning: Could not load checkpoint at step {step}")


def _load_plots_manifest() -> list[dict]:
    manifest_path = os.path.join(os.path.dirname(__file__), "plots_manifest.csv")
    if not os.path.exists(manifest_path):
        return []
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader if row.get("config") and row.get("default_name") and row.get("filename")]


def _lookup_manifest_filename(manifest: list[dict], config_name: str, default_name: str) -> str:
    for row in manifest:
        if row["config"] == config_name and row["default_name"] == default_name:
            return row["filename"]
    return default_name


def _rename_demo_outputs(plots_dir: str, manifest: list[dict], config_name: str) -> None:
    demo_defaults = [
        "v_before_after_demo_0.png",
        "v_before_after_demo_1.png",
        "v_before_after_demo_2.png",
    ]
    for default_name in demo_defaults:
        src = os.path.join(plots_dir, default_name)
        if not os.path.exists(src):
            continue
        dst_name = _lookup_manifest_filename(manifest, config_name, default_name)
        if dst_name == default_name:
            continue
        dst = os.path.join(plots_dir, dst_name)
        if os.path.exists(dst):
            os.remove(dst)
        os.rename(src, dst)
