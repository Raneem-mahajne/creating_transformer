"""Visualization from checkpoints."""
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
    plot_qk_full_attention_heatmap,
    plot_lm_head_probability_heatmaps,
    plot_v_before_after_demo_sequences,
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
    with open(os.path.join(plots_dir, "training_data_samples.txt"), "w", encoding="utf-8") as f:
        f.write(f"# Training Data Samples for: {config_name_actual}\n")
        f.write(f"# {len(decoded_train_sequences)} sample sequences (decoded from train_sequences)\n")
        f.write("# Format: space-separated tokens, one sequence per line\n\n")
        for seq in decoded_train_sequences:
            f.write(" ".join(str(i) for i in seq) + "\n")

    _train_acc, _train_correct, _train_incorrect = plot_training_data_heatmap(
        decoded_train_sequences,
        generator,
        save_path=os.path.join(plots_dir, "training_data_heatmap.png"),
        num_sequences=len(decoded_train_sequences),
        max_length=50,
    )

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
        save_path=os.path.join(plots_dir, "learning_curve.png"),
        eval_interval=eval_interval,
    )

    if generated_sequences_e0:
        (acc0, c0, i0), (accf, cf, inf) = plot_generated_sequences_heatmap_before_after(
            generated_sequences_e0,
            generated_sequences,
            generator,
            save_path=os.path.join(plots_dir, "generated_sequences_heatmap.png"),
            num_sequences=5,
            max_length=50,
        )
        print(f"Generated sequences heatmap (E0): {c0} correct, {i0} incorrect positions ({acc0:.1%} accuracy)")
        print(f"Generated sequences heatmap (Final): {cf} correct, {inf} incorrect positions ({accf:.1%} accuracy)")
    else:
        heatmap_accuracy, correct_count, incorrect_count = plot_generated_sequences_heatmap(
            generated_sequences,
            generator,
            save_path=os.path.join(plots_dir, "generated_sequences_heatmap.png"),
            num_sequences=len(generated_sequences),
            max_length=50,
        )
        print(f"Generated sequences heatmap: {correct_count} correct, {incorrect_count} incorrect positions ({heatmap_accuracy:.1%} accuracy)")

    X1, _ = get_batch_from_sequences(train_sequences, block_size, 1)
    X2, _ = get_batch_from_sequences(train_sequences, block_size, 1)
    X3, _ = get_batch_from_sequences(train_sequences, block_size, 1)
    X_list = [X1, X2, X3]

    arch_path = os.path.join(str(get_plots_dir(config_name_actual)), "architecture.png")
    if not os.path.exists(arch_path):
        plot_architecture_diagram(config, save_path=arch_path, model=model, vocab_size=vocab_size, batch_size=training_config.get('batch_size', 4))

    plot_weights_qkv_two_sequences(
        model, X_list, itos, save_path=os.path.join(plots_dir, "qkv.png"), num_sequences=3
    )
    plot_embeddings_pca(model, itos, save_path=os.path.join(plots_dir, "embeddings.png"))
    plot_embeddings_scatterplots_only(model, itos, save_path=os.path.join(plots_dir, "embeddings_scatterplots.png"))
    plot_embedding_qkv_comprehensive(model, itos, save_path=os.path.join(plots_dir, "embedding_qkv_comprehensive.png"))
    plot_qkv_transformations(model, itos, save_path=os.path.join(plots_dir, "qkv_transformations.png"))
    plot_token_position_embedding_space(
        model, itos, save_path=os.path.join(plots_dir, "token_position_embedding_space.png")
    )
    plot_attention_matrix(
        model, X_list, itos, save_path=os.path.join(plots_dir, "attention_matrix.png"), num_sequences=3
    )
    plot_qk_embedding_space(model, itos, save_path=os.path.join(plots_dir, "qk_embedding_space.png"))
    plot_qk_full_attention_heatmap(
        model, itos, save_path=os.path.join(plots_dir, "qk_full_attention_heatmap.png")
    )
    plot_lm_head_probability_heatmaps(
        model, itos, save_path=os.path.join(plots_dir, "lm_head_probability_heatmaps.png")
    )
    demo_sequences = [s for s in train_sequences[:8] if len(s) >= 2]
    if demo_sequences:
        plot_v_before_after_demo_sequences(
            model, itos, demo_sequences, save_dir=plots_dir,
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
