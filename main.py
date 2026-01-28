import torch
import numpy as np
import os
import random
import sys
from IntegerStringGenerator import OperatorBasedGenerator
from config_loader import load_config, get_generator_from_config

# Import from refactored modules
from model import BigramLanguageModel
from data import (
    generate_integer_string_data,
    build_encoder_for_integers,
    build_encoder_with_operators,
    split_train_val_sequences,
    get_batch_from_sequences,
)
from training import estimate_loss, estimate_rule_error
from checkpoint import (
    get_checkpoint_dir,
    get_plots_dir,
    save_checkpoint,
    load_checkpoint,
    list_available_checkpoints,
    create_decode_from_itos,
)
from visualize import visualize_from_checkpoint, visualize_all_checkpoints
from video import create_embeddings_scatterplots_video
from plotting import (
    plot_training_data_heatmap,
    plot_learning_curve,
    plot_generated_sequences_heatmap_before_after,
    plot_generated_sequences_heatmap,
    plot_architecture_diagram,
    plot_weights_qkv_two_sequences,
    plot_embeddings_pca,
    plot_qkv_transformations,
    plot_token_position_embedding_space,
    plot_attention_matrix,
    plot_qk_embedding_space,
    plot_qk_full_attention_heatmap,
)


def main(config_name: str = "copy_modulo", force_retrain: bool = False, visualize_only: bool = False, step: int = None, visualize_all: bool = False):
    """
    Main training function.
    
    Args:
        config_name: Name of the config file (without .yaml extension) in the configs folder
        force_retrain: If True, retrain even if checkpoint exists
        visualize_only: If True, only generate visualizations (no training)
        step: If visualize_only=True, visualize this specific step. If None, visualize final checkpoint.
        visualize_all: If True, visualize all available checkpoints
    """
    print(f"Starting with config: {config_name}")
    torch.manual_seed(0)

    # Load configuration
    config = load_config(config_name)
    print(f"Loaded configuration: {config['name']}")
    
    # Extract config values
    config_name_actual = config['name']
    data_config = config['data']
    model_config = config['model']
    training_config = config['training']
    
    # Create plots directory with config name subfolder
    plots_dir = get_plots_dir(config_name_actual)
    plots_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = str(plots_dir)  # Convert to string for os.path.join compatibility
    
    # Handle video creation mode (early return)
    if "--video" in sys.argv:
        fps = 2
        max_steps = None
        if "--fps" in sys.argv:
            fps_idx = sys.argv.index("--fps")
            if fps_idx + 1 < len(sys.argv):
                try:
                    fps = int(sys.argv[fps_idx + 1])
                except ValueError:
                    print("Error: --fps must be followed by an integer")
                    sys.exit(1)
        if "--max-steps" in sys.argv:
            max_steps_idx = sys.argv.index("--max-steps")
            if max_steps_idx + 1 < len(sys.argv):
                try:
                    max_steps = int(sys.argv[max_steps_idx + 1])
                except ValueError:
                    print("Error: --max-steps must be followed by an integer")
                    sys.exit(1)
        create_embeddings_scatterplots_video(config_name_actual, config, fps=fps, max_steps=max_steps)
        return
    
    # Handle visualize_only mode
    if visualize_only:
        if visualize_all:
            visualize_all_checkpoints(config_name_actual, config)
        else:
            checkpoint_data = load_checkpoint(config_name_actual, step=step)
            if checkpoint_data is None:
                print(f"Error: No checkpoint found for {config_name_actual}" + (f" at step {step}" if step else ""))
                return
            visualize_from_checkpoint(config_name_actual, checkpoint_data, config, step=step)
        return
    
    # Check for existing checkpoint
    checkpoint_data = None
    if not force_retrain:
        checkpoint_data = load_checkpoint(config_name_actual, step=None)
        if checkpoint_data:
            print("Using existing checkpoint. Set force_retrain=True to retrain.")
    
    if checkpoint_data is None:
        # Need to train
        print("No checkpoint found or force_retrain=True. Training new model...")
        
        # Delete old checkpoints to avoid mixing with new training run
        import shutil
        checkpoint_base = get_checkpoint_dir(config_name_actual)
        if checkpoint_base.exists():
            print(f"Deleting old checkpoints from {checkpoint_base}...")
            shutil.rmtree(checkpoint_base)
            print("Old checkpoints deleted.")

        # 1) Generate integer string data using generator from config
        generator = get_generator_from_config(config)
        sequences = generate_integer_string_data(
            generator, 
            num_sequences=data_config['num_sequences'],
            min_length=data_config['min_length'],
            max_length=data_config['max_length']
        )
        print(f"Generated {len(sequences)} sequences")
        print(f"Sequence lengths: min={min(len(s) for s in sequences)}, max={max(len(s) for s in sequences)}, avg={sum(len(s) for s in sequences)/len(sequences):.1f}")
        
        # 2) Build encoder/decoder for integers (or integers + operators)
        min_value = data_config['min_value']
        max_value = data_config['max_value']
        
        # Check if generator uses operators
        if isinstance(generator, OperatorBasedGenerator):
            operators = generator.operators
            encode, decode, vocab_size, itos, stoi = build_encoder_with_operators(
                min_value=min_value, max_value=max_value, operators=operators
            )
            print("Vocabulary size:", vocab_size)
            print("Vocabulary (integers + operators):", [itos[i] for i in range(vocab_size)])
        else:
            encode, decode, vocab_size, itos, stoi = build_encoder_for_integers(min_value=min_value, max_value=max_value)
            print("Vocabulary size:", vocab_size)
            print("Vocabulary (integers):", [itos[i] for i in range(vocab_size)])

        # 3) Encode sequences (integer values -> token indices)
        encoded_sequences = [encode(seq) for seq in sequences]
        
        # 4) Split sequences into train/val
        train_sequences, val_sequences = split_train_val_sequences(encoded_sequences, train_ratio=0.9)
        print(f"Train: {len(train_sequences)} sequences, Val: {len(val_sequences)} sequences")
        
        # 4.5) Save some training data to a text file
        num_samples_to_save = min(6, len(sequences))  # Save up to 6 sample sequences
        with open(os.path.join(plots_dir, "training_data_samples.txt"), "w", encoding="utf-8") as f:
            f.write(f"# Training Data Samples for: {config_name_actual}\n")
            f.write(f"# {num_samples_to_save} sample sequences (original integer values before encoding)\n")
            f.write(f"# Format: space-separated integers, one sequence per line\n\n")
            for i, seq in enumerate(sequences[:num_samples_to_save]):
                f.write(" ".join(str(val) for val in seq) + "\n")
        print(f"Saved {num_samples_to_save} training data samples to {os.path.join(plots_dir, 'training_data_samples.txt')}")
        
        # 4.6) Visualize training data with rule correctness
        train_heatmap_accuracy, train_correct_count, train_incorrect_count = plot_training_data_heatmap(
            sequences, generator,  # Use original sequences (before encoding)
            save_path=os.path.join(plots_dir, "training_data_heatmap.png"),
            num_sequences=min(6, len(sequences)),  # Show up to 6 training sequences
            max_length=50
        )
        print(f"Training data heatmap: {train_correct_count} correct, {train_incorrect_count} incorrect positions ({train_heatmap_accuracy:.1%} accuracy)")

        # 5) Create model + optimizer
        n_embd = model_config['n_embd']
        block_size = model_config['block_size']
        num_heads = model_config['num_heads']
        head_size = model_config['head_size']
        
        model = BigramLanguageModel(vocab_size, n_embd, block_size, num_heads, head_size)
        optimizer = torch.optim.AdamW(model.parameters(), lr=training_config['learning_rate'])

        # Generate "before training" sequences (E0) without perturbing RNG state for training
        py_state = random.getstate()
        np_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        generated_sequences_e0 = []
        for _ in range(5):
            seq_length = random.randint(data_config['min_length'], data_config['max_length'])
            start_token = random.randint(0, vocab_size - 1)
            start = torch.tensor([[start_token]], dtype=torch.long)
            sample = model.generate(start, max_new_tokens=seq_length - 1)[0].tolist()
            generated_sequences_e0.append(decode(sample))
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)

        # 6) Training loop
        steps_for_plot = []
        train_loss_history = []
        val_loss_history = []
        rule_error_history = []

        batch_size = training_config['batch_size']
        max_steps = training_config['max_steps']
        eval_interval = training_config['eval_interval']
        eval_iterations = training_config['eval_iterations']
        checkpoint_interval = training_config.get('checkpoint_interval', 1000)  # Save checkpoint every N steps
        
        X_fixed, _ = get_batch_from_sequences(train_sequences, block_size, batch_size)

        for step in range(max_steps):
            # Evaluate occasionally
            if step % eval_interval == 0:
                losses = estimate_loss(model, train_sequences, val_sequences, block_size, batch_size, eval_iterations)
                rule_err = estimate_rule_error(model, generator, decode, block_size, num_samples=20, seq_length=30)

                steps_for_plot.append(step)
                train_loss_history.append(losses["train"])
                val_loss_history.append(losses["validation"])
                rule_error_history.append(rule_err)

                print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['validation']:.4f}, rule err {rule_err:.4f}", flush=True)

            # Save checkpoint at intervals
            if checkpoint_interval > 0 and step > 0 and step % checkpoint_interval == 0:
                save_checkpoint(
                    config_name_actual, model, train_sequences, val_sequences,
                    itos, stoi, vocab_size, steps_for_plot, train_loss_history,
                    val_loss_history, rule_error_history, model_config, eval_interval, step=step,
                    generated_sequences_e0=generated_sequences_e0
                )

            # One batch
            X, Y = get_batch_from_sequences(train_sequences, block_size, batch_size)

            # Forward + backward + update
            _, loss = model(X, Y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        # Beep to signal training is complete
        print('\a', end='', flush=True)

        # Show results
        print("Final loss:", loss.item(), flush=True)
        print(f"Final rule error: {rule_error_history[-1]:.4f}" if rule_error_history else "", flush=True)
        
        # Save final checkpoint (step=None means final)
        save_checkpoint(
            config_name_actual, model, train_sequences, val_sequences,
            itos, stoi, vocab_size, steps_for_plot, train_loss_history,
            val_loss_history, rule_error_history, model_config, eval_interval, step=None,
            generated_sequences_e0=generated_sequences_e0
        )
        
        # Store data for visualization
        checkpoint_data = {
            "model": model,
            "train_sequences": train_sequences,
            "val_sequences": val_sequences,
            "itos": itos,
            "stoi": stoi,
            "vocab_size": vocab_size,
            "steps_for_plot": steps_for_plot,
            "train_loss_history": train_loss_history,
            "val_loss_history": val_loss_history,
            "rule_error_history": rule_error_history,
            "generated_sequences_e0": generated_sequences_e0,
            "model_config": model_config,
            "eval_interval": eval_interval,
        }
    
    # Extract data from checkpoint (whether loaded or newly trained)
    model = checkpoint_data["model"]
    train_sequences = checkpoint_data["train_sequences"]
    val_sequences = checkpoint_data["val_sequences"]
    itos = checkpoint_data["itos"]
    stoi = checkpoint_data["stoi"]
    vocab_size = checkpoint_data["vocab_size"]
    steps_for_plot = checkpoint_data["steps_for_plot"]
    train_loss_history = checkpoint_data["train_loss_history"]
    val_loss_history = checkpoint_data["val_loss_history"]
    rule_error_history = checkpoint_data.get("rule_error_history", [])
    model_config = checkpoint_data["model_config"]
    
    # Get decode function (either from checkpoint or create it)
    if "decode" in checkpoint_data:
        decode = checkpoint_data["decode"]
    else:
        # Recreate from itos (for backward compatibility)
        decode = create_decode_from_itos(itos)
    
    # Get block_size and eval_interval for visualization
    block_size = model_config['block_size']
    eval_interval = checkpoint_data.get("eval_interval") or training_config.get('eval_interval', 200)
    
    batch_size = training_config.get('batch_size', 4)  # For getting batches for visualization
    
    # Generate visualizations for final checkpoint
    visualize_from_checkpoint(config_name_actual, checkpoint_data, config, step=None)


if __name__ == "__main__":
    # Parse command line arguments
    # Usage:
    #   python main.py <config_name>                    # Train and visualize
    #   python main.py <config_name> --visualize       # Only visualize (final checkpoint)
    #   python main.py <config_name> --visualize --step <step_num>  # Visualize specific step
    #   python main.py <config_name> --force-retrain    # Force retrain
    
    config_name = "plus_last_even"  # default
    force_retrain = False
    visualize_only = False
    step = None
    
    if len(sys.argv) > 1:
        config_name = sys.argv[1]
    
    # Parse flags
    visualize_all = "--visualize-all" in sys.argv
    if "--visualize" in sys.argv or visualize_all:
        visualize_only = True
        if "--step" in sys.argv:
            step_idx = sys.argv.index("--step")
            if step_idx + 1 < len(sys.argv):
                try:
                    step = int(sys.argv[step_idx + 1])
                except ValueError:
                    print("Error: --step must be followed by an integer")
                    sys.exit(1)
    
    if "--force-retrain" in sys.argv:
        force_retrain = True
    
    main(config_name=config_name, force_retrain=force_retrain, visualize_only=visualize_only, step=step, visualize_all=visualize_all)
