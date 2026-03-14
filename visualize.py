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
    set_journal_mode,
    clear_journal_mode,
    plot_training_data_heatmap,
    plot_learning_curve,
    plot_generated_sequences_heatmap_before_after,
    plot_generated_sequences_heatmap,
    plot_architecture_diagram,
    plot_weights_qkv_two_sequences,
    plot_q_dot_product_gradients,
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
    plot_qk_full_attention_heatmap_last_row,
    plot_qk_full_attention_combined,
    plot_lm_head_probability_heatmaps,
    plot_v_before_after_demo_sequences,
    plot_final_on_output_heatmap_grid,
    plot_residuals,
    plot_probability_heatmap,
    plot_probability_heatmap_with_embeddings,
    plot_probability_heatmap_with_values,
    plot_per_token_frozen_output,
)


def _has_odd_number(decoded_tokens: list) -> bool:
    """True if decoded token list contains at least one odd integer."""
    return any(isinstance(t, int) and t % 2 == 1 for t in decoded_tokens)


def _starts_with_even_odd_plus(decoded_tokens: list) -> bool:
    """True if decoded token list starts with [even number, odd number, '+']."""
    if len(decoded_tokens) < 3:
        return False
    first = decoded_tokens[0]
    second = decoded_tokens[1]
    third = decoded_tokens[2]
    return (isinstance(first, int) and first % 2 == 0 and
            isinstance(second, int) and second % 2 == 1 and
            third == "+")


def _has_two_plus_with_different_evens(decoded_tokens: list) -> bool:
    """True if sequence has 2 plus signs, each followed by a different even number
    and an odd number, and at least one plus is immediately preceded by an odd number."""
    if len(decoded_tokens) < 7:
        return False
    
    plus_positions = [i for i, token in enumerate(decoded_tokens) if token == "+"]
    if len(plus_positions) < 2:
        return False
    
    # Check first plus: should be followed by even, odd
    first_plus_idx = plus_positions[0]
    if first_plus_idx + 2 >= len(decoded_tokens):
        return False
    even1 = decoded_tokens[first_plus_idx + 1]
    odd1 = decoded_tokens[first_plus_idx + 2]
    if not (isinstance(even1, int) and even1 % 2 == 0 and isinstance(odd1, int) and odd1 % 2 == 1):
        return False
    
    # Check second plus: should be followed by different even, odd
    second_plus_idx = plus_positions[1]
    if second_plus_idx + 2 >= len(decoded_tokens):
        return False
    even2 = decoded_tokens[second_plus_idx + 1]
    odd2 = decoded_tokens[second_plus_idx + 2]
    if not (isinstance(even2, int) and even2 % 2 == 0 and isinstance(odd2, int) and odd2 % 2 == 1):
        return False
    
    # Check that the two even numbers are different
    if even1 == even2:
        return False

    # At least one plus must be immediately preceded by an odd number
    has_odd_before_plus = False
    for pidx in [first_plus_idx, second_plus_idx]:
        if pidx > 0:
            before = decoded_tokens[pidx - 1]
            if isinstance(before, int) and before % 2 == 1:
                has_odd_before_plus = True
                break
    return has_odd_before_plus


def _matches_pattern_2_3_1_plus_2_4_plus_4(decoded_tokens: list) -> bool:
    """Check if sequence matches pattern like '2 3 1 + 2 4 + 4' (numbers + numbers + single_number)."""
    plus_positions = [i for i, token in enumerate(decoded_tokens) if token == "+"]
    if len(plus_positions) < 2:
        return False
    
    first_plus_idx = plus_positions[0]
    second_plus_idx = plus_positions[1]
    
    # Check that second plus is followed by exactly one number (or at least starts with one number)
    if second_plus_idx + 1 >= len(decoded_tokens):
        return False
    
    # Check that after second plus, there's at least one number
    after_second_plus = decoded_tokens[second_plus_idx + 1]
    if not isinstance(after_second_plus, int):
        return False
    
    # Check that first plus has numbers before and after it
    if first_plus_idx == 0 or first_plus_idx + 1 >= len(decoded_tokens):
        return False
    
    # Pattern: [numbers] + [numbers] + [number]
    # We have at least one token before first plus, tokens after first plus, and at least one after second plus
    return True


def _has_two_plus_with_same_even(decoded_tokens: list) -> bool:
    """True if sequence has 2 plus signs, each followed by the same even number."""
    plus_positions = [i for i, token in enumerate(decoded_tokens) if token == "+"]
    if len(plus_positions) < 2:
        return False
    
    first_plus_idx = plus_positions[0]
    second_plus_idx = plus_positions[1]
    
    # Check that both plus signs are followed by at least one token
    if first_plus_idx + 1 >= len(decoded_tokens) or second_plus_idx + 1 >= len(decoded_tokens):
        return False
    
    # Get the token immediately after each plus sign
    after_first = decoded_tokens[first_plus_idx + 1]
    after_second = decoded_tokens[second_plus_idx + 1]
    
    # Both must be integers and even numbers, and they must be the same
    if not (isinstance(after_first, int) and isinstance(after_second, int)):
        return False
    
    if after_first % 2 != 0 or after_second % 2 != 0:
        return False
    
    return after_first == after_second


def _has_two_plus_with_different_even_numbers(decoded_tokens: list) -> bool:
    """True if sequence has exactly 2 plus signs, each followed by a different even number.
    Sequence should not start with a plus sign."""
    plus_positions = [i for i, token in enumerate(decoded_tokens) if token == "+"]
    # Must have exactly 2 plus signs
    if len(plus_positions) != 2:
        return False
    
    # Sequence should not start with a plus sign
    if len(decoded_tokens) == 0 or decoded_tokens[0] == "+":
        return False
    
    first_plus_idx = plus_positions[0]
    second_plus_idx = plus_positions[1]
    
    # Check that both plus signs are followed by at least one token
    if first_plus_idx + 1 >= len(decoded_tokens) or second_plus_idx + 1 >= len(decoded_tokens):
        return False
    
    # Get the token immediately after each plus sign
    after_first = decoded_tokens[first_plus_idx + 1]
    after_second = decoded_tokens[second_plus_idx + 1]
    
    # Both must be integers and even numbers, and they must be different
    if not (isinstance(after_first, int) and isinstance(after_second, int)):
        return False
    
    if after_first % 2 != 0 or after_second % 2 != 0:
        return False
    
    return after_first != after_second


def visualize_from_checkpoint(
    config_name_actual: str, checkpoint_data: dict, config: dict, step: int = None,
    plots_subfolder: str | None = None, sequence_seed: int | None = None, sequence_index: int | None = None,
    fixed_sequence_decoded: list | None = None, require_odd_in_sequence: bool = False,
    generate_journal: bool = False, _is_journal_pass: bool = False,
    only_figures: list[int] | None = None,
):
    """Generate all visualizations from checkpoint data.
    If only_figures is set (e.g. [8]), only generate plots whose figure number is in the list.
    If plots_subfolder is set, save plots to that subfolder (do not overwrite main plots).
    If sequence_seed or sequence_index are set, use a different sequence for sequence-dependent plots.
    If fixed_sequence_decoded is set (and reseed params are not), use this sequence for plots (decoded form, e.g. [10, '+', 10, 6, '+', 6, 4, 8]).
    If require_odd_in_sequence is True and selecting by seed/index, keep trying indices until the chosen sequence contains an odd number.
    """
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
    stoi = checkpoint_data.get("stoi")  # for encoding fixed_sequence_decoded to token ids

    block_size = model_config["block_size"]
    data_config = config["data"]
    training_config = config.get("training", {})

    plots_dir = get_plots_dir(config_name_actual, step, subfolder=plots_subfolder)
    plots_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = str(plots_dir)
    if plots_subfolder:
        print(f"Plots will be saved to subfolder: {plots_subfolder}")

    # Deterministic defaults when no seed override: same generated/consistent sequence every run
    _default_seed = 42
    if sequence_seed is None:
        random.seed(_default_seed)
        torch.manual_seed(_default_seed)

    generator = get_generator_from_config(config)

    decoded_train_sequences = [decode(seq) for seq in train_sequences[:6]]
    manifest = _load_plots_manifest()

    def _plot_path(default_name: str, base_dir: str = plots_dir) -> str:
        mapped = _lookup_manifest_filename(manifest, config_name_actual, default_name)
        return os.path.join(base_dir, mapped)

    def _plot(default_name: str) -> bool:
        return _should_plot_figure(manifest, config_name_actual, default_name, only_figures)

    with open(os.path.join(plots_dir, "training_data_samples.txt"), "w", encoding="utf-8") as f:
        f.write(f"# Training Data Samples for: {config_name_actual}\n")
        f.write(f"# {len(decoded_train_sequences)} sample sequences (decoded from train_sequences)\n")
        f.write("# Format: space-separated tokens, one sequence per line\n\n")
        for seq in decoded_train_sequences:
            f.write(" ".join(str(i) for i in seq) + "\n")

    num_sequences_to_generate = 5
    # Reseed immediately before generation so sequences are identical every run (no drift from prior code)
    if sequence_seed is not None:
        random.seed(sequence_seed)
        torch.manual_seed(sequence_seed)
    else:
        random.seed(_default_seed)
        torch.manual_seed(_default_seed)
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

    if _plot("training_data_heatmap.png"):
        _train_acc, _train_correct, _train_incorrect = plot_training_data_heatmap(
            decoded_train_sequences,
            generator,
            save_path=_plot_path("training_data_heatmap.png"),
            num_sequences=min(4, len(decoded_train_sequences)),
            max_length=20,
        )

    if _plot("learning_curve.png"):
        plot_learning_curve(
            steps_for_plot,
            train_loss_history,
            val_loss_history,
            rule_error_history=rule_error_history,
            save_path=_plot_path("learning_curve.png"),
            eval_interval=eval_interval,
        )

    if _plot("generated_sequences_heatmap.png"):
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
    if fixed_sequence_decoded is not None and sequence_seed is None and sequence_index is None and stoi is not None:
        # Use the explicitly requested sequence (e.g. "10 + 10 6 + 6 4 8") as token ids
        try:
            consistent_sequence = [stoi[str(t)] for t in fixed_sequence_decoded]
        except KeyError as e:
            print(f"Warning: fixed_sequence_decoded token not in vocab: {e}; falling back to seed/index selection")
            consistent_sequence = None
    else:
        consistent_sequence = None
    if consistent_sequence is None:
        # Pick deterministically: sort so order does not depend on train_sequences iteration
        seed_for_seq = 43 if sequence_seed is None else sequence_seed
        random.seed(seed_for_seq)
        valid_sequences = sorted(
            [seq for seq in train_sequences if len(seq) >= 7],
            key=lambda s: tuple(s),
        )  # Need at least 7 tokens for pattern with 2 plus signs
        if valid_sequences:
            # Try indices until we find a sequence with 2 plus signs, each followed by different even numbers
            start_idx = (sequence_index if sequence_index is not None else 0) % len(valid_sequences)
            for k in range(len(valid_sequences)):
                idx = (start_idx + k) % len(valid_sequences)
                cand = valid_sequences[idx]
                decoded_cand = decode(cand)
                if _has_two_plus_with_different_evens(decoded_cand):
                    consistent_sequence = cand
                    if k > 0:
                        print(f"Reseed: using sequence index {idx} (first with pattern: 2 plus signs, each followed by different even numbers)")
                    break
            else:
                # Fallback: try the old pattern if new one doesn't match
                valid_sequences_old = sorted(
                    [seq for seq in train_sequences if len(seq) >= 3],
                    key=lambda s: tuple(s),
                )
                if valid_sequences_old:
                    for k in range(len(valid_sequences_old)):
                        idx = (start_idx + k) % len(valid_sequences_old)
                        cand = valid_sequences_old[idx]
                        decoded_cand = decode(cand)
                        if _starts_with_even_odd_plus(decoded_cand):
                            consistent_sequence = cand
                            print(f"Warning: no sequence with 2 plus signs pattern found; using sequence with pattern [even, odd, '+']")
                            break
                    else:
                        consistent_sequence = valid_sequences_old[start_idx] if valid_sequences_old else valid_sequences[start_idx]
                        print("Warning: no matching pattern found; using default sequence")
                else:
                    consistent_sequence = valid_sequences[start_idx]
                    print("Warning: no matching pattern found; using default sequence")
        elif train_sequences:
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

    # Hardcoded demo sequence for figures 13–17, 18, and frozen_output supp (length 8): final token is '+' and NOT immediately after another '+'
    _demo_decoded = [4, 1, "+", 4, 6, 9, 5, "+"]
    _stoi = stoi if stoi is not None else {str(itos[i]): i for i in range(len(itos))}
    demo_sequence = None
    try:
        _demo_ids = [_stoi[str(t)] for t in _demo_decoded]
        demo_sequence = _demo_ids[:block_size]
        X_demo = torch.tensor(demo_sequence, dtype=torch.long).unsqueeze(0)
        X_list_demo = [X_demo, X_demo, X_demo]
    except KeyError:
        X_list_demo = X_list
        X_demo = X_consistent

    # When using a subfolder, write architecture there too so the folder is self-contained
    base_plots_dir = str(get_plots_dir(config_name_actual, step, subfolder=plots_subfolder))
    arch_path = _plot_path("architecture.png", base_dir=base_plots_dir)
    if _plot("architecture.png") and not os.path.exists(arch_path):
        plot_architecture_diagram(config, save_path=arch_path, model=model, vocab_size=vocab_size, batch_size=training_config.get('batch_size', 4))

    if _plot("qkv_query_key_attention.png"):
        plot_weights_qkv_two_sequences(
            model, X_list_demo, itos, save_path=_plot_path("qkv_query_key_attention.png"), num_sequences=1
        )
    if _plot("q_dot_product_gradients.png"):
        plot_q_dot_product_gradients(
            model, X_list_demo, itos, save_path=_plot_path("q_dot_product_gradients.png"), num_sequences=1
        )
    if _plot("residuals.png"):
        plot_residuals(
            model, X_list_demo, itos, save_path=_plot_path("residuals.png"), num_sequences=1
        )
    if _plot("embeddings.png"):
        plot_embeddings_pca(model, itos, save_path=_plot_path("embeddings.png"))
    # plot_embeddings_scatterplots_only removed — redundant with bottom row of embeddings.png (Figure 05)
    path_qkv_comp = _plot_path_if_in_manifest("embedding_qkv_comprehensive.png", manifest, config_name_actual, plots_dir)
    if path_qkv_comp and _plot("embedding_qkv_comprehensive.png"):
        plot_embedding_qkv_comprehensive(model, itos, save_path=path_qkv_comp)
    if _plot("qkv_transformations.png"):
        plot_qkv_transformations(model, itos, save_path=_plot_path("qkv_transformations.png"))
    path_tp_space = _plot_path_if_in_manifest("token_position_embedding_space.png", manifest, config_name_actual, plots_dir)
    if path_tp_space and _plot("token_position_embedding_space.png"):
        plot_token_position_embedding_space(model, itos, save_path=path_tp_space)
    # plot_attention_matrix moved to supplementary
    if _plot("qk_embedding_space.png"):
        plot_qk_embedding_space(model, itos, save_path=_plot_path("qk_embedding_space.png"))
    if _plot("qk_embedding_space_plus5_focus.png"):
        plot_qk_embedding_space_focused_query(
            model, itos, token_str="+", position=5,
            save_path=_plot_path("qk_embedding_space_plus5_focus.png"),
        )
    # Plot embeddings for demo sequence (fig 13: final is + not after +)
    if _plot("sequence_embeddings.png"):
        plot_sequence_embeddings(
            model, X_demo, itos, save_path=_plot_path("sequence_embeddings.png")
        )
    if _plot("qk_full_attention_heatmap.png"):
        if _is_journal_pass:
            plot_qk_full_attention_combined(
                model, itos, save_path=_plot_path("qk_full_attention_heatmap.png")
            )
        else:
            plot_qk_full_attention_heatmap(
                model, itos, save_path=_plot_path("qk_full_attention_heatmap.png")
            )
            plot_qk_full_attention_heatmap_last_row(
                model, itos, save_path=_plot_path("qk_full_attention_heatmap_last_row.png")
            )
    path_lm_head = _plot_path_if_in_manifest("lm_head_probability_heatmaps.png", manifest, config_name_actual, plots_dir)
    if path_lm_head and _plot("lm_head_probability_heatmaps.png"):
        plot_lm_head_probability_heatmaps(model, itos, save_path=path_lm_head)
    if _plot("probability_heatmap.png"):
        plot_probability_heatmap(
            model, itos, save_path=_plot_path("probability_heatmap.png")
        )
    if _plot("probability_heatmap_with_embeddings.png"):
        plot_probability_heatmap_with_embeddings(
            model, itos, save_path=_plot_path("probability_heatmap_with_embeddings.png")
        )
    if _plot("probability_heatmap_with_values.png"):
        plot_probability_heatmap_with_values(
            model, itos, save_path=_plot_path("probability_heatmap_with_values.png")
        )
    # Final-on-output (18) and frozen_output supp: same demo sequence as figs 13–17
    seq_for_18_and_supp = demo_sequence if demo_sequence is not None else consistent_sequence
    if seq_for_18_and_supp and _plot("final_on_output_heatmap_grid.png"):
        plot_final_on_output_heatmap_grid(
            model, itos, seq_for_18_and_supp, save_path=_plot_path("final_on_output_heatmap_grid.png")
        )
        frozen_dir = os.path.join(plots_dir, "frozen_output")
        os.makedirs(frozen_dir, exist_ok=True)
        plot_per_token_frozen_output(
            model, itos, seq_for_18_and_supp, save_dir=frozen_dir
        )

    print(f"All visualizations saved to {plots_dir}")

    if generate_journal and not _is_journal_pass:
        journal_subfolder = "a4" if plots_subfolder is None else f"{plots_subfolder}/a4"
        print(f"\n--- Generating A4 journal versions in {journal_subfolder}/ ---")
        set_journal_mode(max_width=7.0, max_height=9.5, dpi=300)
        try:
            visualize_from_checkpoint(
                config_name_actual, checkpoint_data, config, step=step,
                plots_subfolder=journal_subfolder,
                sequence_seed=sequence_seed, sequence_index=sequence_index,
                fixed_sequence_decoded=fixed_sequence_decoded,
                require_odd_in_sequence=require_odd_in_sequence,
                generate_journal=False, _is_journal_pass=True,
                only_figures=only_figures,
            )
        finally:
            clear_journal_mode()


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


def _lookup_manifest_number(manifest: list[dict], config_name: str, default_name: str) -> int | None:
    """Return figure number (e.g. 8 for '08') for this config and default_name, or None."""
    for row in manifest:
        if row["config"] == config_name and row["default_name"] == default_name:
            num_str = row.get("number", "")
            try:
                return int(str(num_str).split("_")[0])
            except (ValueError, AttributeError):
                return None
    return None


def _should_plot_figure(
    manifest: list[dict], config_name: str, default_name: str, only_figures: list[int] | None
) -> bool:
    """If only_figures is None, return True. Else return True only if this plot's figure number is in only_figures."""
    if only_figures is None:
        return True
    num = _lookup_manifest_number(manifest, config_name, default_name)
    return num is not None and num in only_figures


def _plot_path_if_in_manifest(
    default_name: str, manifest: list[dict], config_name: str, base_dir: str
) -> str | None:
    """Return save path only if this config has a manifest entry for default_name (so we get a numbered filename)."""
    mapped = _lookup_manifest_filename(manifest, config_name, default_name)
    if mapped == default_name:
        return None
    return os.path.join(base_dir, mapped)


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
