"""Video creation from checkpoints."""
import numpy as np
import tempfile
from pathlib import Path

try:
    import imageio.v2 as imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    try:
        import imageio
        IMAGEIO_AVAILABLE = True
    except ImportError:
        IMAGEIO_AVAILABLE = False
        print("Warning: imageio not available. Install with: pip install imageio")

from checkpoint import (
    load_checkpoint,
    get_plots_dir,
    list_available_checkpoints,
)
from plotting import plot_embeddings_scatterplots_only, plot_embedding_qkv_comprehensive


def create_embeddings_scatterplots_video(config_name_actual: str, config: dict, fps: int = 2, max_steps: int = None):
    """
    Create a video showing embedding scatterplots evolving across training steps.
    
    Args:
        config_name_actual: Configuration name
        config: Configuration dict
        fps: Frames per second for the video
        max_steps: Maximum number of steps to include (None = all)
    """
    if not IMAGEIO_AVAILABLE:
        print("Error: imageio is required to create videos. Install with: pip install imageio")
        return
    
    steps = list_available_checkpoints(config_name_actual)
    if not steps:
        print(f"No step checkpoints found for {config_name_actual}")
        return
    
    # Ensure steps are sorted numerically
    steps = sorted(steps)
    
    # Filter to only include checkpoints within the config's max_steps
    training_max_steps = config.get('training', {}).get('max_steps', float('inf'))
    steps = [s for s in steps if s <= training_max_steps]
    print(f"Filtered to {len(steps)} checkpoints within training range (max_steps={training_max_steps})")
    
    if not steps:
        print(f"No checkpoints found within training range (0-{training_max_steps})")
        return
    
    # Limit number of frames if requested
    if max_steps:
        steps = steps[:max_steps]
    
    print(f"Creating video from {len(steps)} checkpoints: {steps[:5]}...{steps[-5:] if len(steps) > 10 else ''}")
    
    # First pass: calculate global axis limits across all checkpoints
    print("Calculating global axis limits...")
    all_token_xlims = []
    all_token_ylims = []
    all_pos_xlims = []
    all_pos_ylims = []
    all_comb_xlims = []
    all_comb_ylims = []
    
    for step in steps:
        checkpoint_data = load_checkpoint(config_name_actual, step=step)
        if not checkpoint_data:
            continue
        
        model = checkpoint_data["model"]
        itos = checkpoint_data["itos"]
        
        # Get embeddings
        embeddings = model.token_embedding.weight.detach().cpu().numpy()
        vocab_size, n_embd = embeddings.shape
        block_size = model.block_size
        pos_emb_all = model.position_embedding_table.weight.detach().cpu().numpy()
        
        # Token embeddings limits
        X_emb = embeddings.astype(np.float64)
        X_emb = X_emb - X_emb.mean(axis=0, keepdims=True)
        if n_embd > 2:
            _, _, Vt = np.linalg.svd(X_emb, full_matrices=False)
            X2 = X_emb @ Vt[:2].T
            margin = 0.15 * max(X2[:, 0].max() - X2[:, 0].min(), X2[:, 1].max() - X2[:, 1].min())
            all_token_xlims.append((X2[:, 0].min() - margin, X2[:, 0].max() + margin))
            all_token_ylims.append((X2[:, 1].min() - margin, X2[:, 1].max() + margin))
        elif n_embd == 2:
            margin = 0.15 * max(X_emb[:, 0].max() - X_emb[:, 0].min(), X_emb[:, 1].max() - X_emb[:, 1].min())
            all_token_xlims.append((X_emb[:, 0].min() - margin, X_emb[:, 0].max() + margin))
            all_token_ylims.append((X_emb[:, 1].min() - margin, X_emb[:, 1].max() + margin))
        
        # Position embeddings limits
        X_pos = pos_emb_all.astype(np.float64)
        X_pos = X_pos - X_pos.mean(axis=0, keepdims=True)
        if n_embd > 2:
            _, _, Vt_pos = np.linalg.svd(X_pos, full_matrices=False)
            X2_pos = X_pos @ Vt_pos[:2].T
            margin = 0.15 * max(X2_pos[:, 0].max() - X2_pos[:, 0].min(), X2_pos[:, 1].max() - X2_pos[:, 1].min())
            all_pos_xlims.append((X2_pos[:, 0].min() - margin, X2_pos[:, 0].max() + margin))
            all_pos_ylims.append((X2_pos[:, 1].min() - margin, X2_pos[:, 1].max() + margin))
        elif n_embd == 2:
            margin = 0.15 * max(X_pos[:, 0].max() - X_pos[:, 0].min(), X_pos[:, 1].max() - X_pos[:, 1].min())
            all_pos_xlims.append((X_pos[:, 0].min() - margin, X_pos[:, 0].max() + margin))
            all_pos_ylims.append((X_pos[:, 1].min() - margin, X_pos[:, 1].max() + margin))
        
        # Combined embeddings limits
        num_combinations = vocab_size * block_size
        all_combinations = np.zeros((num_combinations, n_embd))
        for token_idx in range(vocab_size):
            for pos_idx in range(block_size):
                idx = token_idx * block_size + pos_idx
                all_combinations[idx] = embeddings[token_idx] + pos_emb_all[pos_idx]
        
        if n_embd > 2:
            X_comb = all_combinations.astype(np.float64)
            X_comb = X_comb - X_comb.mean(axis=0, keepdims=True)
            _, _, Vt_comb = np.linalg.svd(X_comb, full_matrices=False)
            X2_comb = X_comb @ Vt_comb[:2].T
            margin = 0.15 * max(X2_comb[:, 0].max() - X2_comb[:, 0].min(), X2_comb[:, 1].max() - X2_comb[:, 1].min())
            all_comb_xlims.append((X2_comb[:, 0].min() - margin, X2_comb[:, 0].max() + margin))
            all_comb_ylims.append((X2_comb[:, 1].min() - margin, X2_comb[:, 1].max() + margin))
        elif n_embd == 2:
            margin = 0.15 * max(all_combinations[:, 0].max() - all_combinations[:, 0].min(), 
                                all_combinations[:, 1].max() - all_combinations[:, 1].min())
            all_comb_xlims.append((all_combinations[:, 0].min() - margin, all_combinations[:, 0].max() + margin))
            all_comb_ylims.append((all_combinations[:, 1].min() - margin, all_combinations[:, 1].max() + margin))
    
    # Calculate global limits
    fixed_limits = {}
    if all_token_xlims:
        fixed_limits['token'] = (
            (min(x[0] for x in all_token_xlims), max(x[1] for x in all_token_xlims)),
            (min(y[0] for y in all_token_ylims), max(y[1] for y in all_token_ylims))
        )
    if all_pos_xlims:
        fixed_limits['position'] = (
            (min(x[0] for x in all_pos_xlims), max(x[1] for x in all_pos_xlims)),
            (min(y[0] for y in all_pos_ylims), max(y[1] for y in all_pos_ylims))
        )
    if all_comb_xlims:
        fixed_limits['combined'] = (
            (min(x[0] for x in all_comb_xlims), max(x[1] for x in all_comb_xlims)),
            (min(y[0] for y in all_comb_ylims), max(y[1] for y in all_comb_ylims))
        )
    
    print(f"Global limits calculated: {fixed_limits}")
    
    # Second pass: generate plots with fixed limits
    print("Generating plots for video frames...")
    temp_dir = Path(tempfile.mkdtemp())
    # Store frames with step number for proper ordering
    frame_data = []
    
    for step in steps:
        checkpoint_data = load_checkpoint(config_name_actual, step=step)
        if not checkpoint_data:
            print(f"Warning: Could not load checkpoint for step {step}, skipping...")
            continue
        
        model = checkpoint_data["model"]
        itos = checkpoint_data["itos"]
        
        frame_path = temp_dir / f"frame_{step:06d}.png"
        try:
            plot_embeddings_scatterplots_only(
                model, itos, 
                save_path=str(frame_path),
                fixed_limits=fixed_limits,
                step_label=step
            )
            if frame_path.exists():
                frame_data.append((step, frame_path))
            else:
                print(f"Warning: Frame file was not created for step {step}")
        except Exception as e:
            print(f"Error generating frame for step {step}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        if len(frame_data) % 10 == 0:
            print(f"  Generated {len(frame_data)}/{len(steps)} frames...")
    
    # Sort frames by step number to ensure correct order
    frame_data.sort(key=lambda x: x[0])
    
    # Create video and GIF
    print(f"Creating video and GIF from {len(frame_data)} frames...")
    plots_dir = get_plots_dir(config_name_actual)
    plots_dir.mkdir(parents=True, exist_ok=True)
    video_path = plots_dir / "embeddings_scatterplots_evolution.mp4"
    gif_path = plots_dir / "embeddings_scatterplots_evolution.gif"
    
    try:
        # Read all frames in correct order
        frames = []
        missing_frames = []
        for step, frame_path in frame_data:
            if frame_path.exists():
                try:
                    frames.append(imageio.imread(frame_path))
                except Exception as e:
                    print(f"Warning: Failed to read frame for step {step}: {e}")
                    missing_frames.append(step)
            else:
                print(f"Warning: Frame file missing for step {step}: {frame_path}")
                missing_frames.append(step)
        
        if missing_frames:
            print(f"Warning: {len(missing_frames)} frames were missing or failed to load")
        
        print(f"Successfully loaded {len(frames)} frames out of {len(frame_data)} expected")
        if len(frames) != len(frame_data):
            print(f"Warning: Expected {len(frame_data)} frames but only loaded {len(frames)}")
        
        if frames:
            # Write MP4 video
            try:
                imageio.mimwrite(str(video_path), frames, fps=fps, codec='libx264', quality=8)
                print(f"Video saved to: {video_path} ({len(frames)} frames, {len(frames)/fps:.1f}s at {fps} fps)")
            except Exception as e:
                print(f"Error writing MP4: {e}")
                import traceback
                traceback.print_exc()
            
            # Write GIF (use duration instead of fps)
            try:
                duration = 1.0 / fps  # Duration per frame in seconds
                imageio.mimwrite(str(gif_path), frames, duration=duration, loop=0)
                print(f"GIF saved to: {gif_path} ({len(frames)} frames, {len(frames)/fps:.1f}s at {fps} fps)")
            except Exception as e:
                print(f"Error writing GIF: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("Error: No frames were generated")
    except Exception as e:
        print(f"Error creating video/GIF: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up temporary files
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("Temporary files cleaned up")


def create_embedding_qkv_video(config_name_actual: str, config: dict, fps: int = 2, max_steps: int = None):
    """
    Create a video showing the comprehensive embedding/QKV figure evolving across training steps.
    Single pass - generates frames directly without fixed limits.
    """
    import sys
    
    if not IMAGEIO_AVAILABLE:
        print("Error: imageio is required to create videos. Install with: pip install imageio", flush=True)
        return
    
    steps = list_available_checkpoints(config_name_actual)
    if not steps:
        print(f"No step checkpoints found for {config_name_actual}", flush=True)
        return
    
    # Ensure steps are sorted numerically
    steps = sorted(steps)
    
    # Filter to only include checkpoints within the config's max_steps
    training_max_steps = config.get('training', {}).get('max_steps', float('inf'))
    steps = [s for s in steps if s <= training_max_steps]
    print(f"Filtered to {len(steps)} checkpoints within training range (max_steps={training_max_steps})", flush=True)
    
    if not steps:
        print(f"No checkpoints found within training range (0-{training_max_steps})", flush=True)
        return
    
    # Limit number of frames if requested
    if max_steps:
        steps = steps[:max_steps]
    
    print(f"Creating comprehensive video from {len(steps)} checkpoints (single pass)", flush=True)
    print(f"Steps: {steps[0]} to {steps[-1]}", flush=True)
    
    # Single pass: generate all frames
    temp_dir = Path(tempfile.mkdtemp())
    print(f"Temp dir: {temp_dir}", flush=True)
    frame_data = []
    
    for i, step in enumerate(steps):
        try:
            checkpoint_data = load_checkpoint(config_name_actual, step=step)
            if not checkpoint_data:
                print(f"Warning: Could not load checkpoint for step {step}, skipping...", flush=True)
                continue
            
            model = checkpoint_data["model"]
            itos = checkpoint_data["itos"]
            
            frame_path = temp_dir / f"frame_{step:06d}.png"
            plot_embedding_qkv_comprehensive(
                model, itos, 
                save_path=str(frame_path),
                fixed_limits=None,  # Dynamic limits per frame
                step_label=step
            )
            if frame_path.exists():
                frame_data.append((step, frame_path))
            else:
                print(f"Warning: Frame not created for step {step}", flush=True)
                
        except Exception as e:
            print(f"Error at step {step}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            continue
        
        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{len(steps)} frames...", flush=True)
    
    print(f"Frame generation complete: {len(frame_data)} frames", flush=True)
    
    # Sort frames by step number
    frame_data.sort(key=lambda x: x[0])
    
    # Create video and GIF
    print(f"Creating video and GIF from {len(frame_data)} frames...", flush=True)
    plots_dir = get_plots_dir(config_name_actual)
    plots_dir.mkdir(parents=True, exist_ok=True)
    video_path = plots_dir / "embedding_qkv_evolution.mp4"
    gif_path = plots_dir / "embedding_qkv_evolution.gif"
    
    try:
        frames = []
        for step, frame_path in frame_data:
            if frame_path.exists():
                try:
                    frames.append(imageio.imread(frame_path))
                except Exception as e:
                    print(f"Warning: Failed to read frame for step {step}: {e}", flush=True)
        
        print(f"Successfully loaded {len(frames)} frames", flush=True)
        
        if frames:
            try:
                print(f"Writing MP4...", flush=True)
                imageio.mimwrite(str(video_path), frames, fps=fps, codec='libx264', quality=8)
                print(f"Video saved to: {video_path} ({len(frames)} frames, {len(frames)/fps:.1f}s at {fps} fps)", flush=True)
            except Exception as e:
                print(f"Error writing MP4: {e}", flush=True)
            
            try:
                print(f"Writing GIF...", flush=True)
                duration = 1.0 / fps
                imageio.mimwrite(str(gif_path), frames, duration=duration, loop=0)
                print(f"GIF saved to: {gif_path} ({len(frames)} frames, {len(frames)/fps:.1f}s at {fps} fps)", flush=True)
            except Exception as e:
                print(f"Error writing GIF: {e}", flush=True)
        else:
            print("Error: No frames were generated", flush=True)
    except Exception as e:
        print(f"Error creating video/GIF: {e}", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("Temporary files cleaned up", flush=True)
