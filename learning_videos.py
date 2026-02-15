"""Additional learning-dynamics videos (Q/K and output head views)."""
import tempfile
from pathlib import Path

import numpy as np

try:
    import imageio.v2 as imageio
    IMAGEIO_AVAILABLE = True
except ImportError:  # pragma: no cover - handled at runtime
    try:
        import imageio  # type: ignore
        IMAGEIO_AVAILABLE = True
    except ImportError:
        IMAGEIO_AVAILABLE = False
        print("Warning: imageio not available. Install with: pip install imageio")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:  # pragma: no cover - handled at runtime
    PIL_AVAILABLE = False
    print("Warning: Pillow (PIL) not available. Install with: pip install pillow for robust video sizing.")

from checkpoint import load_checkpoint, get_plots_dir, list_available_checkpoints
from plotting import (
    plot_qk_embedding_space,
    plot_qk_space_and_attention_heatmap,
    plot_probability_heatmap_with_embeddings,
)


def _get_learning_dynamics_dir(config_name_actual: str) -> Path:
    """Mirror video._get_learning_dynamics_dir without importing private names."""
    base_plots = get_plots_dir(config_name_actual)
    ld_dir = base_plots / "learning_dynamics"
    ld_dir.mkdir(parents=True, exist_ok=True)
    return ld_dir


def _normalize_frame_sizes(frames: list[np.ndarray]) -> list[np.ndarray]:
    """
    Ensure all frames have identical HxW so imageio/ffmpeg are happy.
    Uses Pillow to resize mismatched frames to the size of the first frame.
    """
    if not frames:
        return frames
    base_h, base_w = frames[0].shape[:2]
    if all(f.shape[:2] == (base_h, base_w) for f in frames):
        return frames

    if not PIL_AVAILABLE:
        print(
            "Warning: frame sizes differ but Pillow is not installed; "
            "video writing may fail with 'All images in a movie should have same size'.",
            flush=True,
        )
        return frames

    normalized: list[np.ndarray] = []
    for f in frames:
        h, w = f.shape[:2]
        if (h, w) == (base_h, base_w):
            normalized.append(f)
        else:
            img = Image.fromarray(f)
            img = img.resize((base_w, base_h), resample=Image.BILINEAR)
            normalized.append(np.asarray(img))
    return normalized


def _iter_steps(config_name_actual: str, config: dict, max_steps: int | None):
    """Yield sorted checkpoint steps within training range and optional max_steps cap."""
    steps = list_available_checkpoints(config_name_actual)
    if not steps:
        print(f"No step checkpoints found for {config_name_actual}", flush=True)
        return []
    steps = sorted(steps)

    training_max_steps = config.get("training", {}).get("max_steps", float("inf"))
    steps = [s for s in steps if s <= training_max_steps]
    if not steps:
        print(f"No checkpoints found within training range (0-{training_max_steps})", flush=True)
        return []
    if max_steps:
        steps = steps[:max_steps]
    return steps


def create_qk_space_video(config_name_actual: str, config: dict, fps: int = 20, max_steps: int | None = None):
    """Video of the Q/K embedding space (all token–position queries and keys)."""
    if not IMAGEIO_AVAILABLE:
        print("Error: imageio is required to create videos. Install with: pip install imageio", flush=True)
        return

    steps = _iter_steps(config_name_actual, config, max_steps)
    if not steps:
        return

    print(f"Creating Q/K space video from {len(steps)} checkpoints", flush=True)

    temp_dir = Path(tempfile.mkdtemp())
    frame_data: list[tuple[int, Path]] = []

    try:
        for i, step in enumerate(steps):
            try:
                checkpoint_data = load_checkpoint(config_name_actual, step=step)
            except (FileNotFoundError, OSError) as e:
                print(f"Warning: Could not load checkpoint for step {step}, skipping... ({e})", flush=True)
                continue
            if not checkpoint_data:
                print(f"Warning: Could not load checkpoint for step {step}, skipping...", flush=True)
                continue

            model = checkpoint_data["model"]
            itos = checkpoint_data["itos"]
            frame_path = temp_dir / f"frame_qk_{step:06d}.png"

            try:
                plot_qk_embedding_space(model, itos, save_path=str(frame_path), step_label=step)
                if frame_path.exists():
                    frame_data.append((step, frame_path))
            except Exception as e:  # pragma: no cover - visualization runtime
                print(f"Error generating Q/K space frame for step {step}: {e}", flush=True)
                import traceback

                traceback.print_exc()

            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{len(steps)} Q/K frames...", flush=True)

        frame_data.sort(key=lambda x: x[0])

        print(f"Creating Q/K space video/GIF from {len(frame_data)} frames...", flush=True)
        ld_dir = _get_learning_dynamics_dir(config_name_actual)
        video_path = ld_dir / "03_qk_embedding_space.mp4"
        gif_path = ld_dir / "03_qk_embedding_space.gif"

        frames = []
        for _, frame_path in frame_data:
            if frame_path.exists():
                frames.append(imageio.imread(frame_path))
        print(f"Successfully loaded {len(frames)} frames", flush=True)

        if frames:
            frames = _normalize_frame_sizes(frames)
            imageio.mimwrite(str(video_path), frames, fps=fps, codec="libx264", quality=8)
            print(f"Q/K space video saved to: {video_path}", flush=True)

            duration = 1.0 / fps
            imageio.mimwrite(str(gif_path), frames, duration=duration, loop=0)
            print(f"Q/K space GIF saved to: {gif_path}", flush=True)
        else:
            print("Error: No Q/K frames were generated", flush=True)
    finally:
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)
        print("Temporary Q/K space frames cleaned up", flush=True)


def create_qk_space_and_attention_video(
    config_name_actual: str, config: dict, fps: int = 20, max_steps: int | None = None
):
    """Video combining Q/K embedding space and full attention heatmap in a single frame."""
    if not IMAGEIO_AVAILABLE:
        print("Error: imageio is required to create videos. Install with: pip install imageio", flush=True)
        return

    steps = _iter_steps(config_name_actual, config, max_steps)
    if not steps:
        return

    print(f"Creating Q/K+attention video from {len(steps)} checkpoints", flush=True)

    temp_dir = Path(tempfile.mkdtemp())
    frame_data: list[tuple[int, Path]] = []

    try:
        for i, step in enumerate(steps):
            try:
                checkpoint_data = load_checkpoint(config_name_actual, step=step)
            except (FileNotFoundError, OSError) as e:
                print(f"Warning: Could not load checkpoint for step {step}, skipping... ({e})", flush=True)
                continue
            if not checkpoint_data:
                print(f"Warning: Could not load checkpoint for step {step}, skipping...", flush=True)
                continue

            model = checkpoint_data["model"]
            itos = checkpoint_data["itos"]
            frame_path = temp_dir / f"frame_qk_attn_{step:06d}.png"

            try:
                plot_qk_space_and_attention_heatmap(model, itos, save_path=str(frame_path), step_label=step)
                if frame_path.exists():
                    frame_data.append((step, frame_path))
            except Exception as e:  # pragma: no cover - visualization runtime
                print(f"Error generating Q/K+attention frame for step {step}: {e}", flush=True)
                import traceback

                traceback.print_exc()

            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{len(steps)} Q/K+attention frames...", flush=True)

        frame_data.sort(key=lambda x: x[0])

        print(f"Creating Q/K+attention video/GIF from {len(frame_data)} frames...", flush=True)
        ld_dir = _get_learning_dynamics_dir(config_name_actual)
        video_path = ld_dir / "04_qk_space_plus_attention.mp4"
        gif_path = ld_dir / "04_qk_space_plus_attention.gif"

        frames = []
        for _, frame_path in frame_data:
            if frame_path.exists():
                frames.append(imageio.imread(frame_path))
        print(f"Successfully loaded {len(frames)} frames", flush=True)

        if frames:
            frames = _normalize_frame_sizes(frames)
            imageio.mimwrite(str(video_path), frames, fps=fps, codec="libx264", quality=8)
            print(f"Q/K+attention video saved to: {video_path}", flush=True)

            duration = 1.0 / fps
            imageio.mimwrite(str(gif_path), frames, duration=duration, loop=0)
            print(f"Q/K+attention GIF saved to: {gif_path}", flush=True)
        else:
            print("Error: No Q/K+attention frames were generated", flush=True)
    finally:
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)
        print("Temporary Q/K+attention frames cleaned up", flush=True)


def create_output_heatmaps_video(
    config_name_actual: str, config: dict, fps: int = 10, max_steps: int | None = None
):
    """Video of LM head output probability heatmaps overlaid with token+position embeddings."""
    if not IMAGEIO_AVAILABLE:
        print("Error: imageio is required to create videos. Install with: pip install imageio", flush=True)
        return

    steps = _iter_steps(config_name_actual, config, max_steps)
    if not steps:
        return

    print(f"Creating output-heatmap video from {len(steps)} checkpoints", flush=True)

    temp_dir = Path(tempfile.mkdtemp())
    frame_data: list[tuple[int, Path]] = []

    try:
        for i, step in enumerate(steps):
            try:
                checkpoint_data = load_checkpoint(config_name_actual, step=step)
            except (FileNotFoundError, OSError) as e:
                print(f"Warning: Could not load checkpoint for step {step}, skipping... ({e})", flush=True)
                continue
            if not checkpoint_data:
                print(f"Warning: Could not load checkpoint for step {step}, skipping...", flush=True)
                continue

            model = checkpoint_data["model"]
            itos = checkpoint_data["itos"]
            frame_path = temp_dir / f"frame_output_{step:06d}.png"

            try:
                plot_probability_heatmap_with_embeddings(
                    model, itos, save_path=str(frame_path), step_label=step
                )
                if frame_path.exists():
                    frame_data.append((step, frame_path))
            except Exception as e:  # pragma: no cover - visualization runtime
                print(f"Error generating output-heatmap frame for step {step}: {e}", flush=True)
                import traceback

                traceback.print_exc()

            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{len(steps)} output-heatmap frames...", flush=True)

        frame_data.sort(key=lambda x: x[0])

        print(f"Creating output-heatmap video/GIF from {len(frame_data)} frames...", flush=True)
        ld_dir = _get_learning_dynamics_dir(config_name_actual)
        video_path = ld_dir / "05_output_heatmaps_with_embeddings.mp4"
        gif_path = ld_dir / "05_output_heatmaps_with_embeddings.gif"

        frames = []
        for _, frame_path in frame_data:
            if frame_path.exists():
                frames.append(imageio.imread(frame_path))
        print(f"Successfully loaded {len(frames)} frames", flush=True)

        if frames:
            frames = _normalize_frame_sizes(frames)
            imageio.mimwrite(str(video_path), frames, fps=fps, codec="libx264", quality=8)
            print(f"Output-heatmap video saved to: {video_path}", flush=True)

            duration = 1.0 / fps
            imageio.mimwrite(str(gif_path), frames, duration=duration, loop=0)
            print(f"Output-heatmap GIF saved to: {gif_path}", flush=True)
        else:
            print("Error: No output-heatmap frames were generated", flush=True)
    finally:
        import shutil

        shutil.rmtree(temp_dir, ignore_errors=True)
        print("Temporary output-heatmap frames cleaned up", flush=True)

