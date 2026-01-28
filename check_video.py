import imageio.v2 as imageio
try:
    reader = imageio.get_reader('plus_last_even/plots/embedding_qkv_evolution.mp4')
    print(f'Video frames: {reader.count_frames()}')
    meta = reader.get_meta_data()
    print(f'FPS: {meta.get("fps", "unknown")}')
    print(f'Duration: {reader.count_frames() / meta.get("fps", 10):.1f}s')
    reader.close()
except Exception as e:
    print(f'Error reading video: {e}')

# Check checkpoints
from checkpoint import list_available_checkpoints
from config_loader import load_config
config = load_config('plus_last_even')
steps = list_available_checkpoints('plus_last_even')
training_max = config.get('training', {}).get('max_steps', 'unknown')
print(f'\nCheckpoints found: {len(steps)}')
print(f'Training max_steps: {training_max}')
steps_filtered = [s for s in steps if s <= training_max]
print(f'Steps within range: {len(steps_filtered)}')
print(f'Range: {min(steps_filtered)} to {max(steps_filtered)}')
