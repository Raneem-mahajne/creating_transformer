"""Generate a sequence from the trained model for each seed 1-100."""
import torch, random, numpy as np
from config_loader import load_config
from checkpoint import load_checkpoint, create_decode_from_itos

config = load_config('plus_last_even')
cp = load_checkpoint('plus_last_even')
model = cp['model']
itos = cp['itos']
decode = create_decode_from_itos(itos)
data_config = config['data']
block_size = cp['model_config']['block_size']
vocab_size = cp['vocab_size']

model.eval()

print(f"block_size={block_size}, vocab_size={vocab_size}")
print(f"{'seed':>4}  sequence")
print("-" * 70)

for seed in range(1, 101):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    seq_length = random.randint(block_size, min(block_size + 3, data_config["max_length"]))
    start_token = random.randint(0, vocab_size - 1)
    start = torch.tensor([[start_token]], dtype=torch.long)
    with torch.no_grad():
        sample = model.generate(start, max_new_tokens=seq_length - 1)[0].tolist()
    decoded = decode(sample)
    trunc = decoded[:block_size]
    print(f"{seed:4d}  {' '.join(str(t) for t in trunc)}")
