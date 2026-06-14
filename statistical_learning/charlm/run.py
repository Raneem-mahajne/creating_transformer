"""End-to-end character-level statistical-learning experiment.

Pipeline:
  1. Generate a long character corpus from a chosen word regime  -> input.txt
  2. Build the ground-truth trie / DFA                           -> trie.png, dfa.png
  3. Train the shared Transformer (model.py) as a char-level LM  -> model.pt, learning_curve.png
  4. Probe internals: attention heatmaps + hidden-state PCA
  5. Sample from the model                                       -> transformer_samples.txt
  6. Evaluate generated words (validity, frequencies)
  7. Build empirical trie / DFA from the samples                 -> generated_{trie,dfa}.png
  8. Compare true vs generated structure                         -> {trie,dfa}_comparison.png

Run from the repository root, e.g.:
    python -m statistical_learning.charlm.run --regime six_word_overlap
    python -m statistical_learning.charlm.run --regime ten_word_overlap --word-space
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

PACKAGE_DIR = Path(__file__).resolve().parents[2]
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from data import get_batch_from_sequences  # noqa: E402
from model import BigramLanguageModel  # noqa: E402
from training import estimate_loss  # noqa: E402

from statistical_learning.charlm import corpus as corpus_mod  # noqa: E402
from statistical_learning.charlm import viz  # noqa: E402
from statistical_learning.charlm.automata import (  # noqa: E402
    PrefixDFA,
    Trie,
    evaluate_words,
    segment_words,
)
from statistical_learning.charlm.regimes import get_regime  # noqa: E402
from statistical_learning.plotting.learning_curve import plot_learning_curve  # noqa: E402


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def states_after(dfa: PrefixDFA, stream: str) -> list[str]:
    """DFA state after consuming each prefix of the stream."""
    res: list[str] = []
    state = dfa.start
    for ch in stream:
        nxt = dfa.step(state, ch)
        state = nxt if nxt is not None else dfa.start
        res.append(state)
    return res


def train_model(
    model: BigramLanguageModel,
    train_ids: list[int],
    val_ids: list[int],
    block_size: int,
    batch_size: int,
    steps: int,
    lr: float,
    eval_interval: int,
    eval_iterations: int,
) -> dict:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    train_seqs = [train_ids]
    val_seqs = [val_ids]
    history = {"steps": [], "train": [], "val": []}
    for step in range(steps + 1):
        if step % eval_interval == 0 or step == steps:
            losses = estimate_loss(model, train_seqs, val_seqs, block_size, batch_size, eval_iterations)
            history["steps"].append(step)
            history["train"].append(float(losses["train"]))
            history["val"].append(float(losses["validation"]))
            print(f"  step {step:5d} | train {losses['train']:.4f} | val {losses['validation']:.4f}", flush=True)
        if step == steps:
            break
        X, Y = get_batch_from_sequences(train_seqs, block_size, batch_size)
        _, loss = model(X, Y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return history


@torch.no_grad()
def sample_stream(
    model: BigramLanguageModel,
    stoi: dict[str, int],
    itos: dict[int, str],
    n_chars: int,
    word_space: bool,
    start_chars: list[str],
) -> str:
    start_ch = " " if (word_space and " " in stoi) else random.choice(start_chars)
    idx = torch.tensor([[stoi[start_ch]]], dtype=torch.long)
    out = model.generate(idx, max_new_tokens=n_chars)[0].tolist()
    text = corpus_mod.decode(out, itos)
    return text[1:] if word_space else text  # drop the seeded leading space


@torch.no_grad()
def gather_hidden_states(
    model: BigramLanguageModel,
    stream: str,
    stoi: dict[str, int],
    dfa: PrefixDFA,
    block_size: int,
    max_points: int,
    seed: int,
):
    """Collect last-position hidden vectors over full-context windows."""
    ids = corpus_mod.encode(stream, stoi)
    st_after = states_after(dfa, stream)
    rng = random.Random(seed)
    candidates = list(range(block_size - 1, len(stream) - 1))
    if len(candidates) > max_points:
        candidates = rng.sample(candidates, max_points)
        candidates.sort()
    windows = [ids[i - block_size + 1 : i + 1] for i in candidates]
    X = torch.tensor(windows, dtype=torch.long)
    _, hidden, _ = model.features(X)
    H_out = hidden[:, -1, :].cpu().numpy()
    # Input side: token + positional embedding fed into the transformer.
    T = X.shape[1]
    positions = torch.arange(T, device=X.device) % model.block_size
    emb_in = model.token_embedding(X) + model.position_embedding_table(positions)
    H_in = emb_in[:, -1, :].cpu().numpy()
    current = [stream[i] for i in candidates]
    nxt = [stream[i + 1] for i in candidates]
    state = [st_after[i] for i in candidates]
    return H_in, H_out, current, nxt, state


def example_attention(
    model: BigramLanguageModel,
    stream: str,
    stoi: dict[str, int],
    itos: dict[int, str],
    start: int,
    length: int,
):
    ids = corpus_mod.encode(stream[start : start + length], stoi)
    X = torch.tensor([ids], dtype=torch.long)
    _, _, attn = model.features(X)
    chars = [itos[i] for i in ids]
    heads = [attn[h][0].cpu().numpy() for h in range(len(attn))]
    return chars, heads


def main() -> None:
    parser = argparse.ArgumentParser(description="Character-level statistical learning")
    parser.add_argument("--regime", default="six_word_overlap")
    parser.add_argument("--chars", type=int, default=50000)
    parser.add_argument("--context-length", type=int, default=40)
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--word-space", action="store_true")
    parser.add_argument("--n-embd", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=1)
    parser.add_argument("--head-size", type=int, default=2)
    parser.add_argument("--n-layer", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-iterations", type=int, default=50)
    parser.add_argument("--sample-chars", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=18)
    args = parser.parse_args()

    set_seed(args.seed)
    regime = get_regime(args.regime)
    words = list(regime.words)
    word_len = regime.word_len
    block_size = args.context_length

    run_name = regime.name + ("_space" if args.word_space else "")
    out_dir = Path(__file__).resolve().parent / "runs" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"== regime '{regime.name}' (word_space={args.word_space}) -> {out_dir}")
    print(f"   vocabulary: {words}")

    # 1) Corpus ------------------------------------------------------------- #
    text = corpus_mod.generate_corpus(words, args.chars, word_space=args.word_space, seed=args.seed)
    corpus_mod.save_corpus(text, out_dir / "input.txt")
    alphabet, stoi, itos = corpus_mod.build_char_vocab(text)
    vocab_size = len(alphabet)
    print(f"   corpus: {len(text)} chars | alphabet({vocab_size}): {alphabet}")

    # 2) Ground-truth structures ------------------------------------------- #
    true_trie = Trie.from_words(words)
    true_dfa = PrefixDFA.from_words(words, word_space=args.word_space)
    viz.plot_trie(true_trie, out_dir / "trie.png", title=f"True trie: {regime.name}")
    viz.plot_dfa(true_dfa, out_dir / "dfa.png", title=f"True DFA: {regime.name}"
                 + (" (with space)" if args.word_space else ""))

    # 3) Train -------------------------------------------------------------- #
    ids = corpus_mod.encode(text, stoi)
    split = int(0.9 * len(ids))
    train_ids, val_ids = ids[:split], ids[split:]
    model = BigramLanguageModel(
        vocab_size, args.n_embd, block_size, args.num_heads, args.head_size,
        use_residual=True, n_layer=args.n_layer, use_layernorm=True,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   model: n_embd={args.n_embd} heads={args.num_heads} head_size={args.head_size} "
          f"layers={args.n_layer} | {n_params} params")
    print("   training...")
    history = train_model(
        model, train_ids, val_ids, block_size, args.batch_size, args.steps, args.lr,
        args.eval_interval, args.eval_iterations,
    )
    torch.save(model.state_dict(), out_dir / "model.pt")
    model.eval()
    plot_learning_curve(
        steps=history["steps"], train_losses=history["train"], val_losses=history["val"],
        rule_error_history=None, save_path=out_dir / "learning_curve.png",
        eval_interval=args.eval_interval,
    )

    model_config = {
        "vocab_size": vocab_size, "n_embd": args.n_embd, "block_size": block_size,
        "num_heads": args.num_heads, "head_size": args.head_size, "n_layer": args.n_layer,
        "use_layernorm": True, "use_residual": True,
    }
    with open(out_dir / "model_config.json", "w", encoding="utf-8") as f:
        json.dump({"alphabet": alphabet, **model_config}, f, indent=2)

    # 4) Internals: attention + PCA ---------------------------------------- #
    start_chars = sorted({w[0] for w in words})
    # Begin examples at a word boundary for readability.
    ex_len = min(24, block_size)
    chars0, heads0 = example_attention(model, text, stoi, itos, start=0, length=ex_len)
    viz.plot_attention(chars0, heads0, out_dir / "attention_example1.png",
                       title="Self-attention (last layer) — example 1")
    chars1, heads1 = example_attention(model, text, stoi, itos,
                                       start=word_len * 7, length=ex_len)
    viz.plot_attention(chars1, heads1, out_dir / "attention_example2.png",
                       title="Self-attention (last layer) — example 2")

    H_in, H_out, cur, nxt, st = gather_hidden_states(
        model, text, stoi, true_dfa, block_size, max_points=1500, seed=args.seed,
    )
    viz.plot_hidden_2d(H_in, H_out, cur, nxt, st, out_dir / "hidden_pca.png",
                       title=f"Hidden states (2-D, no reduction) — {regime.name}")

    # 5) Sample ------------------------------------------------------------- #
    set_seed(args.seed + 1)
    sample_text = sample_stream(model, stoi, itos, args.sample_chars, args.word_space, start_chars)
    (out_dir / "transformer_samples.txt").write_text(sample_text, encoding="utf-8")

    # 6) Evaluate words ----------------------------------------------------- #
    gen_words = segment_words(sample_text, args.word_space, word_len)
    evaluation = evaluate_words(gen_words, words)
    viz.plot_word_frequencies(evaluation, out_dir / "word_frequencies.png")
    with open(out_dir / "word_frequencies.json", "w", encoding="utf-8") as f:
        json.dump(evaluation, f, indent=2)
    print(f"   generated words: {evaluation['total_words']} | "
          f"valid {evaluation['valid_count']} ({100*evaluation['validity_rate']:.1f}%) | "
          f"invalid {evaluation['invalid_count']}")

    # 7) Empirical structures ---------------------------------------------- #
    observed_words = sorted({w for w in gen_words if w})
    gen_trie = Trie.from_words(observed_words)
    gen_dfa = PrefixDFA.from_words(observed_words, word_space=args.word_space)
    viz.plot_trie(gen_trie, out_dir / "generated_trie.png",
                  title="Generated trie (from Transformer samples)")
    viz.plot_dfa(gen_dfa, out_dir / "generated_dfa.png",
                 title="Generated DFA (from Transformer samples)")

    # 8) Compare ------------------------------------------------------------ #
    viz.plot_trie_comparison(true_trie, gen_trie, out_dir / "trie_comparison.png")
    viz.plot_dfa_comparison(true_dfa, gen_dfa, out_dir / "dfa_comparison.png")

    summary = {
        "regime": regime.name,
        "word_space": args.word_space,
        "vocabulary": words,
        "alphabet": alphabet,
        "corpus_chars": len(text),
        "final_train_loss": history["train"][-1],
        "final_val_loss": history["val"][-1],
        "sample_chars": len(sample_text),
        "validity_rate": evaluation["validity_rate"],
        "valid_count": evaluation["valid_count"],
        "invalid_count": evaluation["invalid_count"],
        "missing_words": sorted(set(words) - set(observed_words)),
        "spurious_words": sorted(set(observed_words) - set(words)),
        "invalid_word_types": list(evaluation["invalid_frequencies"].keys()),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("== summary")
    print(f"   missing words : {summary['missing_words']}")
    print(f"   spurious words: {summary['spurious_words']}")
    print(f"   outputs in    : {out_dir}")


if __name__ == "__main__":
    main()
