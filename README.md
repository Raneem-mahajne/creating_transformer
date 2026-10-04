# Fully Interpretable Minimal Transformers: From Geometry to Algorithm

Raneem Mahajne and Toviah Moldwin
Edmond and Lily Safra Center for Brain Sciences, The Hebrew University of Jerusalem

This repository trains **minimal transformers** — single-layer, single-head, decoder-only models whose embedding dimension and head size are both 2 — and produces a suite of visualizations that expose every internal representation directly in $\mathbb{R}^2$, with no dimensionality reduction.

> **Full paper:** [Paper.md](Paper.md) (source) · [FullyInterpretableMinimalTransformer.pdf](FullyInterpretableMinimalTransformer.pdf) (PDF)

---

## Abstract

We present a framework for building and interpreting minimal transformer models. By constraining a transformer's embedding dimension and head size to $2$, we enable full two-dimensional visualization of its internal representations. Embeddings, query/key/value transforms, attention outputs, residual streams, and decision boundaries can all be seen directly. Our central claim is that the learned geometry implies an algorithm; the arrangement of points and boundaries in $\mathbb{R}^2$ can be read as a step-by-step procedure. We train a transformer on a simple task where it must produce the most recently observed even number whenever the '+' operator appears in a sequence of digits. Once trained, we visually walk through every step of the transformer's computation. We show how the model embeds the tokens and their respective positions in the sequence, transforms them via the Q, K, and V matrices, uses the dot product between the Q and K representations to form the attention matrix, and uses the attention matrix to select values that move the representation of each input token to the region of the domain of the output layer that will correctly predict the next token. We introduce a suite of interpretability visualizations that make the algorithmic interpretation of this procedure explicit. Our framework offers a pedagogical and experimental testbed to explore how transformers use informational geometry to implement next-token prediction.

---

## Quick Start

**Requirements:** Python 3.8+. Core dependencies are in `requirements.txt`; video generation additionally needs `imageio` and `pillow`.

```bash
pip install -r requirements.txt imageio pillow
```

**Train and visualize:**
```bash
python main.py plus_last_even
```

**Visualize from an existing checkpoint (skip training):**
```bash
python main.py plus_last_even --visualize
```

**Visualize a specific training step:**
```bash
python main.py plus_last_even --visualize --step 5000
```

**Regenerate only specific figures, including the A4 versions used in the paper:**
```bash
python main.py plus_last_even --visualize --figure 8 9 --journal
```

**Generate learning-dynamics videos (Movies 1–5):**
```bash
python main.py plus_last_even --video            # all five movies
python main.py plus_last_even --video-qkv        # only the embedding/QKV movie
python main.py plus_last_even --video --fps 30   # custom frame rate (default 20)
```

**Force retrain (overwrite existing checkpoints):**
```bash
python main.py plus_last_even --force-retrain
```

**Rebuild the paper PDF from `Paper.md`** (requires pandoc and pdflatex):
```powershell
.\scripts\build_paper.ps1
```

Figures go to `<config_name>/plots/` (A4 journal versions in `<config_name>/plots/a4/`), movies to `<config_name>/plots/learning_dynamics/`, and checkpoints to `<config_name>/checkpoints/`.

---

## The Plus-Last-Even Task

The vocabulary has 12 tokens: the integers 0–10 and the operator `+`. Whenever `+` occurs, the next token must be the most recent even number that appeared before it; all other positions are unconstrained.

```
5  3  8  7  +  8  10  2  4  +  4  ...
            ↑                 ↑
       last even = 8     last even = 4
```

## Model and Training

| Parameter | Value |
|-----------|-------|
| Embedding dimension ($n_{\mathrm{embed}}$) | 2 |
| Block size ($T$) | 8 |
| Number of heads | 1 |
| Head size ($d_k$) | 2 |
| Vocabulary size ($V$) | 12 (integers 0–10, operator +) |
| Feed-forward hidden size | $16 \times n_{\mathrm{embed}} = 32$ |
| Optimizer | AdamW, learning rate $10^{-3}$ |
| Batch size | 8 |
| Training steps | 20,000 (checkpoint every 100 steps) |
| Training data | 2,000 sequences of length 20–50, `+` probability 0.3 |

The forward pass at position $i$:

$$
\begin{aligned}
\mathbf{e}_i &= \mathbf{x}_i + \mathbf{p}_i, \\
\mathbf{z}_i &= \mathbf{e}_i + \mathrm{Attn}(\mathbf{e})_i, \\
\mathbf{h}_i &= \mathbf{z}_i + \mathrm{FFN}(\mathbf{z}_i), \\
P(t_{i+1} \mid \mathbf{h}_i) &= \mathrm{softmax}\!\left(\mathbf{h}_i \, W_{\mathrm{lm}}^\top + \mathbf{b}\right).
\end{aligned}
$$

A full run of 20,000 steps takes roughly 10–15 minutes on a laptop CPU.

---

## Geometry as Algorithm

The paper shows that the trained model's behavior decomposes into five steps, each visible as a geometric structure (see Section 4 of [Paper.md](Paper.md)):

1. **Encode.** Each (token, position) pair maps to a 2D point $\mathbf{e}_i = \mathbf{x}_i + \mathbf{p}_i$; even numbers, odd numbers, and `+` occupy distinct regions (Figure 5).
2. **Detect the operator.** $W_Q$ maps `+` embeddings to queries that are geometrically distinct from number queries (Figure 7, top).
3. **Retrieve the last even number.** `+` queries have the highest dot products with even-number keys, with recency encoded in the positional layout of the keys (Figures 7, 8, and 13).
4. **Produce attention vectors.** Multiplying the attention matrix by the values selects, for each `+`, the value vector of the most recent even number (Figure 14).
5. **Push embeddings into the correct output zone.** Adding the attention vector to the residual embedding nudges each `+` into the output-landscape zone of the correct even number (Figures 10 and 15).

The demo sequence traced through the pipeline in the paper is `4 1 + 4 6 9 5 +`; the model correctly predicts 4 after the first `+` and 6 after the final `+`.

---

## Paper Figures

All figures are in `plus_last_even/plots/a4/`.

| Paper figure | File | Content |
|---|---|---|
| 1 | `01_architecture_overview.png` | Architecture of the minimal transformer |
| 2 | `02_training_data.png` | Sample training sequences |
| 3 | `03_learning_curve.png` | Loss and rule error over training |
| 4 | `04_generated_sequences.png` | Generations before vs. after training |
| 5 | `05_token_embeddings.png` | Token, position, and combined embeddings |
| 6 | `08_qkv_transforms.png` | Q, K, and V projections |
| 7 | `09_10_qk_space_combined.png` | Joint query–key space and the focused `+` query |
| 8 | `11_1_qk_full_heatmap_last_row.png` | $QK^\top$ scores restricted to `+` queries |
| 9 | `06_output_probs.png` | Per-token output probability landscape |
| 10 | `07_output_landscape_summary.png` | Entropy, argmax map, embedding and value overlays |
| 11 | `13_sequence_embeddings.png` | Demo-sequence embeddings |
| 12 | `14_qk_attention.png` | Demo-sequence queries and keys |
| 13 | `15_q_dot_product_gradients.png` | Demo-sequence dot products and attention matrix |
| 14 | `16_value_output.png` | Demo-sequence values and attention outputs |
| 15 | `17_residuals.png` | Next-token prediction from embeddings plus attention vectors |

### Movies

In `plus_last_even/plots/learning_dynamics/` (GIF and MP4, one frame per checkpoint):

| Movie | File | Content |
|---|---|---|
| 1 | `01_embeddings_scatterplots` | Evolution of token, position, and combined embeddings |
| 2 | `05_output_heatmaps_with_embeddings` | Co-evolution of the output landscape and embeddings |
| 3 | `02_embedding_qkv_comprehensive` | Specialization of the Q, K, and V subspaces |
| 4 | `03_qk_embedding_space` | Separation of query and key subspaces |
| 5 | `04_qk_space_plus_attention` | Evolution of the attention matrix alongside the Q/K scatter |

### Supplementary Figures

| File | Content |
|---|---|
| `supplementary/07_qkv_overview.png` | 3×3 panel of embeddings, Q/K/V spaces, Q+K overlay, and attention output |
| `supplementary/14_attention_matrix.png` | Per-sequence attention matrices, LM-head inputs, logits, and probabilities |
| `supplementary/16_value_arrows.png` | Value vectors (original, transformed, residual) for three demo sequences |
| `extended/08_qkv_transforms_extended.png` | Per-dimension heatmaps for Q, K, and V |
| `a4/11_qk_full_heatmap.png` | Full 96×96 pre-softmax $QK^\top$ matrix |
| `a4/07_output_probs_embed.png` | Output probability heatmaps with embeddings overlaid |
| `a4/12_probability_heatmap_with_values.png` | Output probability heatmaps with value vectors overlaid |

---

## Repository Structure

```
├── main.py                   # Entry point: train, visualize, generate videos
├── model.py                  # The minimal transformer (Head, FeedForward, BigramLanguageModel)
├── training.py               # Training loop, loss estimation, rule-error evaluation
├── data.py                   # Sequence generation, encoding, batching
├── visualize.py              # Orchestrates all figures from a checkpoint
├── plotting/                 # One script per figure (01_architecture_overview.py, ...)
├── video.py                  # Embedding and QKV evolution movies
├── learning_videos.py        # Q/K space, attention, and output-landscape movies
├── checkpoint.py             # Save/load checkpoints
├── config_loader.py          # YAML config loading
├── IntegerStringGenerator.py # Sequence rule definitions
├── configs/                  # One YAML config per task (17 total)
├── plus_last_even/           # Plots and checkpoints for the paper's task
├── scripts/                  # Paper build script (pandoc + pdflatex) and helpers
├── poster/                   # Conference poster
├── Paper.md                  # Paper source
├── paper.tex                 # LaTeX generated from Paper.md
└── FullyInterpretableMinimalTransformer.pdf  # Compiled paper
```

---

## Other Tasks

The paper analyzes `plus_last_even`, but the same pipeline runs on any rule defined in `IntegerStringGenerator.py` with a config in `configs/`:

```bash
python main.py <config_name>
```

| Config | Description |
|--------|-------------|
| `plus_last_even` | After `+`, output the most recent even number |
| `lucky7` | After `7`, output the token that appeared before the `7` |
| `step_back` | Each token is one less than the previous |
| `successor` | Each token is one more than the previous |
| `copy_modulo` | Copy with modular arithmetic |
| `plus_max_of_two` | After `+`, output the max of the two preceding numbers |
| `plus_means_even` | After `+`, output any even number |
| `parity_based` | Next token parity depends on previous parities |
| `lucky7_no_resid` | Lucky7 without residual connections |
| `even_abs_diff` | Absolute difference of consecutive even numbers |
| `conditional_transform` | Transform conditioned on context |
| `lookup_permutation` | Permutation-based lookup |

See `configs/` for the full list.

### Adding a new task

1. Define a rule class in `IntegerStringGenerator.py` with `generate_sequence`, `verify_sequence`, and `valence_mask` methods.
2. Create a YAML config in `configs/` (copy `plus_last_even.yaml` as a template).
3. Run `python main.py <your_config_name>`.

---

## Citation and References

Please cite the paper as given in [Paper.md](Paper.md); the complete reference list is at the end of that file.
