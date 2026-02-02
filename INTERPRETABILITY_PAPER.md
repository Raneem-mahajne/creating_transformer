# Interpretable Minimal Transformers: Geometry as Algorithm

## Abstract

We present a framework for building and interpreting minimal transformer language models trained on procedurally generated integer sequences. By constraining embedding dimension ($n_{\text{embd}}=2$) and head size ($d_k=2$), we enable *full 2D visualization* of every internal representation—embeddings, query/key/value transforms, attention outputs, and the LM head’s decision boundaries. Our central claim: **the learned geometry implies an algorithm**. The arrangement of points and boundaries in $\mathbb{R}^2$ can be read as a step-by-step procedure—e.g., for *plus-last-even*, the model encodes “+” at a query position, attends to keys of the last even number, retrieves its value, and the residual plus LM head map the resulting state into the correct output region. We introduce interpretability tools that make this algorithmic reading explicit: (1) **LM head probability heatmaps** (`lm_head_probability_heatmaps.png`, `probability_heatmap_with_embeddings.png`) showing $P(\text{token} \mid \mathbf{x})$ over the 2D pre-LM-head space; (2) **V before/after demos** (`v_before_after_demo_*.png`, `v_before_after_arrows.png`) tracing value vectors through attention and residual connections overlaid on these landscapes; (3) **Q/K embedding space and full attention heatmaps** (`qk_embedding_space.png`, `qk_full_attention_heatmap.png`) over all token–position combinations; (4) **comprehensive QKV and embedding figures** (`embedding_qkv_comprehensive.png`, `embeddings_scatterplots.png`, `qkv_transformations.png`, `residuals.png`). Training-evolution videos (`embedding_qkv_evolution.mp4`, `embeddings_scatterplots_evolution.mp4`) show how this algorithmic geometry emerges during learning. The framework offers a pedagogical testbed where *seeing the geometry is seeing the algorithm*.

---

## 1. Introduction

Understanding how transformers process sequences remains a central challenge in interpretability and mechanistic analysis. Large-scale models achieve impressive performance but their internal representations are high-dimensional and opaque; we can probe attention or activations, but we rarely get a *complete* picture of how information flows from input to output. At the other extreme, toy models on synthetic tasks are easier to analyze but often lack the full machinery of real transformers—attention over variable context, residual connections, and a learned output head that maps internal state to predictions. We bridge this gap with **minimal transformers**: models that retain the full structure of a decoder-only transformer (token and positional embeddings, single-head self-attention with causal masking, residual connections, feedforward layers, and an LM head) but are constrained to **two-dimensional** embeddings and head dimension. As a result, every internal state—the embedding at each position, the query and key and value vectors, the attention output, the residual sum, and the pre-softmax logit space—lives in $\mathbb{R}^2$. There is no need for PCA, t-SNE, or UMAP; the model’s geometry is directly visible in the plane.

**Geometry implies an algorithm.** In this setting, we argue that the model does not merely *use* geometry as an internal representation; the geometry *is* the algorithm. By “algorithm” we mean a step-by-step procedure that implements the task: for the *plus-last-even* rule (after seeing “+”, output the most recent even number that appeared before it), the procedure is: detect the “+” token, look back over the context to find the last even number, retrieve that number’s representation, and output it. In our minimal transformers, this procedure is legible from the learned 2D layout. (1) *Representation*: Token and position embeddings occupy distinct locations in the plane; the combined embedding space (`embeddings.png`, `token_position_embedding_space.png`) shows how each (token, position) pair is encoded. (2) *Retrieval*: The linear maps $W_Q$ and $W_K$ arrange queries and keys so that the query at a “+” position is close (in dot-product sense) to the keys of even-number positions, with recency—e.g., “most recent”—reflected in the key layout. The scatter plot of all queries and keys (`qk_embedding_space.png`) and the full attention matrix (`qk_full_attention_heatmap.png`) make this “who attends to whom” structure explicit. (3) *Read-out*: Attention weights select the relevant positions; the value vectors at those positions are summed, and the residual connection adds this sum to the current position’s embedding. The figures that show value before attention, after attention, and after residual (`residuals.png`, `v_before_after_demo_*.png`) trace this flow: at the “+” position, the state is *moved* in the plane toward the representation of the last even number. (4) *Output*: The LM head is a linear map plus softmax; it partitions the 2D pre-LM-head space into regions, one per output token. The final state (after residual and feedforward) lands in one of these regions; that region’s token is the prediction. The probability heatmaps (`lm_head_probability_heatmaps.png`, `probability_heatmap_with_embeddings.png`) display these regions and boundaries. Thus the full algorithm—detect “+”, find last even, output it—can be read off from the figures. No black box remains: the geometry *is* the procedure.

Our contributions are as follows. **First**, we adopt an interpretability-first design: $n_{\text{embd}}=2$ and $d_k=2$ with a single attention head, so that every hidden state lies in $\mathbb{R}^2$. The model’s geometry is directly visible, and we argue that this geometry is the algorithm the model implements. **Second**, we introduce and describe a set of novel visualizations, all available under `plus_last_even/plots/`. These include: LM head probability heatmaps that show $P(\text{next}=\text{token} \mid \mathbf{x})$ over a 2D grid, with decision boundaries partitioning the space into output regions; V-before/after demo sequences that overlay the trajectory of value vectors (original, after attention, after residual) on these probability landscapes, with green/red indicating correct or incorrect next-token predictions; the Q/K embedding-space scatter and the full Q/K attention heatmap over all token–position combinations; comprehensive QKV figures that combine embeddings, weight heatmaps, and Q/K/V scatter and dimension heatmaps; residuals figures showing the value after attention, the embeddings, and their sum across example sequences; and training-evolution videos that show how the embedding and Q/K/V geometry evolve frame-by-frame during training. **Third**, we implement a suite of procedural sequence rules (14+ rules, including copy-modulo, successor, parity-based, plus-last-even, plus-means-even, plus-max-of-two) with explicit `generate_sequence`, `verify_sequence`, and `valence_mask` functions, enabling controlled experiments and rule-aware evaluation. The paper is organized so that each visualization is tied to a specific step of the algorithmic reading; we flesh out both the *method* (how we produce the figure) and the *algorithmic reading* (what step of the procedure the figure reveals).

---

## 2. Related Work

**Mechanistic interpretability** aims to reverse-engineer how models compute their outputs. Existing work has focused on probing attention patterns to see which positions attend to which (Clark et al., 2019; Vig, 2019), on circuit analysis to identify minimal subgraphs that implement a behavior (Wang et al., 2022; Olsson et al., 2022), and on representation geometry—e.g., what directions in activation space correspond to what features (Hewitt & Liang, 2019). Our work differs in three ways. (a) We use *minimal* architectures where the full hidden state is 2D, so there is no need to project or reduce dimensions; the entire geometry is visible. (b) We go beyond attention and embeddings to visualize the LM head’s decision boundaries and the trajectory of value vectors through attention and residual connections, so that the path from “current state” to “predicted token” is explicit. (c) We **frame the geometry as the algorithm**: the learned 2D layout is not just a representation of the task; it is directly readable as the step-by-step procedure (encode, retrieve, read-out, output) that the model executes. This “geometry as algorithm” stance is a central claim of the paper.

**Toy models and synthetic tasks** have a long history in interpretability and in studying the computational power of sequence models. RNNs and small transformers have been trained on formal languages, arithmetic, and synthetic sequence tasks (Weiss et al., 2018; Liška et al., 2018; Bhattamishra et al., 2020) to study what kinds of structure they can learn. We extend this tradition with a systematic suite of procedural sequence rules—each with a clear specification (e.g., “after +, output the last even”) and with `verify_sequence` and `valence_mask` for evaluation—and with a visualization pipeline designed specifically so that each figure maps onto an algorithmic step. The result is a testbed where one can train a minimal transformer on a rule and then *read off* the algorithm from the figures.

**2D visualization of neural representations** has been used for word embeddings (Mikolov et al., 2013) and for latent spaces in generative models (van der Maaten & Hinton, 2008). Typically, high-dimensional vectors are projected to 2D for display; information is lost, and the picture is an approximation. Our contribution is to *enforce* 2D structure in the model itself: the embedding dimension and head dimension are set to 2, so that no projection is needed. Every point we plot is the actual model state. We then go further by **interpreting that 2D structure as the algorithm**: the arrangement of points and boundaries in the plane is the procedure the model runs.

---

## 3. Setup

### 3.1 Model

We use a decoder-only transformer with the following components. **Token embedding**: $E \in \mathbb{R}^{V \times 2}$, so each of the $V$ tokens (integers 0–9, “+”, and any special symbols) is mapped to a point in $\mathbb{R}^2$. **Positional embedding**: $P \in \mathbb{R}^{T \times 2}$, so each position index is also mapped to a point in $\mathbb{R}^2$. The input at position $i$ is $\mathbf{x}_i = E_{t_i} + P_i$, the sum of token and positional embeddings; thus each (token, position) pair has a unique location in the plane. **Self-attention**: A single head with $W_Q, W_K, W_V \in \mathbb{R}^{2 \times 2}$. For each position we compute query $\mathbf{q}_i = W_Q \mathbf{x}_i$, key $\mathbf{k}_i = W_K \mathbf{x}_i$, and value $\mathbf{v}_i = W_V \mathbf{x}_i$; attention weights are $\alpha_{ij} \propto \exp(\mathbf{q}_i^\top \mathbf{k}_j / \sqrt{2})$ with a causal mask (zeros for $j > i$), and the attention output at position $i$ is $\sum_j \alpha_{ij} \mathbf{v}_j$. **Residual**: The block output is $\mathbf{x}_i + \text{Attn}(\mathbf{x})_i$, so the state is updated by adding the attention output to the current embedding. **Feedforward**: We apply $\mathbf{x} \mapsto \mathbf{x} + \text{FFN}(\mathbf{x})$ where FFN is a two-layer MLP with ReLU and hidden size $16 \cdot n_{\text{embd}} = 32$; the FFN operates in the same 2D space (input and output dimension 2). **LM head**: The final layer computes $\mathbf{x} W_{\text{lm}}^\top + \mathbf{b}$ to produce logits over the vocabulary, followed by softmax for the next-token distribution.

With $n_{\text{embd}}=2$ and $d_k=2$, every vector in this pipeline—embeddings, queries, keys, values, attention outputs, residual sums, and pre-softmax states—lives in $\mathbb{R}^2$. A schematic of the architecture is provided in `plus_last_even/plots/architecture.png`; training progress (loss and accuracy) is shown in `learning_curve.png`.

### 3.2 Plus-Last-Even Rule

The primary demonstration task in this paper is the **plus-last-even** rule. Sequences are generated over a vocabulary that includes integers 0–9 and a special token “+”. The rule is: **whenever the current token is “+”, the next token must be the *most recent even number* that appeared before that “+”.** For example, in the sequence

```
5, 3, 8, 7, +, 8, 11, 2, 4, +, 4, ...
```

the even numbers before the first “+” are 8 (and 3, 5 are odd); the most recent even is 8, so the token after the first “+” must be 8. Similarly, before the second “+” the evens are 2 and 4; the most recent is 4, so the next token must be 4. The task is operator-conditional (the rule applies only after “+”) and depends on variable-length context (the “last” even could be one step back or many steps back). It therefore requires the model to (1) *detect* that the current position is “+”, (2) *search* backward over the context to find the most recent even number, and (3) *output* that number. The **algorithm** that solves this is exactly what we claim the 2D geometry encodes: the query at the “+” position is aligned with keys of even-number positions (with recency encoded in the key layout so that “last” wins under softmax); attention retrieves the value at that position; the residual adds it to the current state; and the LM head maps the resulting 2D point to the correct output token. The rest of the paper shows how each figure makes one of these steps visible.

### 3.3 Other Rules

Beyond plus-last-even, we implement a suite of procedural sequence rules to support controlled experiments and to test whether the “geometry as algorithm” view holds for other tasks. Examples include: **copy-modulo** (position $i$ must copy the token at position $i \bmod k$ for some $k$), which tests positional and periodic structure; **successor** (next token = current token + 1 mod vocabulary size), which tests simple deterministic transition; **parity-based** rules where the next token’s parity depends on the parities of previous tokens; **plus-means-even** (after “+”, the next token must be any even number, a simpler variant of plus-last-even); **plus-max-of-two** (after “+”, the next token is the maximum of the previous two numbers). Each rule has a procedural `generate_sequence` function for creating training data and a `verify_sequence` function that checks whether a sequence satisfies the rule; we also use `valence_mask` for rule-aware evaluation (e.g., only certain positions are “critical” for the rule). This setup allows us to train minimal transformers on different types of structure—positional, content-based, operator-conditional—and to inspect whether the resulting geometry is again readable as an algorithm.

---

## 4. Interpretability Methods and the Algorithmic Reading

Each visualization in our pipeline serves two roles: it is a *method* (a reproducible way to plot something from the model) and a *window onto the algorithm* (it reveals a specific step of the procedure the model implements). Below we flesh out both aspects for each figure type. All paths are relative to `plus_last_even/plots/`.

### 4.1 LM Head Probability Heatmaps

**Figures**: `lm_head_probability_heatmaps.png`, `probability_heatmap_with_embeddings.png`.

**Goal.** The LM head maps the 2D pre-softmax state (after residual and feedforward) to a distribution over the vocabulary. We want to show how this map *partitions* the 2D space: which regions of the plane lead to high probability for which output token. That partition is the “output” step of the algorithm: “if the state is here, predict that token.”

**Method.** We define a grid over $\mathbb{R}^2$ covering the range of states that occur in practice (or a slightly expanded range). For each grid point $\mathbf{x}$, we pass it through the same path the model uses before the LM head: we apply the feedforward layer (residual + FFN) so that the heatmap reflects the *actual* pre-LM-head space as the model uses it. Then we compute $\text{logits} = \mathbf{x} W_{\text{lm}}^\top + \mathbf{b}$ and $p_d = \exp(\text{logits}_d) / \sum_j \exp(\text{logits}_j)$ for each output token $d$. For each $d$, we plot $p_d$ as a heatmap over the grid. In `lm_head_probability_heatmaps.png` we show one heatmap per token (or a subset); in `probability_heatmap_with_embeddings.png` we overlay the same heatmaps with the actual embedding or state locations for one or more example sequences, so the reader can see where each position lands relative to the decision boundaries.

**Algorithmic reading.** These heatmaps are the *output step* of the algorithm. The LM head is linear (plus softmax), so the decision boundaries between “predict token A” and “predict token B” are linear in the 2D space. High-confidence regions appear as warm zones; boundaries appear as sharp transitions. Correct behavior for plus-last-even means: at the position immediately after “+”, the state (after attention and residual) must land in the region corresponding to the last even number. If we overlay the trajectory of states (e.g., from the V-before/after demos), we can check that the “+” position’s final state indeed falls in the correct token’s region. Thus the heatmaps answer: “Given where the state is, what does the model output?”—and the algorithm’s final step is “output the token whose region contains the current state.”

### 4.2 V Before/After Demo Sequences

**Figures**: `v_before_after_demo_0.png`, `v_before_after_demo_1.png`, `v_before_after_demo_2.png`, `v_before_after_arrows.png`.

**Goal.** We want to trace how the *value* representation at each position is transformed by attention and then by the residual connection, and to show where each position’s state ends up in the probability landscape. This makes the “retrieval and read-out” step of the algorithm visible: we see the state *move* in the plane from “before attention” to “after attention” to “after residual,” and we see whether it lands in the correct output region.

**Method.** For one or more example sequences, we run the model and extract, for each position $i$: (1) $\mathbf{v}_{\text{before}}^{(i)}$, the value vector at $i$ (output of $W_V \mathbf{x}_i$) before any attention; (2) $\mathbf{v}_{\text{after}}^{(i)}$, the attention output at $i$, i.e. $\sum_j \alpha_{ij} \mathbf{v}_j$; (3) $\mathbf{x}_i + \mathbf{v}_{\text{after}}^{(i)}$, the state after the residual connection (which is then passed to the FFN and LM head). We plot these in a structured layout: often a grid where rows correspond to “original V,” “transformed V,” and “V + residual,” and columns correspond to different output-token probability heatmaps, so that we can overlay the 2D coordinates of each position’s state on the relevant heatmap. We annotate each point with a label like `8p3` (token 8 at position 3). For the final row (state after residual), we color the annotation green if the model predicts the correct next token at that position, and red otherwise. The arrow figure (`v_before_after_arrows.png`) emphasizes the *trajectory* from one stage to the next with arrows.

**Algorithmic reading.** These figures show the *retrieval and read-out* in action. At a “+” position, the “before” value is just the value of the “+” token at that position. The “after” value is the attention-weighted sum of values from all previous positions; for a well-trained model, this sum is dominated by the value at the “last even” position. So we see the representation *shift* in the plane from “+” to “last even.” The residual then adds this to the embedding at the “+” position, so the final state is a blend that sits near the “last even” representation—which is exactly what we need for the LM head to output that token. The trajectory from before → after → final is the algorithm’s execution trace in 2D. Green/red coding immediately shows whether the model’s “move” landed in the right output region.

### 4.3 Q/K Embedding Space

**Figure**: `qk_embedding_space.png`.

**Goal.** Attention weights are determined by dot products between queries and keys. We want to visualize how all query and key vectors (for every token–position pair in the vocabulary × position space) are arranged in $\mathbb{R}^2$. Proximity in this space (or alignment, given dot product) indicates which positions will attend to which: if the query at position $i$ is close to the key at position $j$, then $i$ will attend strongly to $j$.

**Method.** For each token $t$ and position $p$ in our grid (typically all vocab tokens and a range of positions), we compute the combined embedding $\mathbf{e}_{t,p} = E_t + P_p$, then the query $\mathbf{q}_{t,p} = W_Q \mathbf{e}_{t,p}$ and key $\mathbf{k}_{t,p} = W_K \mathbf{e}_{t,p}$. We plot all query vectors in one color (e.g., blue) and all key vectors in another (e.g., red), with labels of the form `{token}p{position}`. The result is a scatter plot in $\mathbb{R}^2$ where each point is either a query or a key for some (token, position) pair.

**Algorithmic reading.** This figure encodes *who attends to whom*. For plus-last-even, we expect the query corresponding to “+” at some position to lie near the keys corresponding to even-number tokens at recent positions—and “near” here means high dot product, so strong attention. The key layout must also encode recency so that among the even-number keys, the “most recent” one has the highest dot product with the “+” query. So the geometry directly encodes the “find last even” lookup: the algorithm’s retrieval step is “attend to keys that are close to my query,” and the scatter plot shows how the model has arranged keys and queries so that the right keys are close to the right queries.

### 4.4 Full Q/K Attention Heatmap

**Figure**: `qk_full_attention_heatmap.png`.

**Goal.** The Q/K embedding space shows spatial arrangement; we also want to see the *actual* attention scores (or pre-softmax dot products) for every query–key pair. This gives a $(T \cdot V) \times (T \cdot V)$ matrix (or a suitable subset) where rows are query (token, position) and columns are key (token, position), and the entry is the dot product $q \cdot k / \sqrt{d_k}$ or the softmax weight.

**Method.** We compute the full matrix of dot products between all query and key vectors, applying the causal mask (zero out entries where key position $>$ query position). We optionally apply softmax along rows to get attention weights. The result is displayed as a heatmap: rows = queries, columns = keys; bright entries indicate strong attention.

**Algorithmic reading.** Each row is “from this (token, position), how much do we attend to each (token, position)?” For rows corresponding to “+” at various positions, we expect bright entries in columns corresponding to the last even number at the appropriate position. The heatmap is thus the algorithm’s *attention table*: it explicitly shows the lookup pattern the model uses. One can verify that “+” positions attend primarily to the correct source positions.

### 4.5 Comprehensive QKV and Embedding Figures

**Figures**: `embedding_qkv_comprehensive.png`, `embeddings_scatterplots.png`, `qkv_transformations.png`, `qkv_query_key_attention.png`, `qkv_value_output.png`, `token_position_embedding_space.png`, `residuals.png`.

**Content.** These figures provide a multi-panel view of the full pipeline. **Embeddings**: Token-only, position-only, and combined token+position embeddings as scatter plots (`embeddings.png`, `token_position_embedding_space.png`, `embeddings_scatterplots.png`), so we see how the model encodes (token, position) in 2D. **Weights**: Heatmaps of $W_Q$, $W_K$, $W_V$ show the linear maps that produce Q, K, V from the combined embedding. **Q/K/V transformed**: Scatter plots and sometimes dimension-wise heatmaps (token × position) for the Q-, K-, and V-transformed embeddings (`qkv_transformations.png`, `qkv_query_key_attention.png`, `qkv_value_output.png`). **Residuals**: For a few example sequences, we plot the value vector after attention at each position, the embedding at each position, and their sum (`residuals.png`), showing how the residual connection updates the state. The comprehensive figure (`embedding_qkv_comprehensive.png`) combines several of these in one layout.

**Algorithmic reading.** Together, these show the full pipeline: *encode* (embeddings assign each (token, position) a point) → *query/key* (Q and K layout determines who attends to whom) → *value* (V carries the “content” to be retrieved) → *residual* (state is updated by adding the attention output) → *LM head* (output region). The geometry at each stage is one step of the algorithm; moving through the panels is moving through the algorithm.

### 4.6 Training Evolution Videos

**Figures**: `embedding_qkv_evolution.mp4` (or `.gif`), `embeddings_scatterplots_evolution.mp4` (or `.gif`).

**Content.** We save model checkpoints at regular intervals during training (e.g., every 100 steps). For each checkpoint we generate the same static figures described above (embedding scatterplots, comprehensive QKV figure). These are then concatenated into a video (or GIF) so that the viewer sees the embeddings and Q/K/V geometry evolve frame by frame over training.

**Algorithmic reading.** The videos show the *emergence* of the algorithm. Early in training, the scatter plots are often disorganized—points may be clustered or scattered without a clear structure that implements the rule. As training progresses, the points and boundaries arrange into the structure we have described: queries and keys align so that “+” attends to last even, values and residuals move states into the right regions, and the LM head’s partition becomes clear. The algorithm is not hand-coded; it is the *stable geometry* that gradient descent discovers. Watching the video is watching the model “find” the algorithm in 2D.

---

## 5. Geometry as Algorithm: Summary

For the plus-last-even rule, the model’s behavior can be summarized as a five-step algorithm. Each step is visible in the figures we have described.

1. **Encode.** Map each (token, position) to a 2D point via $\mathbf{e} = E_{\text{token}} + P_{\text{position}}$. The embeddings and token/position scatter plots show this mapping.

2. **Detect “+”.** The current position’s embedding (and hence its query after $W_Q$) identifies “I am at +.” The Q/K scatter and attention heatmap show that “+” queries form a distinct cluster or pattern.

3. **Retrieve last even.** The query at the “+” position has high dot product with keys of even-number positions; recency is reflected in the key layout (e.g., more recent positions have keys that align better with the “+” query) so that softmax attention assigns highest weight to the *last* even. The Q/K embedding space and full attention heatmap make this explicit.

4. **Read value.** The value at the selected position is summed in with attention weights; the residual connection adds this to the current position’s embedding. The result is a 2D state that has been “moved” toward the last even number’s representation. The V-before/after demos and residuals figure show this movement.

5. **Output.** The resulting 2D state (after the feedforward layer) lies in one of the regions defined by the LM head’s linear boundaries. That region corresponds to a token; softmax outputs that token. The probability heatmaps show the regions; the V-before/after overlay shows where the state lands and whether it is the correct region (green) or not (red).

Every step is visible in the figures: embeddings and token/position space → Q/K layout and attention heatmap → value and residuals → probability heatmaps and V-before/after demos. **The geometry is the algorithm.** There is no separate “algorithm” hidden in the weights; the 2D arrangement of points and boundaries *is* the procedure the model runs.

---

## 6. Implementation

**Framework and training.** Models are implemented in PyTorch. Training uses standard next-token prediction (cross-entropy loss) on sequences generated by the procedural rules. Hyperparameters (learning rate, batch size, sequence length, etc.) are set in YAML configuration files (e.g., `plus_last_even.yaml`), so that each rule has a dedicated config specifying the data generator, model size ($n_{\text{embd}}=2$, $d_k=2$), and training schedule.

**Checkpointing.** We save checkpoints at two granularities: (1) at regular step intervals (e.g., every 100 steps) for generating the training-evolution videos; (2) the final checkpoint for producing all static figures (heatmaps, scatter plots, V-before/after demos, etc.). This allows us to both study the emergence of the algorithmic geometry over time and to inspect the fully trained model in detail.

**Evaluation.** Each rule has a `verify_sequence` function that checks whether a sequence satisfies the rule, and a `valence_mask` (or equivalent) that identifies which positions are “critical” for the rule (e.g., only positions immediately after “+” in plus-last-even). We report accuracy on the critical positions and optionally overall next-token accuracy. This ensures that we measure whether the model has learned the intended procedure, not just memorized training sequences.

**Visualization pipeline.** All figures are produced by a plotting module that loads a checkpoint, runs the model on designated sequences (or on a grid for heatmaps), and saves the figures to `plus_last_even/plots/` (or the rule-specific directory). The video scripts concatenate per-checkpoint figures into MP4 or GIF. Reproducibility is achieved by fixing the random seed and the checkpoint path.

---

## 7. Discussion

**Why 2D?** By fixing $n_{\text{embd}}=2$ and $d_k=2$, we sacrifice expressiveness for interpretability. The model must implement the rule in a low-dimensional space; every representation is forced into the plane. This constraint does two things. First, it makes the full geometry visible without any dimensionality reduction, so we can plot every state and every boundary. Second, it encourages the model to find a *compact* solution—there is no room for redundant or tangled representations—which in practice often yields a layout that is readable as a simple algorithm. For many procedural rules (copy-modulo, successor, plus-last-even), 2D suffices to reach high accuracy; the model “compresses” the procedure into the plane. For harder or more expressive tasks, we might need $n_{\text{embd}}>2$, in which case we could use paired 2D projections (e.g., dimensions 0–1 and 2–3) or other techniques to retain interpretability.

**Limitations.** (1) The rules are synthetic and the vocabulary is small; transfer to natural language or to more complex reasoning is unclear. (2) We use a single head and a single block; deeper or wider models would have many more states and would require dimension reduction or other tools to visualize. (3) Some rules may require more than 2 dimensions to learn efficiently; we have not systematically tested the minimal sufficient dimension for each rule. (4) The “algorithmic reading” is a narrative we impose on the geometry; we do not formally prove that the model “implements” that algorithm, though the figures provide strong qualitative evidence.

**Future work.** Natural extensions include: extending to $n_{\text{embd}}=4$ or 8 with paired 2D projections or slicing to retain interpretability; automating the “geometry → algorithm” description (e.g., by summarizing attention patterns and decision boundaries in natural language); comparing the learned algorithmic geometry across different rules to see if there are common motifs; and testing whether interventions (e.g., ablating a direction in the embedding space) produce the predicted change in behavior, as a further validation that the geometry is causal for the algorithm.

---

## 8. Conclusion

We introduce a framework for interpretable minimal transformers in which **the learned 2D geometry implies the algorithm**. By constraining the model to two-dimensional embeddings and head dimension, we make every internal state visible in the plane. We then argue that this geometry is not merely a representation—it *is* the step-by-step procedure the model runs. For the plus-last-even rule, the figures show how the model encodes (token, position), how it arranges queries and keys so that “+” attends to the last even number, how attention and the residual move the state toward that number’s representation, and how the LM head’s partition of the plane maps the final state to the correct output token. Key figures—`embedding_qkv_comprehensive.png`, `qk_embedding_space.png`, `qk_full_attention_heatmap.png`, `v_before_after_demo_*.png`, `probability_heatmap_with_embeddings.png`, and the training-evolution videos—each reveal a part of this procedure. The framework provides a testbed where *seeing the geometry is seeing the algorithm*, and where training-evolution videos show how that algorithmic geometry emerges during learning. We hope this approach will serve as a pedagogical and experimental tool for mechanistic interpretability research.

---

## References

- Bhattamishra et al. (2020). On the computational power of transformers and its implications in sequence modeling. *CoRL*.
- Clark et al. (2019). What does BERT look at? *ACL SRW*.
- Hewitt & Liang (2019). Designing and interpreting probes with control tasks. *EMNLP*.
- Liška et al. (2018). The LAMBADA dataset. *arXiv*.
- Mikolov et al. (2013). Distributed representations of words. *NeurIPS*.
- Olsson et al. (2022). In-context learning and induction heads. *Transformer Circuits*.
- van der Maaten & Hinton (2008). Visualizing data using t-SNE. *JMLR*.
- Vig (2019). A multiscale visualization of attention. *ACL*.
- Wang et al. (2022). Interpretability in the wild. *NeurIPS*.
- Weiss et al. (2018). On the practical computational power of finite precision RNNs. *NeurIPS*.
