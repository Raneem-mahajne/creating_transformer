---
name: Related work catalog
overview: A catalog of 20 papers not yet cited in Paper.md, each with full citation, abstract, a short relevance summary, and a suggested insertion point. This is a shortlist for later citation-only edits; it does not insert any prose into the paper.
todos:
  - id: pick-cites
    content: User picks which of the 20 to add (recommend starting with Olsson, Lindner, Power, Quirke, Park)
    status: pending
  - id: approve-wording
    content: For each chosen cite, draft and get approval of exact BEFORE/AFTER sentences (paper wording rule)
    status: pending
  - id: sync-refs
    content: Add approved bibliography entries to Paper.md and References.md only
    status: pending
isProject: false
---

# Twenty related papers to consider citing

These are **not currently cited**. They are grouped by the best slot in [Paper.md](Paper.md). Abstracts are the authors’ own. After you pick which ones to add, each in-text cite still needs an approved BEFORE/AFTER sentence (paper wording rule).

Do **not** dump all 20 into §1.1; the strongest first five are Olsson, Lindner, Power, Quirke & Barez, and Park.

---

## §1.1 Related work — circuits, toy models, algorithms-as-programs

### 1. Olsson et al. (2022). In-context learning and induction heads

**Citation.** Olsson, C., Elhage, N., Nanda, N., Joseph, N., DasSarma, N., Henighan, T., Mann, B., Askell, A., Bai, Y., Chen, A., Conerly, T., Drain, D., Ganguli, D., Hatfield-Dodds, Z., Hernandez, D., Johnston, S., Jones, A., Kernion, J., Lovitt, L., Ndousse, K., Amodei, D., Bulatov, Y., Clark, J., Kaplan, J., McCandlish, S., & Olah, C. (2022). In-context learning and induction heads. *Transformer Circuits Thread*. [https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) (arXiv:2209.11895)

**Where.** §1.1 (attention circuits) and/or §3.4 (who attends to whom). Plus-last-even is content-dependent retrieval, the same family as induction (look back, copy a value).

**Summary.** Attention heads can implement a copy-from-context algorithm (`[A][B] … [A] → [B]`). That mechanism appears at a training phase change and is a candidate source of in-context learning.

**Abstract.** “Induction heads” are attention heads that implement a simple algorithm to complete token sequences like [A][B] … [A] → [B]. In this work, we present preliminary and indirect evidence for a hypothesis that induction heads might constitute the mechanism for the majority of all “in context learning” in large transformer models (i.e. decreasing loss at increasing token indices). We find that induction heads develop at precisely the same point as a sudden sharp increase in in context learning ability, visible as a bump in the training loss. We present six complementary lines of evidence, arguing that induction heads may be the mechanistic source of general in-context learning in transformer models of any size. For small attention-only models, we present strong, causal evidence; for larger models with MLPs, we present correlational evidence.

### 2. Lindner et al. (2023). Tracr

**Citation.** Lindner, D., Kramár, J., Farquhar, S., Rahtz, M., McGrath, T., & Mikulik, V. (2023). Tracr: Compiled transformers as a laboratory for interpretability. *NeurIPS*. [https://arxiv.org/abs/2301.05062](https://arxiv.org/abs/2301.05062)

**Where.** §1.1, immediately after RASP (Weiss et al., 2021). Contrast: they compile a known program into weights; you reverse-engineer a learned 2D geometry.

**Summary.** Compiler from human-readable programs into decoder-only transformer weights, giving ground-truth circuits (sorting, parentheses, frequencies) and a lab for superposition.

**Abstract.** We show how to "compile" human-readable programs into standard decoder-only transformer models. Our compiler, Tracr, generates models with known structure. This structure can be used to design experiments. For example, we use it to study "superposition" in transformers that execute multi-step algorithms. Additionally, the known structure of Tracr-compiled models can serve as ground-truth for evaluating interpretability methods. Commonly, because the "programs" learned by transformers are unknown it is unclear whether an interpretation succeeded. We demonstrate our approach by implementing and examining programs including computing token frequencies, sorting, and parenthesis checking. We provide an open-source implementation of Tracr at [https://github.com/google-deepmind/tracr](https://github.com/google-deepmind/tracr).

### 3. Friedman, Wettig, & Chen (2023). Learning Transformer Programs

**Citation.** Friedman, D., Wettig, A., & Chen, D. (2023). Learning Transformer Programs. *NeurIPS*. [https://arxiv.org/abs/2306.01128](https://arxiv.org/abs/2306.01128)

**Where.** §1.1 next to RASP/Tracr. Contrast: they constrain the architecture so the net decompiles to Python; you keep a standard GPT block and read the algorithm off 2D geometry.

**Summary.** Train a RASP-like transformer that can be discretized into a human-readable program, so the algorithm is recovered as code rather than as a picture of embeddings.

**Abstract.** Recent research in mechanistic interpretability has attempted to reverse-engineer Transformer models by carefully inspecting network weights and activations. However, these approaches require considerable manual effort and still fall short of providing complete, faithful descriptions of the underlying algorithms. In this work, we introduce a procedure for training Transformers that are mechanistically interpretable by design. We build on RASP [Weiss et al., 2021], a programming language that can be compiled into Transformer weights. Instead of compiling human-written programs into Transformers, we design a modified Transformer that can be trained using gradient-based optimization and then automatically converted into a discrete, human-readable program. We refer to these models as Transformer Programs. To validate our approach, we learn Transformer Programs for a variety of problems, including an in-context learning task, a suite of algorithmic problems (e.g. sorting, recognizing Dyck languages), and NLP tasks including named entity recognition and text classification. The Transformer Programs can automatically find reasonable solutions, performing on par with standard Transformers of comparable size; and, more importantly, they are easy to interpret. To demonstrate these advantages, we convert Transformers into Python programs and use off-the-shelf code analysis tools to debug model errors and identify the “circuits” used to solve different sub-problems. We hope that Transformer Programs open a new path toward the goal of intrinsically interpretable machine learning.

### 4. Zhou et al. (2024). What algorithms can Transformers learn?

**Citation.** Zhou, H., Bradley, A., Littwin, E., Razin, N., Saremi, O., Susskind, J., Bengio, S., & Nakkiran, P. (2024). What algorithms can Transformers learn? A study in length generalization. *ICLR*. [https://arxiv.org/abs/2310.16028](https://arxiv.org/abs/2310.16028)

**Where.** §1.1 with RASP; optional §5 Future directions (other algorithmic tasks).

**Summary.** Length generalization tracks whether a short RASP-L program exists. Fits the claim that transformers implement algorithms, not just fit tables.

**Abstract.** Large language models exhibit surprising emergent generalization properties, yet also struggle on many simple reasoning tasks such as arithmetic and parity. In this work, we focus on length generalization, and we propose a unifying framework to understand when and how Transformers can be expected to length generalize on a given task. First, we show that there exist algorithmic tasks for which standard decoder-only Transformers trained from scratch naturally exhibit strong length generalization. For these tasks, we leverage the RASP programming language (Weiss et al., 2021) to show that the correct algorithmic solution which solves the task can be represented by a simple Transformer. We thus propose the RASP-Generalization Conjecture: Transformers tend to learn a length-generalizing solution if there exists a short RASP-L program that works for all input lengths. We present empirical evidence to support the correlation between RASP-simplicity and generalization. We leverage our insights to give new scratchpad formats which yield strong length generalization on traditionally hard tasks (such as parity and addition), and we illustrate how scratchpad can hinder generalization when it increases the complexity of the corresponding RASP-L program. Overall, our work provides a novel perspective on the mechanisms of length generalization and the algorithmic capabilities of Transformers.

### 5. Power et al. (2022). Grokking

**Citation.** Power, A., Burda, Y., Edwards, H., Babuschkin, I., & Misra, V. (2022). Grokking: Generalization beyond overfitting on small algorithmic datasets. *arXiv*. [https://arxiv.org/abs/2201.02177](https://arxiv.org/abs/2201.02177)

**Where.** §1.1 in the modular-arithmetic / small-algorithmic sentence (Nanda, Gromov, Zhong, …). You already cite the mechanistic sequels; this is the phenomenon paper.

**Summary.** Small transformers on algorithmic data can sit at chance on held-out data long after overfitting, then jump to perfect generalization (“grokking”).

**Abstract.** In this paper we propose to study generalization of neural networks on small algorithmically generated datasets. In this setting, questions about data efficiency, memorization, generalization, and speed of learning can be studied in great detail. In some situations we show that neural networks learn through a process of “grokking” a pattern in the data, improving generalization performance from random chance level to perfect generalization, and that this improvement in generalization can happen well past the point of overfitting. We also study generalization as a function of dataset size and find that smaller datasets require increasing amounts of optimization for generalization. We argue that these datasets provide a fertile ground for studying a poorly understood aspect of deep learning: generalization of overparametrized neural networks beyond memorization of the finite training dataset.

### 6. Chughtai, Chan, & Nanda (2023). A toy model of universality

**Citation.** Chughtai, B., Chan, L., & Nanda, N. (2023). A toy model of universality: Reverse engineering how networks learn group operations. *ICML*. [https://arxiv.org/abs/2302.03025](https://arxiv.org/abs/2302.03025)

**Where.** §1.1 with grokking/geometry papers; §5 Future directions (seeds / what is necessary vs discretionary).

**Summary.** Small nets on group composition implement a representation-theory algorithm. The family of circuits is shared; the exact circuit and learning order are not — mixed evidence for universality.

**Abstract.** Universality is a key hypothesis in mechanistic interpretability – that different models learn similar features and circuits when trained on similar tasks. In this work, we study the universality hypothesis by examining how small networks learn to implement group compositions. We present a novel algorithm by which neural networks may implement composition for any finite group via mathematical representation theory. We then show that these networks consistently learn this algorithm by reverse engineering model logits and weights, and confirm our understanding using ablations. By studying networks trained on various groups and architectures, we find mixed evidence for universality: using our algorithm, we can completely characterize the family of circuits and features that networks learn on this task, but for a given network the precise circuits learned – as well as the order they develop – are arbitrary.

### 7. Quirke & Barez (2024)Quirke & Barez (2024). Understanding addition in transformers

**Citation.** Quirke, P., & Barez, F. (2024). Understanding addition in transformers. *ICLR*. [https://arxiv.org/abs/2310.13121](https://arxiv.org/abs/2310.13121)

**Where.** §1.1 with small algorithmic transformers (closest sibling: one-layer transformer, synthetic numbers, reverse-engineered algorithm).

**Summary.** A one-layer transformer on n-digit addition uses parallel per-digit streams, not human-like right-to-left addition, with a rare high-loss carry failure.

**Abstract.** Understanding the inner workings of machine learning models like Transformers is vital for their safe and ethical use. This paper provides a comprehensive analysis of a one-layer Transformer model trained to perform n-digit integer addition. Our findings suggest that the model dissects the task into parallel streams dedicated to individual digits, employing varied algorithms tailored to different positions within the digits. Furthermore, we identify a rare scenario characterized by high loss, which we explain. By thoroughly elucidating the model's algorithm, we provide new insights into its functioning. These findings are validated through rigorous testing and mathematical modeling, thereby contributing to the broader fields of model understanding and interpretability. Our approach opens the door for analyzing more complex tasks and multi-layer Transformer models.

### 8. Hanna, Liu, & Variengien (2023). How does GPT-2 compute greater-than?

**Citation.** Hanna, M., Liu, O., & Variengien, A. (2023). How does GPT-2 compute greater-than?: Interpreting mathematical abilities in a pre-trained language model. *NeurIPS*. [https://arxiv.org/abs/2305.00586](https://arxiv.org/abs/2305.00586)

**Where.** §1.1 next to Wang 2022 (IOI). Contrast: they find a circuit in GPT-2 small; you walk a full 2D forward pass on a synthetic rule.

**Summary.** Circuit in GPT-2 small for year greater-than: attention finds the start year; late MLPs boost later years.

**Abstract.** Pre-trained language models can be surprisingly adept at tasks they were not explicitly trained on, but how they implement these capabilities is poorly understood. In this paper, we investigate the basic mathematical abilities often acquired by pre-trained language models. Concretely, we use mechanistic interpretability techniques to explain the (limited) mathematical abilities of GPT-2 small. As a case study, we examine its ability to take in sentences such as “The war lasted from the year 1732 to the year 17”, and predict valid two-digit end years (years > 32). We first identify a circuit, a small subset of GPT-2 small’s computational graph that computes this task’s output. Then, we explain the role of each circuit component, showing that GPT-2 small’s final multi-layer perceptrons boost the probability of end years greater than the start year. Finally, we find related tasks that activate our circuit. Our results suggest that GPT-2 small computes greater-than using a complex but general mechanism that activates across diverse contexts.

### 9. Stolfo, Belinkov, & Sachan (2023). Arithmetic reasoning via causal mediation

**Citation.** Stolfo, A., Belinkov, Y., & Sachan, M. (2023). A mechanistic interpretation of arithmetic reasoning in language models using causal mediation analysis. *EMNLP*, 7035–7052. [https://aclanthology.org/2023.emnlp-main.435/](https://aclanthology.org/2023.emnlp-main.435/)

**Where.** §1.1 (arithmetic circuits) or §3.6 (attention moves query info, then residual/FFN write the answer — same information-flow story as plus-last-even, without 2D pictures).

**Summary.** In LMs, attention copies query-relevant info to the last token; MLPs write the numeric result into the residual stream. Also compared to number retrieval (closer to your task).

**Abstract.** Mathematical reasoning in large language models (LMs) has garnered significant attention in recent work, but there is a limited understanding of how these models process and store information related to arithmetic tasks within their architecture. In order to improve our understanding of this aspect of language models, we present a mechanistic interpretation of Transformer-based LMs on arithmetic questions using a causal mediation analysis framework. By intervening on the activations of specific model components and measuring the resulting changes in predicted probabilities, we identify the subset of parameters responsible for specific predictions. This provides insights into how information related to arithmetic is processed by LMs. Our experimental results indicate that LMs process the input by transmitting the information relevant to the query from mid-sequence early layers to the final token using the attention mechanism. Then, this information is processed by a set of MLP modules, which generate result-related information that is incorporated into the residual stream. To assess the specificity of the observed activation dynamics, we compare the effects of different model components on arithmetic queries with other tasks, including number retrieval from prompts and factual knowledge questions.

### 10. Olah et al. (2020). Zoom In: An introduction to circuits

**Citation.** Olah, C., Cammarata, N., Schubert, L., Goh, G., Petrov, M., & Carter, S. (2020). Zoom In: An introduction to circuits. *Distill*. [https://doi.org/10.23915/distill.00024.001](https://doi.org/10.23915/distill.00024.001)

**Where.** §1.1 opening of related work (features + circuits + universality). Distill pieces have no traditional abstract.

**Summary.** Agenda paper: networks have meaningful features, features form interpretable circuits, similar circuits recur across models.

**Abstract.** Distill articles do not include traditional abstracts. The piece proposes three claims as the foundation of the circuits agenda: neural networks contain meaningful features; those features are connected by interpretable circuits; similar features and circuits appear across different models (universality).

### 11. Voita et al. (2019). Analyzing multi-head self-attention

**Citation.** Voita, E., Talbot, D., Moiseev, F., Sennrich, R., & Titov, I. (2019). Analyzing multi-head self-attention: Specialized heads do the heavy lifting, the rest can be pruned. *ACL*, 5797–5808. [https://aclanthology.org/P19-1580/](https://aclanthology.org/P19-1580/)

**Where.** §1.1 with Clark/Vig attention papers. You have one head, so this is background: heads specialize; many are droppable.

**Summary.** Encoder heads have distinct linguistic roles; most can be pruned. Specialized heads survive pruning last.

**Abstract.** Multi-head self-attention is a key component of the Transformer, a state-of-the-art architecture for neural machine translation. In this work we evaluate the contribution made by individual attention heads in the encoder to the overall performance of the model and analyze the roles played by them in the encoder. We find that the most important and confident heads play consistent and often linguistically-interpretable roles. When pruning heads using a method based on stochastic gates and a differentiable relaxation of the L0 penalty, we observe that specialized heads are last to be pruned. Our novel pruning method removes the vast majority of heads without seriously affecting performance. For example, on the English-Russian WMT dataset, pruning 38 out of 48 encoder heads results in a drop of only 0.15 BLEU.

---

## Geometry of representations (intro, §4, embeddings)

### 12. Park, Choe, & Veitch (2024). The linear representation hypothesis

**Citation.** Park, K., Choe, Y. J., & Veitch, V. (2024). The linear representation hypothesis and the geometry of large language models. *ICML*. [https://arxiv.org/abs/2311.03658](https://arxiv.org/abs/2311.03658)

**Where.** Introduction (geometry as algorithm) or §4. You treat directions/regions in \mathbb{R}^2 as the algorithm; this is the formal LLM version of “concepts are directions.”

**Summary.** Formalizes linear representation (input vs output space), links it to probing and steering, and argues the inner product is not naive cosine.

**Abstract.** Informally, the "linear representation hypothesis" is the idea that high-level concepts are represented linearly as directions in some representation space. In this paper, we address two closely related questions: What does "linear representation" actually mean? And, how do we make sense of geometric notions (e.g., cosine similarity and projection) in the representation space? To answer these, we use the language of counterfactuals to give two formalizations of linear representation, one in the output (word) representation space, and one in the input (context) space. We then prove that these connect to linear probing and model steering, respectively. To make sense of geometric notions, we use the formalization to identify a particular (non-Euclidean) inner product that respects language structure in a sense we make precise. Using this causal inner product, we show how to unify all notions of linear representation. In particular, this allows the construction of probes and steering vectors using counterfactual pairs. Experiments with LLaMA-2 demonstrate the existence of linear representations of concepts, the connection to interpretation and control, and the fundamental role of the choice of inner product.

### 13. Li et al. (2023). Emergent world representations (Othello-GPT)

**Citation.** Li, K., Hopkins, A. K., Bau, D., Viégas, F., Pfister, H., & Wattenberg, M. (2023). Emergent world representations: Exploring a sequence model trained on a synthetic task. *ICLR*. [https://arxiv.org/abs/2210.13382](https://arxiv.org/abs/2210.13382)

**Where.** §1.1 synthetic-task transformers; Discussion (internal state vs surface statistics).

**Summary.** GPT trained only to predict legal Othello moves learns an internal board state that can be probed and intervened on.

**Abstract.** Language models show a surprising range of capabilities, but the source of their apparent competence is unclear. Do these networks just memorize a collection of surface statistics, or do they rely on internal representations of the process that generates the sequences they see? We investigate this question in a synthetic setting by applying a variant of the GPT model to the task of predicting legal moves in a simple board game, Othello. Although the network has no a priori knowledge of the game or its rules, we uncover evidence of an emergent nonlinear internal representation of the board state. Interventional experiments indicate this representation can be used to control the output of the network. By leveraging these intervention techniques, we produce “latent saliency maps” that help explain predictions.

### 14. Nanda, Lee, & Wattenberg (2023). Emergent linear representations (Othello)

**Citation.** Nanda, N., Lee, A., & Wattenberg, M. (2023). Emergent linear representations in world models of self-supervised sequence models. *BlackboxNLP*. [https://aclanthology.org/2023.blackboxnlp-1.2/](https://aclanthology.org/2023.blackboxnlp-1.2/)

**Where.** Same slot as Li et al., as the follow-up: board state is linear in a mine/yours/empty basis. Stronger match to your “geometry is the algorithm” claim than the original nonlinear-probe paper.

**Summary.** Othello-GPT’s board is linearly encoded (relative to the current player); simple vector arithmetic steers predictions.

**Abstract.** How do sequence models represent their decision-making process? Prior work suggests that Othello-playing neural networks learned nonlinear models of the board state (Li et al., 2023). In this work, we provide evidence of a closely related linear representation of the board. In particular, we show that probing for “my colour” vs. “opponent’s colour” may be a simple yet powerful way to interpret the model’s internal state. This precise understanding of the internal representations allows us to control the model’s behaviour with simple vector arithmetic. Linear representations enable significant interpretability progress, which we demonstrate with further exploration of how the world model is computed.

---

## Methods / residual stream / output landscape

### 15. Elhage, Lasenby, & Olah (2023). Privileged bases in the residual stream

**Citation.** Elhage, N., Lasenby, R., & Olah, C. (2023). Privileged bases in the transformer residual stream. *Transformer Circuits Thread*. [https://transformer-circuits.pub/2023/privileged-basis/index.html](https://transformer-circuits.pub/2023/privileged-basis/index.html)

**Where.** §2.2 (residual stream) or §5 Limitations (you plot axis-aligned 2D coordinates; theory says residual axes should be arbitrary; Adam can privilege them).

**Summary.** Residual-stream coordinates should be rotation-invariant; in practice they are not. After ruling out LayerNorm and float precision, they point at Adam’s per-dimension scaling.

**Abstract.** Our mathematical theories of the Transformer architecture suggest that individual coordinates in the residual stream should have no special significance (that is, the basis directions should be in some sense "arbitrary" and no more likely to encode information than random directions). Recent work has shown that this observation is false in practice. We investigate this phenomenon and provisionally conclude that the per-dimension normalizers in the Adam optimizer are to blame for the effect. We explore two other obvious sources of basis dependency in a Transformer: Layer normalization, and finite-precision floating-point calculations. We confidently rule these out as being the source of the observed basis-alignment.

### 16. Dar, Geva, Gupta, & Berant (2023). Analyzing transformers in embedding space

**Citation.** Dar, G., Geva, M., Gupta, A., & Berant, J. (2023). Analyzing transformers in embedding space. *ACL*, 16124–16170. [https://aclanthology.org/2023.acl-long.893/](https://aclanthology.org/2023.acl-long.893/)

**Where.** §3.5 next to the logit-lens cites (nostalgebraist, Geva, Belrose). You overlay values on the LM-head plane; they project weights into vocab space without a forward pass.

**Summary.** Interpret transformer parameters by projecting them into vocabulary embedding space; also used to align models that share a tokenizer.

**Abstract.** Understanding Transformer-based models has attracted significant attention, as they lie at the heart of recent technological advances across machine learning. While most interpretability methods rely on running models over inputs, recent work has shown that an input-independent approach, where parameters are interpreted directly without a forward/backward pass is feasible for some Transformer parameters, and for two-layer attention networks. In this work, we present a conceptual framework where all parameters of a trained Transformer are interpreted by projecting them into the embedding space, that is, the space of vocabulary items they operate on. Focusing mostly on GPT-2 for this paper, we provide diverse evidence to support our argument. First, an empirical analysis showing that parameters of both pretrained and fine-tuned models can be interpreted in embedding space. Second, we present two applications of our framework: (a) aligning the parameters of different models that share a vocabulary, and (b) constructing a classifier without training by “translating” the parameters of a fine-tuned classifier to parameters of a different model that was only pretrained. Overall, our findings show that at least in part, we can abstract away model specifics and understand Transformers in the embedding space.

### 17. Phuong & Hutter (2022). Formal algorithms for transformers

**Citation.** Phuong, M., & Hutter, M. (2022). Formal algorithms for transformers. *arXiv*. [https://arxiv.org/abs/2207.09238](https://arxiv.org/abs/2207.09238)

**Where.** §2.2 Model Architecture, next to Vaswani/Radford. Supports “we mean a standard decoder-only block” with compact pseudocode.

**Summary.** Self-contained pseudocode for transformer variants (attention, training, tokenization) aimed at theoreticians and from-scratch implementers.

**Abstract.** This document aims to be a self-contained, mathematically precise overview of transformer architectures and algorithms (not results). It covers what transformers are, how they are trained, what they are used for, their key architectural components, and a preview of the most prominent models. The reader is assumed to be familiar with basic ML terminology and simpler neural network architectures such as MLPs.

---

## Discussion, attention caveats, training dynamics, neuroscience

### 18. Jain & Wallace (2019). Attention is not explanation

**Citation.** Jain, S., & Wallace, B. C. (2019). Attention is not explanation. *NAACL*, 3543–3556. [https://aclanthology.org/N19-1357/](https://aclanthology.org/N19-1357/)

**Where.** §5 Discussion, if you want a caveat: attention weights alone are not an explanation. Your claim is stronger (Q/K geometry + values + residual + LM-head regions), which is the right reply to this paper.

**Summary.** Attention distributions often disagree with gradient importance and can be swapped without changing predictions; do not treat weights as explanations.

**Abstract.** Attention mechanisms have seen wide adoption in neural NLP models. In addition to improving predictive performance, these are often touted as affording transparency: models equipped with attention provide a distribution over attended-to input units, and this is often presented (at least implicitly) as communicating the relative importance of inputs. However, it is unclear what relationship exists between attention weights and model outputs. In this work we perform extensive experiments across a variety of NLP tasks that aim to assess the degree to which attention weights provide meaningful “explanations” for predictions. We find that they largely do not. For example, learned attention weights are frequently uncorrelated with gradient-based measures of feature importance, and one can identify very different attention distributions that nonetheless yield equivalent predictions. Our findings show that standard attention modules do not provide meaningful explanations and should not be treated as though they do.

### 19. Barak et al. (2022). Hidden progress in deep learning

**Citation.** Barak, B., Edelman, B. L., Goel, S., Kakade, S., Malach, E., & Zhang, C. (2022). Hidden progress in deep learning: SGD learns parities near the computational limit. *NeurIPS*. [https://arxiv.org/abs/2207.08799](https://arxiv.org/abs/2207.08799)

**Where.** §3.1 / Movies (geometry forms before rule error hits zero) or §5. Complements grokking: loss can look flat while features amplify.

**Summary.** Nets learning sparse parities show sharp loss jumps; SGD is amplifying a sparse solution under the hood, invisible to train error until a phase change.

**Abstract.** There is mounting evidence of emergent phenomena in the capabilities of deep learning methods as we scale up datasets, model sizes, and training times. While there are some accounts of how these resources modulate statistical capacity, far less is known about their effect on the computational problem of model training. This work conducts such an exploration through the lens of learning a k-sparse parity of n bits, a canonical discrete search problem which is statistically easy but computationally hard. Empirically, we find that a variety of neural networks successfully learn sparse parities, with discontinuous phase transitions in the training curves. On small instances, learning abruptly occurs at approximately n^{O(k)} iterations; this nearly matches SQ lower bounds, despite the apparent lack of a sparse prior. Our theoretical analysis shows that these observations are not explained by a Langevin-like mechanism, whereby SGD "stumbles in the dark" until it finds the hidden set of features (a natural algorithm which also runs in n^{O(k)} time). Instead, we show that SGD gradually amplifies the sparse solution via a Fourier gap in the population gradient, making continual progress that is invisible to loss and error metrics.

### 20. Kriegeskorte & Wei (2021). Neural tuning and representational geometry

**Citation.** Kriegeskorte, N., & Wei, X.-X. (2021). Neural tuning and representational geometry. *Current Opinion in Neurobiology*, 70, 163–173. (arXiv:2104.09743) [https://arxiv.org/abs/2104.09743](https://arxiv.org/abs/2104.09743)

**Where.** §5 neuroscience paragraph, with Caucheteux/Hosseini/Sun/Doerig. This is the methods paper for “geometry is the code,” not another LLM–brain alignment study.

**Summary.** Single-neuron tuning vs population geometry: tuning induces geometry; different tunings can share a geometry; geometry determines what a linear decoder can read out.

**Abstract.** A central goal of neuroscience is to understand the representations formed by brain activity patterns and their connection to behavior. The classical approach is to investigate how individual neurons encode the stimuli and how their tuning determines the fidelity of the neural representation. Tuning analyses often use the Fisher information to characterize the sensitivity of neural responses to small changes of the stimulus. In recent decades, measurements of large populations of neurons have motivated a complementary approach, which focuses on the information available to linear decoders. The decodable information is captured by the geometry of the representational patterns in the multivariate response space. Here we review neural tuning and representational geometry with the goal of clarifying the relationship between them. The tuning induces the geometry, but different sets of tuned neurons can induce the same geometry. The geometry determines the Fisher information, the mutual information, and the behavioral performance of an ideal observer in a range of psychophysical tasks. We argue that future studies can benefit from considering both tuning and geometry to understand neural codes and reveal the connections between stimulus, brain activity, and behavior.

---

## If you only add a handful

- **Must-cite for this paper:** Olsson (retrieval heads), Lindner (Tracr vs learned geometry), Power (grokking source), Quirke & Barez (1-layer numeric transformer), Park (linear geometry).
- **Best Discussion cites:** Jain & Wallace (attention caveat), Kriegeskorte & Wei (neuroscience geometry), Elhage 2023 (privileged residual axes).
- **Skip if §1.1 is already long:** Voita, Phuong & Hutter, Stolfo.

Implementation after you choose: citation-only parentheticals plus matching entries in [Paper.md](Paper.md) and [References.md](References.md); no new prose unless you approve exact sentences.