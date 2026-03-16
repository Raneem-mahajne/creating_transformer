"""Plotting package for embeddings, attention, heatmaps, architecture."""
from plotting._utils import (
    set_journal_mode,
    clear_journal_mode,
    annotate_sequence,
    sparse_ticks,
    collect_epoch_stats,
    get_attention_snapshot_from_X,
    get_multihead_snapshot_from_X,
)

from plotting._heatmap_helpers import (
    plot_heatmaps,
    plot_all_heads_snapshot,
)

_mod_01_architecture_overview = __import__("plotting.01_architecture_overview", fromlist=['plot_architecture_diagram'])
plot_architecture_diagram = _mod_01_architecture_overview.plot_architecture_diagram

_mod_02_training_data_heatmap = __import__("plotting.02_training_data_heatmap", fromlist=['plot_training_data_heatmap'])
plot_training_data_heatmap = _mod_02_training_data_heatmap.plot_training_data_heatmap

_mod_03_learning_curve = __import__("plotting.03_learning_curve", fromlist=['plot_learning_curve', 'estimate_loss', 'estimate_rule_error'])
plot_learning_curve = _mod_03_learning_curve.plot_learning_curve
estimate_loss = _mod_03_learning_curve.estimate_loss
estimate_rule_error = _mod_03_learning_curve.estimate_rule_error

_mod_04_generated_sequences_heatmap = __import__("plotting.04_generated_sequences_heatmap", fromlist=['plot_generated_sequences_heatmap', 'plot_generated_sequences_heatmap_before_after'])
plot_generated_sequences_heatmap = _mod_04_generated_sequences_heatmap.plot_generated_sequences_heatmap
plot_generated_sequences_heatmap_before_after = _mod_04_generated_sequences_heatmap.plot_generated_sequences_heatmap_before_after

_mod_05_token_embeddings = __import__("plotting.05_token_embeddings", fromlist=['plot_token_embeddings_heatmap', 'plot_bigram_logits_heatmap', 'plot_token_embeddings_pca_2d_with_hclust', 'plot_bigram_probability_heatmap_hclust', 'plot_bigram_probability_heatmap', 'plot_embeddings_pca', 'plot_embeddings_scatterplots_only'])
plot_token_embeddings_heatmap = _mod_05_token_embeddings.plot_token_embeddings_heatmap
plot_bigram_logits_heatmap = _mod_05_token_embeddings.plot_bigram_logits_heatmap
plot_token_embeddings_pca_2d_with_hclust = _mod_05_token_embeddings.plot_token_embeddings_pca_2d_with_hclust
plot_bigram_probability_heatmap_hclust = _mod_05_token_embeddings.plot_bigram_probability_heatmap_hclust
plot_bigram_probability_heatmap = _mod_05_token_embeddings.plot_bigram_probability_heatmap
plot_embeddings_pca = _mod_05_token_embeddings.plot_embeddings_pca
plot_embeddings_scatterplots_only = _mod_05_token_embeddings.plot_embeddings_scatterplots_only

_mod_06_output_probability_heatmap = __import__("plotting.06_output_probability_heatmap", fromlist=['plot_probability_heatmap'])
plot_probability_heatmap = _mod_06_output_probability_heatmap.plot_probability_heatmap

_mod_07_output_probs_with_embeddings = __import__("plotting.07_output_probs_with_embeddings", fromlist=['plot_probability_heatmap_with_embeddings'])
plot_probability_heatmap_with_embeddings = _mod_07_output_probs_with_embeddings.plot_probability_heatmap_with_embeddings

_mod_08_qkv_transforms = __import__("plotting.08_qkv_transforms", fromlist=['plot_weights_qkv', 'plot_qkv_transformations'])
plot_weights_qkv = _mod_08_qkv_transforms.plot_weights_qkv
plot_qkv_transformations = _mod_08_qkv_transforms.plot_qkv_transformations

_mod_09_qk_embedding_space = __import__("plotting.09_qk_embedding_space", fromlist=['plot_qk_embedding_space', 'plot_qk_embedding_space_focused_query'])
plot_qk_embedding_space = _mod_09_qk_embedding_space.plot_qk_embedding_space
plot_qk_embedding_space_focused_query = _mod_09_qk_embedding_space.plot_qk_embedding_space_focused_query

_mod_11_qk_full_attention_heatmap = __import__("plotting.11_qk_full_attention_heatmap", fromlist=['plot_qk_full_attention_heatmap', 'plot_qk_full_attention_heatmap_last_row', 'plot_qk_full_attention_combined', 'plot_qk_softmax_attention_heatmap'])
plot_qk_full_attention_heatmap = _mod_11_qk_full_attention_heatmap.plot_qk_full_attention_heatmap
plot_qk_full_attention_heatmap_last_row = _mod_11_qk_full_attention_heatmap.plot_qk_full_attention_heatmap_last_row
plot_qk_full_attention_combined = _mod_11_qk_full_attention_heatmap.plot_qk_full_attention_combined
plot_qk_softmax_attention_heatmap = _mod_11_qk_full_attention_heatmap.plot_qk_softmax_attention_heatmap

_mod_12_probability_heatmap_with_values = __import__("plotting.12_probability_heatmap_with_values", fromlist=['plot_probability_heatmap_with_values'])
plot_probability_heatmap_with_values = _mod_12_probability_heatmap_with_values.plot_probability_heatmap_with_values

_mod_13_sequence_embeddings = __import__("plotting.13_sequence_embeddings", fromlist=['plot_sequence_embeddings'])
plot_sequence_embeddings = _mod_13_sequence_embeddings.plot_sequence_embeddings

_mod_14_qkv_query_key_attention = __import__("plotting.14_qkv_query_key_attention", fromlist=['plot_weights_qkv_two_sequences', 'plot_weights_qkv_single_rows', 'plot_weights_qkv_single'])
plot_weights_qkv_two_sequences = _mod_14_qkv_query_key_attention.plot_weights_qkv_two_sequences
plot_weights_qkv_single_rows = _mod_14_qkv_query_key_attention.plot_weights_qkv_single_rows
plot_weights_qkv_single = _mod_14_qkv_query_key_attention.plot_weights_qkv_single

_mod_15_q_dot_product_gradients = __import__("plotting.15_q_dot_product_gradients", fromlist=['plot_q_dot_product_gradients'])
plot_q_dot_product_gradients = _mod_15_q_dot_product_gradients.plot_q_dot_product_gradients

_mod_17_residuals = __import__("plotting.17_residuals", fromlist=['plot_residuals', 'plot_ffn_second_residual_arrows'])
plot_residuals = _mod_17_residuals.plot_residuals
plot_ffn_second_residual_arrows = _mod_17_residuals.plot_ffn_second_residual_arrows

_mod_18_final_on_output_grid = __import__("plotting.18_final_on_output_grid", fromlist=['plot_final_on_output_heatmap_grid', 'plot_per_token_frozen_output'])
plot_final_on_output_heatmap_grid = _mod_18_final_on_output_grid.plot_final_on_output_heatmap_grid
plot_per_token_frozen_output = _mod_18_final_on_output_grid.plot_per_token_frozen_output

from plotting.attention_matrix import (
    plot_attention_matrix,
    plot_embedding_triplet_matrix,
)

from plotting.embedding_qkv_comprehensive import (
    plot_embedding_qkv_comprehensive,
    plot_tokenpos_qkv_simple,
)

from plotting.lm_head_probability_heatmaps import (
    plot_lm_head_probability_heatmaps,
)

from plotting.token_position_embedding_space import (
    plot_token_position_embedding_space,
)

from plotting.v_before_after_demo import (
    plot_v_before_after_demo_sequences,
)

from plotting.qk_space_and_attention import (
    plot_qk_space_and_attention_heatmap,
)

from plotting.probability_heatmap_with_ffn import (
    plot_probability_heatmap_with_ffn_positions,
)

__all__ = [
    "set_journal_mode",
    "clear_journal_mode",
    "annotate_sequence",
    "sparse_ticks",
    "collect_epoch_stats",
    "get_attention_snapshot_from_X",
    "get_multihead_snapshot_from_X",
    "plot_heatmaps",
    "plot_all_heads_snapshot",
    "plot_architecture_diagram",
    "plot_training_data_heatmap",
    "plot_learning_curve",
    "estimate_loss",
    "estimate_rule_error",
    "plot_generated_sequences_heatmap",
    "plot_generated_sequences_heatmap_before_after",
    "plot_token_embeddings_heatmap",
    "plot_bigram_logits_heatmap",
    "plot_token_embeddings_pca_2d_with_hclust",
    "plot_bigram_probability_heatmap_hclust",
    "plot_bigram_probability_heatmap",
    "plot_embeddings_pca",
    "plot_embeddings_scatterplots_only",
    "plot_probability_heatmap",
    "plot_probability_heatmap_with_embeddings",
    "plot_weights_qkv",
    "plot_qkv_transformations",
    "plot_qk_embedding_space",
    "plot_qk_embedding_space_focused_query",
    "plot_qk_full_attention_heatmap",
    "plot_qk_full_attention_heatmap_last_row",
    "plot_qk_full_attention_combined",
    "plot_qk_softmax_attention_heatmap",
    "plot_probability_heatmap_with_values",
    "plot_sequence_embeddings",
    "plot_weights_qkv_two_sequences",
    "plot_weights_qkv_single_rows",
    "plot_weights_qkv_single",
    "plot_q_dot_product_gradients",
    "plot_residuals",
    "plot_ffn_second_residual_arrows",
    "plot_final_on_output_heatmap_grid",
    "plot_per_token_frozen_output",
    "plot_attention_matrix",
    "plot_embedding_triplet_matrix",
    "plot_embedding_qkv_comprehensive",
    "plot_tokenpos_qkv_simple",
    "plot_lm_head_probability_heatmaps",
    "plot_token_position_embedding_space",
    "plot_v_before_after_demo_sequences",
    "plot_qk_space_and_attention_heatmap",
    "plot_probability_heatmap_with_ffn_positions",
]
