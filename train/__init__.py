"""Micro-TERRA Milestone 3: Ternary QAT, 7:1 Hybrid Recurrence, and 3-Layer Lookahead Distillation."""

from train.bitlinear import (
    BitLinear,
    BitLinearExperts,
    absmean_quantize_ste,
    activation_quantize_int8_ste,
    convert_model_to_bitlinear,
)
from train.hybrid_ssm import (
    GatedLinearRecurrenceAttention,
    HintEmbedding,
    MicroTerraHybridWrapper,
    convert_olmoe_to_hybrid,
)
from train.lookahead_router import (
    LookaheadRouterBank,
    compute_load_balancing_loss,
    compute_lookahead_loss,
    compute_top_k_recall,
)

__all__ = [
    "BitLinear",
    "BitLinearExperts",
    "absmean_quantize_ste",
    "activation_quantize_int8_ste",
    "convert_model_to_bitlinear",
    "GatedLinearRecurrenceAttention",
    "HintEmbedding",
    "MicroTerraHybridWrapper",
    "convert_olmoe_to_hybrid",
    "LookaheadRouterBank",
    "compute_load_balancing_loss",
    "compute_lookahead_loss",
    "compute_top_k_recall",
]
