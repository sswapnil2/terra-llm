"""Micro-TERRA Dual-Path Edge Pre-Processor (`edge_codec/`)."""

from edge_codec.prose_parser import (
    HIGH_ENTROPY_BUCKET,
    LOW_CONFIDENCE_THRESHOLD,
    MODALITY_CODE,
    MODALITY_NONE,
    MODALITY_PROSE,
    OnnxProseCorefParser,
    ProseMention,
    UnifiedEdgeCodec,
    build_default_coref_onnx_bytes,
)

__all__ = [
    "HIGH_ENTROPY_BUCKET",
    "LOW_CONFIDENCE_THRESHOLD",
    "MODALITY_CODE",
    "MODALITY_NONE",
    "MODALITY_PROSE",
    "OnnxProseCorefParser",
    "ProseMention",
    "UnifiedEdgeCodec",
    "build_default_coref_onnx_bytes",
]
