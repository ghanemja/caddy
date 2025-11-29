"""
VLM semantics pipeline for mesh ingestion and semantic parameter extraction.

This module provides:
- VLM client abstraction
- Pre-VLM classification and candidate parameter generation
- Post-VLM parameter refinement
- End-to-end mesh ingestion orchestrator
"""

from .vlm_client import VLMClient, VLMImage, VLMMessage, DummyVLMClient
from .vlm_client_finetuned import FinetunedVLMClient
from .semantics_pre import (
    PreVLMOutput,
    CandidateParameter,
    infer_category_and_candidates,
)
from .semantics_post import (
    PostVLMOutput,
    FinalParameter,
    refine_parameters_with_vlm,
)
from .ingest_mesh import (
    IngestResult,
    ingest_mesh_to_semantic_params,
)
from .types import RawParameter

__all__ = [
    "VLMClient",
    "VLMImage",
    "VLMMessage",
    "DummyVLMClient",
    "FinetunedVLMClient",
    "PreVLMOutput",
    "CandidateParameter",
    "infer_category_and_candidates",
    "PostVLMOutput",
    "FinalParameter",
    "refine_parameters_with_vlm",
    "IngestResult",
    "ingest_mesh_to_semantic_params",
    "RawParameter",
]

