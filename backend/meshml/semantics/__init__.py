"""
VLM semantics pipeline for mesh ingestion and semantic parameter extraction.

This module provides:
- VLM client abstraction
- Pre-VLM classification and candidate parameter generation
- Post-VLM parameter refinement
- End-to-end mesh ingestion orchestrator
"""

from .vlm_client import VLMClient, VLMImage, VLMMessage, DummyVLMClient
from .vlm_client_ollama import OllamaVLMClient
# FinetunedVLMClient moved to backend/app/services/ (depends on optim.py)
# Import it from there: from app.services.vlm_client_finetuned import FinetunedVLMClient
from .semantics_pre import (
    PreVLMOutput,
    CandidateParameter,
    infer_category_and_candidates,
)
from .semantics_post import (
    PostVLMOutput,
    FinalParameter,
    refine_parameters_with_vlm,
    build_user_review_payload,
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
    "OllamaVLMClient",
    "PreVLMOutput",
    "CandidateParameter",
    "infer_category_and_candidates",
    "PostVLMOutput",
    "FinalParameter",
    "refine_parameters_with_vlm",
    "build_user_review_payload",
    "IngestResult",
    "ingest_mesh_to_semantic_params",
    "RawParameter",
]

