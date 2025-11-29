"""
Shared dataclasses and types for the semantics pipeline.
"""
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class RawParameter:
    """Raw geometric parameter extracted from PointNet++ segmentation."""
    id: str  # e.g., "wing_segment_0_span", "bbox_x_length"
    value: float
    units: Optional[str] = None  # e.g., "m" or "normalized"
    description: str = ""  # brief description of what this dimension is


@dataclass
class CandidateParameter:
    """Candidate semantic parameter proposed by pre-VLM step."""
    name: str
    description: str


@dataclass
class FinalParameter:
    """Final semantic parameter after VLM refinement."""
    name: str
    value: float
    units: Optional[str] = None
    description: str = ""
    confidence: float = 0.0  # 0.0 to 1.0
    raw_sources: List[str] = None  # IDs of RawParameters used

    def __post_init__(self):
        if self.raw_sources is None:
            self.raw_sources = []

