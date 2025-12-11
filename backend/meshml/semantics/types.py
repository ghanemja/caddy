"""
Shared dataclasses and types for the semantics pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..parts.parts import PartTable


@dataclass
class RawParameter:
    """Raw geometric parameter extracted from PointNet++ segmentation."""

    id: str  # Generic identifier: "p1", "p2", "p3", ...
    value: float
    units: Optional[str] = None  # e.g., "m" or "normalized"
    description: str = ""  # brief description of what this dimension is
    part_labels: Optional[List[str]] = (
        None  # Optional part labels (e.g., ["wing_left", "wing_right"])
    )

    def __post_init__(self):
        if self.part_labels is None:
            self.part_labels = []


@dataclass
class CandidateParameter:
    """Candidate semantic parameter proposed by pre-VLM step."""

    name: str
    description: str


@dataclass
class FinalParameter:
    """Final semantic parameter after VLM refinement."""

    id: str  # Generic identifier: "p1", "p2", "p3", ...
    semantic_name: str  # Proposed semantic name: "wing_span", "chord_length", etc.
    value: float
    units: Optional[str] = None
    description: str = ""
    confidence: float = 0.0  # 0.0 to 1.0
    raw_sources: List[str] = None  # IDs of RawParameters used (typically just [id])
    part_labels: Optional[List[str]] = (
        None  # Optional part labels (e.g., ["wing", "engine"])
    )

    # Backward compatibility: allow access via .name
    @property
    def name(self) -> str:
        """Backward compatibility: return semantic_name."""
        return self.semantic_name

    def __post_init__(self):
        if self.raw_sources is None:
            self.raw_sources = []
        if self.part_labels is None:
            self.part_labels = []
