from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ClassificationResult:
    """Single label score produced by a classifier."""

    label: str
    score: float
    rationale: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MultiClassResult:
    """Top-ranked plus full ranking classification output."""

    top_label: str | None
    top_score: float
    ranking: list[ClassificationResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
