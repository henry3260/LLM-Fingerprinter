from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FingerprintInput:
    """Input payload for a fingerprint run."""

    target_text: str
    candidate_providers: list[str]
    candidate_models: list[str]
    n_trials: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderScore:
    """Ranked provider/model score entry."""

    provider: str
    model: str | None
    score: float
    evidence: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class FingerprintOutput:
    """Canonical output for provider/model fingerprinting."""

    predicted_provider: str | None
    predicted_model: str | None
    confidence: float
    ranking: list[ProviderScore] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)
