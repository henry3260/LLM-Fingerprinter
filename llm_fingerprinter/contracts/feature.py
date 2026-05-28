from dataclasses import dataclass, field
from typing import Any

FeatureValue = float | int | bool | str


@dataclass(frozen=True)
class Feature:
    """Single extracted feature value with optional weighting."""

    name: str
    value: FeatureValue
    weight: float = 1.0


@dataclass(frozen=True)
class FeatureVector:
    """Provider-agnostic structured feature collection."""

    items: list[Feature] = field(default_factory=list)
    namespace: str = "default"
    metadata: dict[str, Any] = field(default_factory=dict)
