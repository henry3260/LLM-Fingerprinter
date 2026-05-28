from __future__ import annotations

"""Shared typed contracts for cross-module communication.

These DTOs define stable payload formats exchanged between providers,
fingerprinting pipeline components, and classifiers.
"""

from .llm import Message, TokenUsage, LLMRequest, LLMResponse
from .feature import Feature, FeatureVector, FeatureValue
from .fingerprint import FingerprintInput, ProviderScore, FingerprintOutput
from .classify import ClassificationResult, MultiClassResult

__all__ = [
    "Message",
    "TokenUsage",
    "LLMRequest",
    "LLMResponse",
    "Feature",
    "FeatureValue",
    "FeatureVector",
    "FingerprintInput",
    "ProviderScore",
    "FingerprintOutput",
    "ClassificationResult",
    "MultiClassResult",
]
