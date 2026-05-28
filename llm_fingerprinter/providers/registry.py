"""Factory helpers for provider construction."""

from __future__ import annotations

from llm_fingerprinter.providers.base import BaseProvider
from llm_fingerprinter.providers.grok import GrokProvider


def create_provider(provider: str, **kwargs) -> BaseProvider:
    provider_key = provider.strip().lower()
    if provider_key == "grok":
        return GrokProvider(**kwargs)
    raise ValueError(f"Unsupported provider: {provider}")
