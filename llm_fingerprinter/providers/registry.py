"""Factory helpers for provider construction."""

from __future__ import annotations

from llm_fingerprinter.providers.base import BaseProvider
from llm_fingerprinter.providers.deepseek import DeepSeekProvider
from llm_fingerprinter.providers.gemini import GeminiProvider
from llm_fingerprinter.providers.grok import GrokProvider
from llm_fingerprinter.providers.openai import OpenAIProvider


def create_provider(provider: str, **kwargs) -> BaseProvider:
    provider_key = provider.strip().lower()
    if provider_key == "grok":
        return GrokProvider(**kwargs)
    if provider_key == "openai":
        return OpenAIProvider(**kwargs)
    if provider_key == "deepseek":
        return DeepSeekProvider(**kwargs)
    if provider_key == "gemini":
        return GeminiProvider(**kwargs)
    raise ValueError(f"Unsupported provider: {provider}")
