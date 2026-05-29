"""Adapter exposing provider implementations through the legacy client API."""

from __future__ import annotations

import time
from typing import Optional

from llm_fingerprinter.contracts.llm import LLMRequest, Message
from llm_fingerprinter.providers.base import BaseProvider


class ProviderClientAdapter:
    """Bridge normalized providers into the client API used by the CLI."""

    def __init__(
        self,
        provider: BaseProvider,
        provider_name: Optional[str] = None,
        timeout: int = 60,
        health_check_interval: int = 30,
    ):
        self.provider = provider
        self.provider_name = provider_name or provider.name
        self.timeout = timeout
        self._health_check_interval = health_check_interval
        self._last_health_check: float | None = None
        self._is_healthy = False

    def _check_connectivity(self, force: bool = False) -> bool:
        now = time.time()
        if not force and self._last_health_check is not None:
            if now - self._last_health_check < self._health_check_interval:
                return self._is_healthy

        self._is_healthy = self.provider.health_check()
        self._last_health_check = now
        return self._is_healthy

    def generate(
        self,
        model: Optional[str],
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        system: Optional[str] = None,
        top_p: Optional[float] = None,
        **_: object,
    ) -> str:
        messages = []
        if system:
            messages.append(Message(role="system", content=system))
        messages.append(Message(role="user", content=prompt))

        request = LLMRequest(
            provider=self.provider_name,
            model=model or "",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        return self.provider.generate(request).text

    def list_models(self) -> list[str]:
        return self.provider.list_models()

    def close(self) -> None:
        close = getattr(self.provider, "close", None)
        if callable(close):
            close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
