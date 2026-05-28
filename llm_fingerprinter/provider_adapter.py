"""Adapter exposing provider implementations through the legacy client API."""

from __future__ import annotations

from typing import Optional

from llm_fingerprinter.base_client import BaseClient
from llm_fingerprinter.contracts.llm import LLMRequest, Message
from llm_fingerprinter.providers.base import BaseProvider


class ProviderClientAdapter(BaseClient):
    """Bridge normalized providers into the client API used by the CLI."""

    def __init__(
        self,
        provider: BaseProvider,
        provider_name: Optional[str] = None,
        timeout: int = 60,
    ):
        super().__init__(timeout=timeout)
        self.provider = provider
        self.provider_name = provider_name or provider.name

    def _perform_health_check(self) -> bool:
        return self.provider.health_check()

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
