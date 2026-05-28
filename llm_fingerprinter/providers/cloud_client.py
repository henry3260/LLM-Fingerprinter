"""Cloud provider implementation using Ollama Cloud HTTP API."""

from __future__ import annotations

from llm_fingerprinter.providers.ollama_client import OllamaProvider


class CloudProvider(OllamaProvider):
    def __init__(self, api_key: str, base_url: str = "https://api.ollama.com/v1", timeout: int = 60):
        if not api_key:
            raise ValueError("CloudProvider requires a non-empty api_key")
        super().__init__(base_url=base_url, timeout=timeout)
        self.name = "cloud"
        self._session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        )
