"""Ollama provider implementation using local Ollama HTTP API."""

from __future__ import annotations

import requests

from llm_fingerprinter.contracts.llm import LLMRequest, LLMResponse, TokenUsage
from llm_fingerprinter.providers.base import BaseProvider, ProviderCapabilities, validate_request


class OllamaProvider(BaseProvider):
    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 60):
        super().__init__(
            name="ollama",
            capabilities=ProviderCapabilities(supports_system_role=True, supports_tools=False, supports_json_mode=False),
        )
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._session = requests.Session()

    def generate(self, request: LLMRequest) -> LLMResponse:
        validate_request(self, request)
        payload = {
            "model": request.model,
            "prompt": "\n".join(m.content for m in request.messages if m.role != "system"),
            "stream": False,
            "options": {
                "temperature": getattr(request, "temperature", 0.0),
                "num_predict": getattr(request, "max_tokens", None),
            },
        }

        system_messages = [m.content for m in request.messages if m.role == "system"]
        if system_messages:
            payload["system"] = "\n".join(system_messages)

        response = self._session.post(f"{self._base_url}/api/generate", json=payload, timeout=self._timeout)
        response.raise_for_status()
        body = response.json()

        eval_count = int(body.get("eval_count", 0) or 0)
        prompt_eval_count = int(body.get("prompt_eval_count", 0) or 0)

        return LLMResponse(
            provider=self.name,
            model=request.model,
            text=(body.get("response", "") or "").strip(),
            finish_reason=("stop" if body.get("done") else None),
            usage=TokenUsage(
                input_tokens=prompt_eval_count,
                output_tokens=eval_count,
                total_tokens=prompt_eval_count + eval_count,
            ),
            raw=body,
        )

    def health_check(self) -> bool:
        try:
            resp = self._session.get(f"{self._base_url}/api/tags", timeout=min(self._timeout, 10))
            return resp.status_code == 200
        except Exception:
            return False

    def list_models(self) -> list[str]:
        try:
            resp = self._session.get(f"{self._base_url}/api/tags", timeout=min(self._timeout, 10))
            resp.raise_for_status()
            data = resp.json()
            return sorted(m["name"] for m in data.get("models", []) if "name" in m)
        except Exception:
            return []

    def close(self) -> None:
        self._session.close()
