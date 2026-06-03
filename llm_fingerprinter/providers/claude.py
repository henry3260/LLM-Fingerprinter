"""Claude provider implementation using the Anthropic Messages API."""

from __future__ import annotations

from typing import Any

from llm_fingerprinter import config
from llm_fingerprinter.contracts.llm import LLMRequest, LLMResponse, TokenUsage
from llm_fingerprinter.providers.base import BaseProvider, ProviderCapabilities, validate_request


class ClaudeProvider(BaseProvider):
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.anthropic.com",
        timeout: int = 60,
        max_retries: int = 3,
    ):
        super().__init__(
            name="claude",
            capabilities=ProviderCapabilities(
                supports_system_role=True,
                supports_tools=False,
                supports_json_mode=False,
            ),
        )
        if not api_key:
            raise ValueError("ClaudeProvider requires a non-empty api_key")

        from anthropic import Anthropic

        self._client = Anthropic(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            max_retries=max_retries,
        )

    @staticmethod
    def _message_text(response) -> str:
        chunks: list[str] = []
        for block in getattr(response, "content", []) or []:
            block_type = getattr(block, "type", None)
            text = getattr(block, "text", None)
            if isinstance(block, dict):
                block_type = block.get("type", block_type)
                text = block.get("text", text)
            if block_type == "text" and text:
                chunks.append(str(text))
        return "".join(chunks).strip()

    @staticmethod
    def _usage_count(usage, field: str) -> int:
        value = getattr(usage, field, 0) if usage else 0
        if isinstance(usage, dict):
            value = usage.get(field, value)
        return int(value or 0)

    @staticmethod
    def _raw_response(response) -> dict[str, Any]:
        if hasattr(response, "model_dump"):
            return response.model_dump()
        if isinstance(response, dict):
            return dict(response)
        return {}

    def generate(self, request: LLMRequest) -> LLMResponse:
        validate_request(self.name, request)

        system_messages = [
            m.content for m in getattr(request, "messages", []) if m.role == "system"
        ]
        messages = [
            {"role": m.role, "content": m.content}
            for m in getattr(request, "messages", [])
            if m.role != "system"
        ]
        if not messages:
            messages = [{"role": "user", "content": ""}]

        params = {
            "model": request.model,
            "messages": messages,
            "max_tokens": getattr(request, "max_tokens", None) or config.MAX_TOKENS,
        }

        if system_messages:
            params["system"] = "\n".join(system_messages)

        temperature = getattr(request, "temperature", None)
        if temperature is not None:
            params["temperature"] = temperature

        top_p = getattr(request, "top_p", None)
        if top_p is not None:
            params["top_p"] = top_p

        response = self._client.messages.create(**params)
        usage = getattr(response, "usage", None)
        input_tokens = self._usage_count(usage, "input_tokens")
        output_tokens = self._usage_count(usage, "output_tokens")

        return LLMResponse(
            provider=self.name,
            model=request.model,
            text=self._message_text(response),
            finish_reason=getattr(response, "stop_reason", None),
            usage=TokenUsage(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
            raw=self._raw_response(response),
        )

    def health_check(self) -> bool:
        try:
            list(self._model_items())
            return True
        except Exception:
            return False

    def _model_items(self):
        models = self._client.models.list()
        return getattr(models, "data", models)

    def list_models(self) -> list[str]:
        try:
            return sorted(
                model.id
                for model in self._model_items()
                if getattr(model, "id", "")
            )
        except Exception:
            return []

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()
