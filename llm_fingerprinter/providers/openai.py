"""OpenAI provider implementation using OpenAI chat completions."""

from __future__ import annotations

from llm_fingerprinter.contracts.llm import LLMRequest, LLMResponse, TokenUsage
from llm_fingerprinter.providers.base import BaseProvider, ProviderCapabilities, validate_request


class OpenAIProvider(BaseProvider):
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1", timeout: int = 60, max_retries: int = 3):
        super().__init__(
            name="openai",
            capabilities=ProviderCapabilities(supports_system_role=True, supports_tools=False, supports_json_mode=False),
        )
        if not api_key:
            raise ValueError("OpenAIProvider requires a non-empty api_key")

        from openai import OpenAI

        self._client = OpenAI(api_key=api_key, base_url=base_url.rstrip("/"), timeout=timeout, max_retries=max_retries)

    def generate(self, request: LLMRequest) -> LLMResponse:
        validate_request(self.name, request)
        response = self._client.chat.completions.create(
            model=request.model,
            messages=[{"role": m.role, "content": m.content} for m in request.messages],
            temperature=getattr(request, "temperature", 0.0),
            max_tokens=getattr(request, "max_tokens", None),
            top_p=getattr(request, "top_p", None),
        )

        choice = response.choices[0] if response.choices else None
        text = choice.message.content.strip() if choice and choice.message.content else ""
        usage = response.usage

        return LLMResponse(
            provider=self.name,
            model=request.model,
            text=text,
            finish_reason=(choice.finish_reason if choice else None),
            usage=TokenUsage(
                input_tokens=(usage.prompt_tokens if usage else 0),
                output_tokens=(usage.completion_tokens if usage else 0),
                total_tokens=(usage.total_tokens if usage else 0),
            ),
            raw=response.model_dump() if hasattr(response, "model_dump") else {},
        )

    def health_check(self) -> bool:
        try:
            list(self._client.models.list())
            return True
        except Exception:
            return False

    def list_models(self) -> list[str]:
        try:
            return sorted(model.id for model in self._client.models.list())
        except Exception:
            return []

    def close(self) -> None:
        self._client.close()
