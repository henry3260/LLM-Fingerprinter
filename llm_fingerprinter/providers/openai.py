"""OpenAI provider implementation using OpenAI chat completions."""

from __future__ import annotations

from llm_fingerprinter.contracts.llm import LLMRequest, LLMResponse, TokenUsage
from llm_fingerprinter.providers.base import BaseProvider, ProviderCapabilities, validate_request


_MAX_COMPLETION_TOKEN_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")
_DEFAULT_SAMPLING_MODEL_PREFIXES = _MAX_COMPLETION_TOKEN_MODEL_PREFIXES


def _uses_max_completion_tokens(model: str) -> bool:
    model_key = model.lower()
    return model_key.startswith(_MAX_COMPLETION_TOKEN_MODEL_PREFIXES)


def _requires_default_sampling(model: str) -> bool:
    model_key = model.lower()
    return model_key.startswith(_DEFAULT_SAMPLING_MODEL_PREFIXES)


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
        params = {
            "model": request.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        }

        if not _requires_default_sampling(request.model):
            params["temperature"] = getattr(request, "temperature", 0.0)

        max_tokens = getattr(request, "max_tokens", None)
        if max_tokens is not None:
            token_param = "max_completion_tokens" if _uses_max_completion_tokens(request.model) else "max_tokens"
            params[token_param] = max_tokens

        top_p = getattr(request, "top_p", None)
        if top_p is not None and not _requires_default_sampling(request.model):
            params["top_p"] = top_p

        response = self._client.chat.completions.create(**params)

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
