"""Gemini provider implementation using google-genai SDK."""

from __future__ import annotations

from llm_fingerprinter.contracts.llm import LLMRequest, LLMResponse, TokenUsage
from llm_fingerprinter.providers.base import BaseProvider, ProviderCapabilities, validate_request


class GeminiProvider(BaseProvider):
    def __init__(self, api_key: str, timeout: int = 60, max_retries: int = 3):
        super().__init__(
            name="gemini",
            capabilities=ProviderCapabilities(supports_system_role=True, supports_tools=False, supports_json_mode=False),
        )
        if not api_key:
            raise ValueError("GeminiProvider requires a non-empty api_key")

        from google import genai
        from google.genai import types

        self._types = types
        self._client = genai.Client(api_key=api_key)
        self._timeout = timeout
        self._max_retries = max_retries

    def generate(self, request: LLMRequest) -> LLMResponse:
        validate_request(self.name, request)

        config = self._types.GenerateContentConfig(
            temperature=getattr(request, "temperature", 0.0),
            max_output_tokens=getattr(request, "max_tokens", None),
        )

        messages = getattr(request, "messages", [])
        if messages and messages[0].role == "system":
            config.system_instruction = messages[0].content
            prompt = messages[-1].content if len(messages) > 1 else ""
        else:
            prompt = messages[-1].content if messages else ""

        response = self._client.models.generate_content(model=request.model, contents=prompt, config=config)

        usage = getattr(response, "usage_metadata", None)
        output_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0
        input_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
        total_tokens = input_tokens + output_tokens

        return LLMResponse(
            provider=self.name,
            model=request.model,
            text=(response.text.strip() if response.text else ""),
            usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=total_tokens),
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
            models = []
            for model in self._client.models.list():
                name = model.name[7:] if model.name.startswith("models/") else model.name
                methods = getattr(model, "supported_generation_methods", [])
                if "generateContent" in methods:
                    models.append(name)
            return sorted(models)
        except Exception:
            return []
