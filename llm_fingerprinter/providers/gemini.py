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

    @staticmethod
    def _usage_count(usage, field: str) -> int:
        value = getattr(usage, field, 0) if usage else 0
        return int(value or 0)

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
        output_tokens = self._usage_count(usage, "candidates_token_count")
        input_tokens = self._usage_count(usage, "prompt_token_count")
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

    @staticmethod
    def _model_name(model) -> str:
        raw_name = getattr(model, "name", "") or ""
        if not raw_name and hasattr(model, "model_dump"):
            data = model.model_dump()
            raw_name = data.get("name", "") or data.get("model", "")
        return raw_name[7:] if raw_name.startswith("models/") else raw_name

    @staticmethod
    def _supported_generation_methods(model) -> list[str] | None:
        methods = getattr(model, "supported_generation_methods", None)
        if methods is None and hasattr(model, "model_dump"):
            data = model.model_dump()
            methods = (
                data.get("supported_generation_methods")
                or data.get("supportedGenerationMethods")
            )
        return list(methods) if methods is not None else None

    def list_models(self) -> list[str]:
        try:
            models = []
            for model in self._client.models.list():
                name = self._model_name(model)
                if not name:
                    continue

                methods = self._supported_generation_methods(model)
                if methods is None:
                    if name.startswith("gemini-"):
                        models.append(name)
                    continue

                if "generateContent" in methods:
                    models.append(name)
            return sorted(set(models))
        except Exception:
            return []
