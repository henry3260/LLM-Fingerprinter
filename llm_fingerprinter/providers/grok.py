"""Grok provider implementation based on xAI OpenAI-compatible endpoint."""

from __future__ import annotations

import logging
import time

from llm_fingerprinter.contracts.llm import LLMRequest, LLMResponse, TokenUsage
from llm_fingerprinter.providers.base import BaseProvider, ProviderCapabilities, validate_request

logger = logging.getLogger(__name__)


class GrokProvider(BaseProvider):
    """Provider adapter for Grok using the OpenAI Python SDK."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.x.ai/v1",
        timeout: int = 60,
        max_retries: int = 3,
    ):
        super().__init__(
            name="grok",
            capabilities=ProviderCapabilities(
                supports_system_role=True,
                supports_tools=False,
                supports_json_mode=False,
            ),
        )

        if not api_key:
            raise ValueError("GrokProvider requires a non-empty api_key")

        try:
            from openai import OpenAI
            import openai as openai_module
        except ImportError as exc:
            raise ImportError("openai package is required. Install with: pip install openai") from exc

        self._openai_module = openai_module
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client = OpenAI(
            api_key=api_key,
            base_url=self._base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    def generate(self, request: LLMRequest) -> LLMResponse:
        validate_request(self.name, request)
        start = time.time()

        messages = [{"role": m.role, "content": m.content} for m in request.messages]

        response = self._client.chat.completions.create(
            model=request.model,
            messages=messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
        )

        elapsed = time.time() - start
        choice = response.choices[0] if response.choices else None
        text = choice.message.content.strip() if choice and choice.message.content else ""

        usage = response.usage
        usage_obj = TokenUsage(
            input_tokens=(usage.prompt_tokens if usage else 0),
            output_tokens=(usage.completion_tokens if usage else 0),
            total_tokens=(usage.total_tokens if usage else 0),
        )

        logger.debug("Grok completion finished in %.2fs", elapsed)

        return LLMResponse(
            provider=self.name,
            model=request.model,
            text=text,
            finish_reason=(choice.finish_reason if choice else None),
            usage=usage_obj,
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
