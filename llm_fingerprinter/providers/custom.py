"""Custom provider implementation backed by template-driven CustomClient."""

from __future__ import annotations

from llm_fingerprinter.contracts.llm import LLMRequest, LLMResponse, TokenUsage
from llm_fingerprinter.providers.base import BaseProvider, ProviderCapabilities, validate_request


class CustomProvider(BaseProvider):
    """Provider adapter for arbitrary HTTP APIs through a request template file."""

    def __init__(
        self,
        request_file: str,
        api_key: str | None = None,
        timeout: int = 120,
        auth_header: str = "Authorization",
        auth_prefix: str = "Bearer",
        default_model: str | None = None,
        default_temperature: float = 0.7,
        default_max_tokens: int = 512,
        default_system: str | None = None,
        response_path: list | None = None,
    ):
        super().__init__(
            name="custom",
            capabilities=ProviderCapabilities(supports_system_role=True, supports_tools=False, supports_json_mode=False),
        )
        from llm_fingerprinter.custom_client import CustomClient

        self._client = CustomClient(
            request_file=request_file,
            api_key=api_key,
            timeout=timeout,
            auth_header=auth_header,
            auth_prefix=auth_prefix,
            default_model=default_model,
            default_temperature=default_temperature,
            default_max_tokens=default_max_tokens,
            default_system=default_system,
            response_path=response_path,
        )

    def generate(self, request: LLMRequest) -> LLMResponse:
        validate_request(self.name, request)

        messages = getattr(request, "messages", [])
        system_prompt = next((m.content for m in messages if m.role == "system"), None)
        prompt = next((m.content for m in reversed(messages) if m.role == "user"), "")

        text = self._client.generate(
            prompt=prompt,
            model=getattr(request, "model", None),
            temperature=getattr(request, "temperature", None),
            max_tokens=getattr(request, "max_tokens", None),
            system=system_prompt,
        )

        return LLMResponse(
            provider=self.name,
            model=getattr(request, "model", ""),
            text=text or "",
            finish_reason=None,
            usage=TokenUsage(),
            raw={},
        )

    def health_check(self) -> bool:
        return self._client._check_connectivity()

    def list_models(self) -> list[str]:
        return self._client.list_models()

    def close(self) -> None:
        self._client.close()
