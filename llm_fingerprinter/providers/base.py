"""Provider abstraction layer for normalized LLM interactions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ProviderCapabilities:
    """Optional feature flags supported by a provider implementation."""

    supports_system_role: bool = True
    supports_tools: bool = False
    supports_json_mode: bool = False


class BaseProvider(ABC):
    """Abstract provider using normalized request/response objects."""

    name: str
    capabilities: ProviderCapabilities

    def __init__(self, name: str, capabilities: ProviderCapabilities | None = None):
        self.name = name
        self.capabilities = capabilities or ProviderCapabilities()

    @abstractmethod
    def generate(self, request: Any) -> Any:
        """Execute a generation call for the given request object."""

    @abstractmethod
    def health_check(self) -> bool:
        """Return True when provider backend is reachable."""

    @abstractmethod
    def list_models(self) -> list[str]:
        """Return model ids available to this provider account."""


def validate_request(provider_name: str | BaseProvider, request: Any) -> None:
    """Basic validation to ensure request targets the provider instance."""

    request_provider = getattr(request, "provider", None)

    if isinstance(provider_name, BaseProvider):
        canonical_name = provider_name.name
        aliases = set(getattr(provider_name, "aliases", ()) or ())
    else:
        canonical_name = provider_name
        aliases = set()

    accepted_names = {canonical_name, *aliases}
    if request_provider not in accepted_names:
        raise ValueError(
            f"Request provider '{request_provider}' does not match provider '{canonical_name}'"
        )


def redact_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    """Return metadata safe for logging (best-effort key redaction)."""

    redacted = dict(metadata)
    for key in list(redacted.keys()):
        if "key" in key.lower() or "token" in key.lower() or "secret" in key.lower():
            redacted[key] = "***"
    return redacted
