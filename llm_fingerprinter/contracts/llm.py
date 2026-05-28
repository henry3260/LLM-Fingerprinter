from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class Message:
    """Single chat message passed to a provider client."""

    role: Role
    content: str


@dataclass(frozen=True)
class TokenUsage:
    """Token accounting across providers."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class LLMRequest:
    """Normalized request payload for all LLM providers."""

    provider: str
    model: str
    messages: list[Message]
    temperature: float = 0.0
    max_tokens: int | None = None
    top_p: float | None = None
    timeout_s: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMResponse:
    """Normalized response payload returned by provider clients."""

    provider: str
    model: str
    text: str
    finish_reason: str | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    raw: dict[str, Any] = field(default_factory=dict)
