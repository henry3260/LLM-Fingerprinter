from types import SimpleNamespace

from llm_fingerprinter import config
from llm_fingerprinter.contracts.llm import LLMResponse
from llm_fingerprinter.providers.claude import ClaudeProvider


class _Msg:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content


class _Request:
    def __init__(self):
        self.provider = "claude"
        self.model = "claude-test"
        self.messages = [_Msg("system", "sys"), _Msg("user", "hello")]
        self.temperature = 0.4
        self.max_tokens = 32
        self.top_p = 0.8


class _FakeMessages:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        usage = SimpleNamespace(input_tokens=6, output_tokens=8)
        content = [SimpleNamespace(type="text", text=" claude response ")]
        return SimpleNamespace(
            content=content,
            stop_reason="end_turn",
            usage=usage,
            model_dump=lambda: {"provider": "claude"},
        )


class _FakeModels:
    def list(self):
        return SimpleNamespace(
            data=[
                SimpleNamespace(id="claude-3-5-sonnet-latest"),
                SimpleNamespace(id="claude-3-haiku-20240307"),
            ]
        )


class _FakeClient:
    def __init__(self):
        self.messages = _FakeMessages()
        self.models = _FakeModels()
        self.closed = False

    def close(self):
        self.closed = True


def test_claude_provider_generate_and_list_models():
    provider = ClaudeProvider.__new__(ClaudeProvider)
    provider.name = "claude"
    provider._client = _FakeClient()

    resp = provider.generate(_Request())

    assert isinstance(resp, LLMResponse)
    assert resp.provider == "claude"
    assert resp.model == "claude-test"
    assert resp.text == "claude response"
    assert resp.finish_reason == "end_turn"
    assert resp.usage.input_tokens == 6
    assert resp.usage.output_tokens == 8
    assert resp.usage.total_tokens == 14
    assert provider.list_models() == [
        "claude-3-5-sonnet-latest",
        "claude-3-haiku-20240307",
    ]
    assert provider.health_check() is True

    call = provider._client.messages.calls[-1]
    assert call["model"] == "claude-test"
    assert call["system"] == "sys"
    assert call["messages"] == [{"role": "user", "content": "hello"}]
    assert call["temperature"] == 0.4
    assert call["max_tokens"] == 32
    assert call["top_p"] == 0.8


def test_claude_provider_uses_default_max_tokens_when_missing():
    provider = ClaudeProvider.__new__(ClaudeProvider)
    provider.name = "claude"
    provider._client = _FakeClient()
    request = _Request()
    request.max_tokens = None

    provider.generate(request)

    call = provider._client.messages.calls[-1]
    assert call["max_tokens"] == config.MAX_TOKENS


def test_claude_family_is_available_for_simulation():
    assert "claude" in config.MODEL_FAMILIES
