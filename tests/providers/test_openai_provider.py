from types import SimpleNamespace

from llm_fingerprinter.contracts.llm import LLMResponse
from llm_fingerprinter.providers.openai import OpenAIProvider


class _Msg:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content


class _Request:
    def __init__(self, model="gpt-test"):
        self.provider = "openai"
        self.model = model
        self.messages = [_Msg("system", "sys"), _Msg("user", "hello")]
        self.temperature = 0.2
        self.max_tokens = 42
        self.top_p = 0.9


class _FakeModels:
    def list(self):
        return [SimpleNamespace(id="gpt-4o"), SimpleNamespace(id="gpt-4.1")]


class _FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        usage = SimpleNamespace(prompt_tokens=5, completion_tokens=7, total_tokens=12)
        choice = SimpleNamespace(message=SimpleNamespace(content=" hi "), finish_reason="stop")
        return SimpleNamespace(choices=[choice], usage=usage, model_dump=lambda: {"ok": True})


class _FakeClient:
    def __init__(self):
        self.models = _FakeModels()
        self.completions = _FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)

    def close(self):
        return None


def test_openai_provider_generate_and_list_models():
    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider.name = "openai"
    provider._client = _FakeClient()

    resp = provider.generate(_Request())
    assert isinstance(resp, LLMResponse)
    assert resp.provider == "openai"
    assert resp.model == "gpt-test"
    assert resp.text == "hi"
    assert resp.usage.total_tokens == 12
    assert provider.list_models() == ["gpt-4.1", "gpt-4o"]
    assert provider.health_check() is True

    call = provider._client.completions.calls[-1]
    assert call["temperature"] == 0.2
    assert call["max_tokens"] == 42
    assert call["top_p"] == 0.9


def test_openai_provider_uses_default_sampling_for_gpt5_models():
    provider = OpenAIProvider.__new__(OpenAIProvider)
    provider.name = "openai"
    provider._client = _FakeClient()

    provider.generate(_Request(model="gpt-5.5"))
    call = provider._client.completions.calls[-1]

    assert call["max_completion_tokens"] == 42
    assert "max_tokens" not in call
    assert "temperature" not in call
    assert "top_p" not in call
