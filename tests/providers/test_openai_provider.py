from types import SimpleNamespace

from llm_fingerprinter.contracts.llm import LLMResponse
from llm_fingerprinter.providers.openai import OpenAIProvider


class _Msg:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content


class _Request:
    def __init__(self):
        self.provider = "openai"
        self.model = "gpt-test"
        self.messages = [_Msg("system", "sys"), _Msg("user", "hello")]
        self.temperature = 0.2
        self.max_tokens = 42
        self.top_p = 0.9


class _FakeModels:
    def list(self):
        return [SimpleNamespace(id="gpt-4o"), SimpleNamespace(id="gpt-4.1")]


class _FakeCompletions:
    def create(self, **kwargs):
        usage = SimpleNamespace(prompt_tokens=5, completion_tokens=7, total_tokens=12)
        choice = SimpleNamespace(message=SimpleNamespace(content=" hi "), finish_reason="stop")
        return SimpleNamespace(choices=[choice], usage=usage, model_dump=lambda: {"ok": True})


class _FakeClient:
    def __init__(self):
        self.models = _FakeModels()
        self.chat = SimpleNamespace(completions=_FakeCompletions())

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
