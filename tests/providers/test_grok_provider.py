from types import SimpleNamespace

from llm_fingerprinter.contracts.llm import LLMResponse
from llm_fingerprinter.providers.grok import GrokProvider


class _Msg:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content


class _Request:
    def __init__(self):
        self.provider = "grok"
        self.model = "grok-test"
        self.messages = [_Msg("system", "sys"), _Msg("user", "hello")]
        self.temperature = 0.4
        self.max_tokens = 32
        self.top_p = 0.8


class _FakeModels:
    def list(self):
        return [SimpleNamespace(id="grok-2"), SimpleNamespace(id="grok-1")]


class _FakeCompletions:
    def create(self, **kwargs):
        usage = SimpleNamespace(prompt_tokens=6, completion_tokens=8, total_tokens=14)
        choice = SimpleNamespace(message=SimpleNamespace(content=" grok response "), finish_reason="stop")
        return SimpleNamespace(choices=[choice], usage=usage, model_dump=lambda: {"ok": True})


class _FakeClient:
    def __init__(self):
        self.models = _FakeModels()
        self.chat = SimpleNamespace(completions=_FakeCompletions())

    def close(self):
        return None


def test_grok_provider_generate_and_list_models():
    provider = GrokProvider.__new__(GrokProvider)
    provider.name = "grok"
    provider._client = _FakeClient()

    resp = provider.generate(_Request())

    assert isinstance(resp, LLMResponse)
    assert resp.provider == "grok"
    assert resp.model == "grok-test"
    assert resp.text == "grok response"
    assert resp.finish_reason == "stop"
    assert resp.usage.input_tokens == 6
    assert resp.usage.output_tokens == 8
    assert resp.usage.total_tokens == 14
    assert provider.list_models() == ["grok-1", "grok-2"]
    assert provider.health_check() is True
