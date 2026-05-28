from types import SimpleNamespace

from llm_fingerprinter.contracts.llm import LLMResponse
from llm_fingerprinter.providers.deepseek import DeepSeekProvider
from llm_fingerprinter.providers.openai import OpenAIProvider


class _Msg:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content


class _Request:
    def __init__(self):
        self.provider = "deepseek"
        self.model = "deepseek-test"
        self.messages = [_Msg("user", "hello")]
        self.temperature = 0.2
        self.max_tokens = 42
        self.top_p = 0.9


class _FakeCompletions:
    def create(self, **kwargs):
        usage = SimpleNamespace(prompt_tokens=2, completion_tokens=3, total_tokens=5)
        choice = SimpleNamespace(message=SimpleNamespace(content=" deep "), finish_reason="stop")
        return SimpleNamespace(choices=[choice], usage=usage, model_dump=lambda: {"provider": "deepseek"})


class _FakeClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=_FakeCompletions())


def test_deepseek_provider_is_openai_compatible_subclass():
    assert issubclass(DeepSeekProvider, OpenAIProvider)


def test_deepseek_provider_sets_provider_name():
    provider = DeepSeekProvider.__new__(DeepSeekProvider)
    provider.name = "openai"
    provider.name = "deepseek"
    assert provider.name == "deepseek"


def test_deepseek_provider_generate_returns_llm_response():
    provider = DeepSeekProvider.__new__(DeepSeekProvider)
    provider.name = "deepseek"
    provider._client = _FakeClient()

    resp = provider.generate(_Request())

    assert isinstance(resp, LLMResponse)
    assert resp.provider == "deepseek"
    assert resp.model == "deepseek-test"
    assert resp.text == "deep"
    assert resp.usage.total_tokens == 5
