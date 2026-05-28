from types import SimpleNamespace

from llm_fingerprinter.providers.gemini import GeminiProvider


class _Msg:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content


class _Request:
    def __init__(self):
        self.provider = "gemini"
        self.model = "gemini-test"
        self.messages = [_Msg("system", "sys"), _Msg("user", "hello")]
        self.temperature = 0.5
        self.max_tokens = 64


class _FakeConfig:
    def __init__(self, temperature=None, max_output_tokens=None):
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.system_instruction = None


class _FakeTypes:
    GenerateContentConfig = _FakeConfig


class _FakeModels:
    def generate_content(self, **kwargs):
        usage = SimpleNamespace(prompt_token_count=3, candidates_token_count=4)
        return SimpleNamespace(text=" world ", usage_metadata=usage, model_dump=lambda: {"ok": True})

    def list(self):
        return [
            SimpleNamespace(name="models/gemini-1.5-pro", supported_generation_methods=["generateContent"]),
            SimpleNamespace(name="models/embedding-001", supported_generation_methods=["embedContent"]),
        ]


class _FakeClient:
    def __init__(self):
        self.models = _FakeModels()


def test_gemini_provider_generate_and_list_models():
    provider = GeminiProvider.__new__(GeminiProvider)
    provider.name = "gemini"
    provider._types = _FakeTypes()
    provider._client = _FakeClient()

    resp = provider.generate(_Request())
    assert resp.provider == "gemini"
    assert resp.text == "world"
    assert resp.usage.input_tokens == 3
    assert resp.usage.output_tokens == 4
    assert provider.list_models() == ["gemini-1.5-pro"]
    assert provider.health_check() is True
