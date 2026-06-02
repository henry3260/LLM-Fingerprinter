from types import SimpleNamespace

from llm_fingerprinter.contracts.llm import LLMResponse
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
    assert isinstance(resp, LLMResponse)
    assert resp.provider == "gemini"
    assert resp.text == "world"
    assert resp.usage.input_tokens == 3
    assert resp.usage.output_tokens == 4
    assert provider.list_models() == ["gemini-1.5-pro"]
    assert provider.health_check() is True


def test_gemini_provider_list_models_accepts_sdk_dump_and_missing_methods():
    provider = GeminiProvider.__new__(GeminiProvider)
    provider.name = "gemini"
    provider._client = SimpleNamespace(
        models=SimpleNamespace(
            list=lambda: [
                SimpleNamespace(
                    name="models/gemini-2.5-flash",
                    model_dump=lambda: {
                        "name": "models/gemini-2.5-flash",
                        "supportedGenerationMethods": ["generateContent"],
                    },
                ),
                SimpleNamespace(
                    name="models/gemini-2.5-pro",
                    model_dump=lambda: {"name": "models/gemini-2.5-pro"},
                ),
                SimpleNamespace(
                    name="models/text-embedding-004",
                    model_dump=lambda: {"name": "models/text-embedding-004"},
                ),
            ]
        )
    )

    assert provider.list_models() == ["gemini-2.5-flash", "gemini-2.5-pro"]


def test_gemini_provider_treats_missing_usage_counts_as_zero():
    class _ModelsWithPartialUsage:
        def generate_content(self, **kwargs):
            usage = SimpleNamespace(prompt_token_count=5, candidates_token_count=None)
            return SimpleNamespace(text=" ok ", usage_metadata=usage, model_dump=lambda: {"ok": True})

    provider = GeminiProvider.__new__(GeminiProvider)
    provider.name = "gemini"
    provider._types = _FakeTypes()
    provider._client = SimpleNamespace(models=_ModelsWithPartialUsage())

    resp = provider.generate(_Request())

    assert resp.text == "ok"
    assert resp.usage.input_tokens == 5
    assert resp.usage.output_tokens == 0
    assert resp.usage.total_tokens == 5
