import numpy as np

from llm_fingerprinter import cli as cli_mod
from llm_fingerprinter import config
from llm_fingerprinter.contracts.feature import Feature, FeatureVector
from llm_fingerprinter.contracts.llm import LLMRequest, LLMResponse, TokenUsage
from llm_fingerprinter.fingerprinter import LLMFingerprinter
from llm_fingerprinter.promptgen import PromptItem, PromptPackage
from llm_fingerprinter.providers.base import BaseProvider, ProviderCapabilities


class _DirectProvider(BaseProvider):
    def __init__(self):
        super().__init__(name="direct-provider", capabilities=ProviderCapabilities())
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        user_prompt = next(m.content for m in request.messages if m.role == "user")
        return LLMResponse(
            provider=request.provider,
            model=request.model,
            text=f"direct:{user_prompt}",
            finish_reason="stop",
            usage=TokenUsage(input_tokens=2, output_tokens=3, total_tokens=5),
            raw={"request_provider": request.provider, "message_count": len(request.messages)},
        )

    def health_check(self) -> bool:
        return True

    def list_models(self) -> list[str]:
        return ["model-a"]


class _Suite:
    def __init__(self):
        self.package = PromptPackage(
            prompts=(
                PromptItem(
                    text="alpha prompt",
                    layer="alpha",
                    category="cat-a",
                    system="system prompt",
                ),
            )
        )

    def get_prompt_package(self):
        return self.package

    def __len__(self):
        return len(self.package.prompts)


class _Extractor:
    embedding_dim = 1
    LINGUISTIC_DIM = 1
    BEHAVIORAL_DIM = 1

    def get_feature_dim(self):
        return 3

    @staticmethod
    def feature_vector_to_array(feature_vector):
        return np.array(
            [float(feature.value) for feature in feature_vector.items],
            dtype=np.float32,
        )

    def extract_batch_vectors(self, prompt_response_pairs):
        return [
            FeatureVector(
                items=[
                    Feature(name="embedding_0", value=1),
                    Feature(name="total_chars", value=2),
                    Feature(name="refusal_score", value=3),
                ],
                namespace="test",
                metadata={"source": "direct-provider"},
            )
            for _ in prompt_response_pairs
        ]


def test_fingerprint_model_uses_base_provider_contract(monkeypatch):
    monkeypatch.setattr(config, "LAYER_ORDER", ["alpha"])
    provider = _DirectProvider()
    fingerprinter = LLMFingerprinter(
        "test-endpoint",
        provider,
        _Suite(),
        _Extractor(),
        classifier=None,
    )

    fingerprint = fingerprinter.fingerprint_model("model-a", repeats=1, temperature=0.4)

    assert len(provider.requests) == 1
    request = provider.requests[0]
    assert isinstance(request, LLMRequest)
    assert request.provider == "direct-provider"
    assert request.model == "model-a"
    assert request.temperature == 0.4
    assert request.max_tokens == 512
    assert [(m.role, m.content) for m in request.messages] == [
        ("system", "system prompt"),
        ("user", "alpha prompt"),
    ]
    assert fingerprint["responses_sample"][0]["response"] == "direct:alpha prompt"
    assert fingerprint["responses_sample"][0]["response_metadata"]["raw"] == {
        "request_provider": "direct-provider",
        "message_count": 2,
    }


def test_cli_provider_helpers_use_base_provider_directly(monkeypatch):
    provider = _DirectProvider()
    created = {}

    def fake_create_provider(provider_name, **kwargs):
        created["provider_name"] = provider_name
        created["kwargs"] = kwargs
        return provider

    monkeypatch.setattr(cli_mod, "create_provider", fake_create_provider)

    client = cli_mod.get_api_client(
        backend="ollama",
        endpoint="http://example.test",
        api_key=None,
        request_file=None,
    )
    text = cli_mod.generate_api_text(
        client=client,
        backend="ollama",
        model="model-a",
        prompt="hello",
        max_tokens=12,
        temperature=0.2,
    )

    assert client is provider
    assert cli_mod.check_api_client(client) is True
    assert created == {
        "provider_name": "ollama",
        "kwargs": {"base_url": "http://example.test"},
    }
    assert text == "direct:hello"
    request = provider.requests[0]
    assert request.provider == "direct-provider"
    assert request.max_tokens == 12
    assert request.temperature == 0.2
