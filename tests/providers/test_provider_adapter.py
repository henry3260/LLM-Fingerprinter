from llm_fingerprinter.contracts.llm import LLMResponse, TokenUsage
from llm_fingerprinter.provider_adapter import ProviderClientAdapter
from llm_fingerprinter.providers.base import BaseProvider, ProviderCapabilities


class _FakeProvider(BaseProvider):
    def __init__(self):
        super().__init__(name="fake", capabilities=ProviderCapabilities())
        self.requests = []
        self.closed = False

    def generate(self, request):
        self.requests.append(request)
        return LLMResponse(
            provider=request.provider,
            model=request.model,
            text="generated text",
            finish_reason="stop",
            usage=TokenUsage(input_tokens=1, output_tokens=2, total_tokens=3),
        )

    def health_check(self) -> bool:
        return True

    def list_models(self) -> list[str]:
        return ["model-b", "model-a"]

    def close(self) -> None:
        self.closed = True


def test_adapter_builds_normalized_request():
    provider = _FakeProvider()
    client = ProviderClientAdapter(provider, provider_name="fake-alias")

    text = client.generate(
        model="model-a",
        prompt="hello",
        temperature=0.3,
        max_tokens=12,
        system="system prompt",
        top_p=0.9,
    )

    assert text == "generated text"
    request = provider.requests[0]
    assert request.provider == "fake-alias"
    assert request.model == "model-a"
    assert request.temperature == 0.3
    assert request.max_tokens == 12
    assert request.top_p == 0.9
    assert [(m.role, m.content) for m in request.messages] == [
        ("system", "system prompt"),
        ("user", "hello"),
    ]


def test_adapter_exposes_full_response_for_metadata():
    provider = _FakeProvider()
    client = ProviderClientAdapter(provider, provider_name="fake-alias")

    response = client.generate_response(
        model="model-a",
        prompt="hello",
        temperature=0.3,
        max_tokens=12,
        system="system prompt",
        top_p=0.9,
    )

    assert response.text == "generated text"
    assert response.provider == "fake-alias"
    assert response.model == "model-a"
    assert response.finish_reason == "stop"
    assert response.usage.total_tokens == 3


def test_adapter_delegates_health_models_and_close():
    provider = _FakeProvider()
    client = ProviderClientAdapter(provider)

    assert client._check_connectivity() is True
    assert client.list_models() == ["model-b", "model-a"]

    client.close()

    assert provider.closed is True
