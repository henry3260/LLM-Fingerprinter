from llm_fingerprinter.providers import (
    CloudProvider,
    DeepSeekProvider,
    GeminiProvider,
    GrokProvider,
    OllamaProvider,
    OpenAIProvider,
    create_provider,
)


def test_exports_are_available():
    assert create_provider is not None
    assert GrokProvider is not None
    assert OpenAIProvider is not None
    assert OllamaProvider is not None
    assert CloudProvider is not None
    assert DeepSeekProvider is not None
    assert GeminiProvider is not None


def test_create_provider_rejects_unknown_provider():
    try:
        create_provider("unknown")
    except ValueError as exc:
        assert "Unsupported provider" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown provider")


def test_create_provider_ollama_and_cloud_entries():
    ollama_provider = create_provider("ollama")
    cloud_provider = create_provider("cloud", api_key="k")
    cloud_alias_provider = create_provider("ollama-cloud", api_key="k")

    try:
        assert ollama_provider.name == "ollama"
        assert cloud_provider.name == "cloud"
        assert cloud_alias_provider.name == "cloud"
    finally:
        ollama_provider.close()
        cloud_provider.close()
        cloud_alias_provider.close()
