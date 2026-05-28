from llm_fingerprinter.providers import (
    CustomProvider,
    DeepSeekProvider,
    GeminiProvider,
    GrokProvider,
    OpenAIProvider,
    create_provider,
)


def test_exports_are_available():
    assert create_provider is not None
    assert GrokProvider is not None
    assert OpenAIProvider is not None
    assert DeepSeekProvider is not None
    assert CustomProvider is not None
    assert GeminiProvider is not None


def test_create_provider_rejects_unknown_provider():
    try:
        create_provider("unknown")
    except ValueError as exc:
        assert "Unsupported provider" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown provider")
