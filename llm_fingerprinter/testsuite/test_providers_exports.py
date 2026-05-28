from llm_fingerprinter.providers import create_provider


def test_create_provider_rejects_unknown_provider():
    try:
        create_provider("unknown")
    except ValueError as exc:
        assert "Unsupported provider" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown provider")
