from llm_fingerprinter.providers.cloud_client import CloudProvider


def test_cloud_provider_requires_api_key():
    try:
        CloudProvider(api_key="")
    except ValueError as exc:
        assert "requires a non-empty api_key" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty api_key")


def test_cloud_provider_sets_auth_and_name():
    provider = CloudProvider(api_key="secret")
    try:
        assert provider.name == "cloud"
        assert provider._session.headers["Authorization"] == "Bearer secret"
        assert provider._session.headers["Content-Type"] == "application/json"
    finally:
        provider.close()
