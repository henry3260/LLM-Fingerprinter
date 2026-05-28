from llm_fingerprinter.contracts.llm import LLMResponse
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


class _Msg:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content


class _Request:
    def __init__(self):
        self.provider = "ollama-cloud"
        self.model = "llama3.2"
        self.messages = [_Msg("user", "hello")]
        self.temperature = 0.1
        self.max_tokens = 8


class _FakeResponse:
    status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return {"response": "ok", "done": True, "prompt_eval_count": 1, "eval_count": 2}


class _FakeSession:
    def __init__(self):
        self.headers = {}

    def post(self, url, json, timeout):
        return _FakeResponse()

    def get(self, url, timeout):
        return _FakeResponse()

    def close(self):
        return None


def test_cloud_provider_accepts_ollama_cloud_alias_request_provider():
    provider = CloudProvider(api_key="secret")
    try:
        provider._session = _FakeSession()
        resp = provider.generate(_Request())
        assert isinstance(resp, LLMResponse)
        assert resp.provider == "cloud"
        assert resp.text == "ok"
    finally:
        provider.close()
