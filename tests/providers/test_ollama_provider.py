from types import SimpleNamespace

from llm_fingerprinter.providers.ollama_client import OllamaProvider


class _Msg:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content


class _Request:
    def __init__(self):
        self.provider = "ollama"
        self.model = "llama3.2"
        self.messages = [_Msg("system", "sys"), _Msg("user", "hello"), _Msg("assistant", "ack")]
        self.temperature = 0.3
        self.max_tokens = 55


class _FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("http error")

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self):
        self.headers = {}
        self.last_post = None

    def post(self, url, json, timeout):
        self.last_post = (url, json, timeout)
        return _FakeResponse(
            payload={
                "response": "  answer  ",
                "done": True,
                "prompt_eval_count": 9,
                "eval_count": 11,
            }
        )

    def get(self, url, timeout):
        if url.endswith("/api/tags"):
            return _FakeResponse(payload={"models": [{"name": "llama3.2"}, {"name": "qwen3"}]})
        return _FakeResponse(status_code=500)

    def close(self):
        return None


def test_ollama_provider_generate_and_list_models():
    provider = OllamaProvider.__new__(OllamaProvider)
    provider.name = "ollama"
    provider._base_url = "http://localhost:11434"
    provider._timeout = 60
    provider._session = _FakeSession()

    resp = provider.generate(_Request())
    assert resp.provider == "ollama"
    assert resp.model == "llama3.2"
    assert resp.text == "answer"
    assert resp.finish_reason == "stop"
    assert resp.usage.input_tokens == 9
    assert resp.usage.output_tokens == 11
    assert resp.usage.total_tokens == 20

    post_url, payload, timeout = provider._session.last_post
    assert post_url.endswith("/api/generate")
    assert payload["system"] == "sys"
    assert payload["prompt"] == "hello\nack"
    assert timeout == 60

    assert provider.health_check() is True
    assert provider.list_models() == ["llama3.2", "qwen3"]
