from llm_fingerprinter.contracts.llm import LLMResponse
from llm_fingerprinter.providers.custom import CustomProvider


class _Msg:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content


class _Request:
    def __init__(self):
        self.provider = "custom"
        self.model = "custom-model"
        self.messages = [
            _Msg("system", "sys prompt"),
            _Msg("user", "first"),
            _Msg("assistant", "ignored"),
            _Msg("user", "last user"),
        ]
        self.temperature = 0.6
        self.max_tokens = 128


class _FakeResponse:
    def __init__(self, status_code=200, body=None, text=None):
        self.status_code = status_code
        self._body = body if body is not None else {"response": " custom response "}
        self.text = text if text is not None else '{"response": " custom response "}'

    def json(self):
        return self._body


class _FakeSession:
    def __init__(self):
        self.posts = []
        self.closed = False

    def post(self, url, json, timeout):
        self.posts.append({"url": url, "json": json, "timeout": timeout})
        return _FakeResponse()

    def head(self, url, timeout):
        return _FakeResponse(status_code=200, body={}, text="")

    def close(self):
        self.closed = True


def test_custom_provider_generate_normalizes_response_and_maps_args():
    provider = CustomProvider.__new__(CustomProvider)
    provider.name = "custom"
    provider.url = "https://example.test/generate"
    provider.payload_template = (
        '{"model": "$MODEL$", "prompt": "$PROMPT$", "system": "$SYSTEM$", '
        '"temperature": $TEMPERATURE$, "max_tokens": $MAX_TOKENS$}'
    )
    provider.default_model = None
    provider.default_temperature = 0.7
    provider.default_max_tokens = 512
    provider.default_system = ""
    provider.response_path = None
    provider.timeout = 120
    provider.session = _FakeSession()

    resp = provider.generate(_Request())

    assert isinstance(resp, LLMResponse)
    assert provider.session.posts[-1] == {
        "url": "https://example.test/generate",
        "timeout": 120,
        "json": {
            "model": "custom-model",
            "prompt": "last user",
            "system": "sys prompt",
            "temperature": 0.6,
            "max_tokens": 128,
        },
    }
    assert resp.provider == "custom"
    assert resp.model == "custom-model"
    assert resp.text == "custom response"
    assert resp.usage.total_tokens == 0


def test_custom_provider_health_and_list_models():
    provider = CustomProvider.__new__(CustomProvider)
    provider.name = "custom"
    provider.url = "https://example.test/generate"
    provider.session = _FakeSession()

    assert provider.health_check() is True
    assert provider.list_models() == []
    provider.close()
    assert provider.session.closed is True


def test_custom_provider_build_payload_escapes_prompt_content():
    provider = CustomProvider.__new__(CustomProvider)
    provider.payload_template = '{"prompt": "$PROMPT$"}'
    provider.default_system = ""
    provider.default_model = None
    provider.default_temperature = 0.7
    provider.default_max_tokens = 512

    assert provider._build_payload('hello "quoted"') == {"prompt": 'hello "quoted"'}
