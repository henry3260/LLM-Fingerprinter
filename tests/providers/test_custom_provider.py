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


class _FakeCustomClient:
    def __init__(self):
        self.called_with = None

    def generate(self, **kwargs):
        self.called_with = kwargs
        return " custom response "

    def list_models(self):
        return ["m1", "m2"]

    def _check_connectivity(self):
        return True

    def close(self):
        return None


def test_custom_provider_generate_normalizes_response_and_maps_args():
    provider = CustomProvider.__new__(CustomProvider)
    provider.name = "custom"
    provider._client = _FakeCustomClient()

    resp = provider.generate(_Request())

    assert provider._client.called_with == {
        "prompt": "last user",
        "model": "custom-model",
        "temperature": 0.6,
        "max_tokens": 128,
        "system": "sys prompt",
    }
    assert resp.provider == "custom"
    assert resp.model == "custom-model"
    assert resp.text == " custom response "
    assert resp.usage.total_tokens == 0


def test_custom_provider_health_and_list_models():
    provider = CustomProvider.__new__(CustomProvider)
    provider.name = "custom"
    provider._client = _FakeCustomClient()

    assert provider.health_check() is True
    assert provider.list_models() == ["m1", "m2"]
