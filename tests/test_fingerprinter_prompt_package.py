import numpy as np

from llm_fingerprinter import config
from llm_fingerprinter.fingerprinter import LLMFingerprinter
from llm_fingerprinter.promptgen import PromptItem, PromptPackage


class _PromptSuite:
    def __init__(self):
        self.get_prompts_called = False
        self.package = PromptPackage(
            prompts=(
                PromptItem(text="alpha prompt", layer="alpha", category="cat-a", system="sys-a"),
                PromptItem(text="beta prompt", layer="beta", category="cat-b"),
            )
        )

    def get_prompt_package(self):
        return self.package

    def get_prompts(self, layer=None):
        self.get_prompts_called = True
        raise AssertionError("fingerprint_model should use PromptPackage")

    def __len__(self):
        return len(self.package.prompts)


class _Client:
    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        system = kwargs.get("system") or ""
        return f"response:{kwargs['prompt']}:{system}"


class _Extractor:
    embedding_dim = 1
    LINGUISTIC_DIM = 1
    BEHAVIORAL_DIM = 1

    def __init__(self):
        self.pairs = []

    def get_feature_dim(self):
        return 3

    def extract_batch(self, prompt_response_pairs):
        self.pairs.extend(prompt_response_pairs)
        return [
            np.array([idx + 1, idx + 2, idx + 3], dtype=np.float32)
            for idx, _ in enumerate(prompt_response_pairs)
        ]


def test_fingerprint_model_consumes_prompt_package(monkeypatch):
    monkeypatch.setattr(config, "LAYER_ORDER", ["alpha", "beta"])
    suite = _PromptSuite()
    client = _Client()
    extractor = _Extractor()
    fingerprinter = LLMFingerprinter("test-endpoint", client, suite, extractor, classifier=None)

    fingerprint = fingerprinter.fingerprint_model("model-a", repeats=1, temperature=0.25)

    assert suite.get_prompts_called is False
    assert [call["prompt"] for call in client.calls] == ["alpha prompt", "beta prompt"]
    assert client.calls[0]["system"] == "sys-a"
    assert "system" not in client.calls[1]
    assert [call["temperature"] for call in client.calls] == [0.25, 0.25]
    assert extractor.pairs == [
        ("alpha prompt", "response:alpha prompt:sys-a"),
        ("beta prompt", "response:beta prompt:"),
    ]
    assert fingerprint["metadata"]["queries_executed"] == 2
    assert fingerprint["metadata"]["queries_total"] == 2
    assert fingerprint["responses_sample"] == [
        {
            "prompt": "alpha prompt",
            "response": "response:alpha prompt:sys-a",
            "layer": "alpha",
            "category": "cat-a",
        },
        {
            "prompt": "beta prompt",
            "response": "response:beta prompt:",
            "layer": "beta",
            "category": "cat-b",
        },
    ]
