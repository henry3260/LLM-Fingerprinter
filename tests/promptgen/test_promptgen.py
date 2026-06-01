import numpy as np

from llm_fingerprinter import config
from llm_fingerprinter.contracts.feature import Feature, FeatureVector
from llm_fingerprinter.contracts.llm import LLMResponse, TokenUsage
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
        return LLMResponse(
            provider="test-provider",
            model=kwargs["model"],
            text=f"response:{kwargs['prompt']}:{system}",
            finish_reason="stop",
            usage=TokenUsage(
                input_tokens=len(kwargs["prompt"]),
                output_tokens=3,
                total_tokens=len(kwargs["prompt"]) + 3,
            ),
        )


class _Extractor:
    embedding_dim = 1
    LINGUISTIC_DIM = 1
    BEHAVIORAL_DIM = 1

    def __init__(self):
        self.pairs = []

    def get_feature_dim(self):
        return 3

    @staticmethod
    def feature_vector_to_array(feature_vector):
        return np.array(
            [float(feature.value) for feature in feature_vector.items],
            dtype=np.float32,
        )

    def extract_batch_vectors(self, prompt_response_pairs):
        self.pairs.extend(prompt_response_pairs)
        return [
            FeatureVector(
                items=[
                    Feature(name="embedding_0", value=idx + 1),
                    Feature(name="total_chars", value=idx + 2),
                    Feature(name="refusal_score", value=idx + 3),
                ],
                namespace="test",
                metadata={"pair_index": idx},
            )
            for idx, _ in enumerate(prompt_response_pairs)
        ]

    def extract_batch(self, prompt_response_pairs):
        raise AssertionError("fingerprint_model should use extract_batch_vectors")


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
    assert [prompt for prompt, _ in extractor.pairs] == [
        "alpha prompt",
        "beta prompt",
    ]
    assert [response.text for _, response in extractor.pairs] == [
        "response:alpha prompt:sys-a",
        "response:beta prompt:",
    ]
    assert fingerprint["metadata"]["queries_executed"] == 2
    assert fingerprint["metadata"]["queries_total"] == 2
    assert fingerprint["responses_sample"] == [
        {
            "prompt": "alpha prompt",
            "response": "response:alpha prompt:sys-a",
            "response_metadata": {
                "provider": "test-provider",
                "model": "model-a",
                "finish_reason": "stop",
                "usage": {
                    "input_tokens": 12,
                    "output_tokens": 3,
                    "total_tokens": 15,
                },
            },
            "layer": "alpha",
            "category": "cat-a",
            "feature_metadata": {"pair_index": 0},
        },
        {
            "prompt": "beta prompt",
            "response": "response:beta prompt:",
            "response_metadata": {
                "provider": "test-provider",
                "model": "model-a",
                "finish_reason": "stop",
                "usage": {
                    "input_tokens": 11,
                    "output_tokens": 3,
                    "total_tokens": 14,
                },
            },
            "layer": "beta",
            "category": "cat-b",
            "feature_metadata": {"pair_index": 0},
        },
    ]
