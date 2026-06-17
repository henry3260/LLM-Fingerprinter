import numpy as np
import pytest

from llm_fingerprinter.contracts.feature import FeatureVector
from llm_fingerprinter.contracts.llm import LLMResponse
import llm_fingerprinter.feature_extractor as feature_extractor_module


class _FakeEmbeddingModel:
    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            return np.array([1.0, 2.0, 3.0], dtype=np.float32)
        return np.array(
            [
                [idx + 1.0, idx + 2.0, idx + 3.0]
                for idx, _ in enumerate(texts)
            ],
            dtype=np.float32,
        )


@pytest.fixture
def extractor(monkeypatch):
    monkeypatch.setattr(
        feature_extractor_module,
        "SentenceTransformer",
        lambda model_name: _FakeEmbeddingModel(),
    )
    return feature_extractor_module.FeatureExtractor(model_name="fake-embedding")


def test_extract_vector_returns_named_feature_vector(extractor):
    vector = extractor.extract_vector(
        "List three facts.",
        "First, this is a concise answer. Second, it follows the requested format.",
    )

    assert isinstance(vector, FeatureVector)
    assert vector.namespace == "llm_response"
    assert len(vector.items) == extractor.get_feature_dim()
    assert vector.metadata["schema_version"] == "feature_extractor.v2"
    assert vector.metadata["embedding_model"] == "fake-embedding"
    assert vector.metadata["embedding_dim"] == 3
    assert vector.metadata["empty_response"] is False

    names = [item.name for item in vector.items]
    assert names[:3] == ["embedding_0", "embedding_1", "embedding_2"]
    assert names[3:15] == list(extractor.LINGUISTIC_FEATURE_NAMES)
    assert names[15:] == list(extractor.BEHAVIORAL_FEATURE_NAMES)


def test_extract_keeps_legacy_numpy_output_compatible(extractor):
    response = LLMResponse(
        provider="test-provider",
        model="test-model",
        text="This is a short response.",
    )

    vector = extractor.extract_vector("Explain briefly.", response)
    legacy = extractor.extract("Explain briefly.", response)

    np.testing.assert_allclose(legacy, extractor.feature_vector_to_array(vector))
    assert legacy.dtype == np.float32
    assert legacy.shape == (extractor.get_feature_dim(),)


def test_extract_batch_vectors_preserve_empty_response_as_zero_vector(extractor):
    pairs = [
        ("Prompt A", ""),
        ("Prompt B", "A valid response."),
    ]

    vectors = extractor.extract_batch_vectors(pairs)
    legacy = extractor.extract_batch(pairs)

    assert len(vectors) == 2
    assert len(legacy) == 2
    assert vectors[0].metadata["empty_response"] is True
    assert vectors[1].metadata["empty_response"] is False

    np.testing.assert_allclose(
        extractor.feature_vector_to_array(vectors[0]),
        np.zeros(extractor.get_feature_dim(), dtype=np.float32),
    )
    np.testing.assert_allclose(
        legacy[0],
        extractor.feature_vector_to_array(vectors[0]),
    )
    np.testing.assert_allclose(
        legacy[1],
        extractor.feature_vector_to_array(vectors[1]),
    )


def test_extract_vector_uses_cjk_aware_linguistic_features(extractor):
    vector = extractor.extract_vector(
        "請用兩句話簡短說明。",
        "這是第一句。這是第二句！",
    )
    values = {feature.name: float(feature.value) for feature in vector.items}

    assert vector.metadata["language_profile"] == "cjk"
    assert values["total_words"] > 2
    assert values["sentence_count"] == 2
    assert values["punctuation_ratio"] > 0


def test_extract_vector_detects_chinese_behavioral_signals(extractor):
    vector = extractor.extract_vector(
        "請條列說明理由。",
        "一、我無法協助這項要求。\n二、因為這可能造成傷害，因此我會拒絕。",
    )
    values = {feature.name: float(feature.value) for feature in vector.items}

    assert values["refusal_score"] > 0
    assert values["reasoning_presence_score"] > 0
    assert values["instruction_compliance_score"] == 1.0


def test_feature_extractor_defaults_to_configured_embedding(monkeypatch):
    seen = {}

    class _ConfiguredEmbeddingModel(_FakeEmbeddingModel):
        def __init__(self, model_name):
            seen["model_name"] = model_name

    monkeypatch.setattr(feature_extractor_module.config, "EMBEDDING_MODEL", "configured-model")
    monkeypatch.setattr(
        feature_extractor_module,
        "SentenceTransformer",
        lambda model_name: _ConfiguredEmbeddingModel(model_name),
    )

    feature_extractor_module.FeatureExtractor()

    assert seen["model_name"] == "configured-model"
