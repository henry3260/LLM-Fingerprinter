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
    assert vector.metadata["schema_version"] == "feature_extractor.v1"
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
