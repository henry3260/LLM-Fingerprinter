import numpy as np

from llm_fingerprinter.contracts.classify import (
    ClassificationResult,
    MultiClassResult,
)
from llm_fingerprinter.fingerprinter import LLMFingerprinter


class _FakeExtractor:
    def get_feature_dim(self):
        return 2


class _FakeSuite:
    def __len__(self):
        return 0


class _ContractClassifier:
    is_trained = True

    def __init__(self, result):
        self.result = result
        self.seen_fingerprint = None

    def classify_result(self, fingerprint):
        self.seen_fingerprint = fingerprint
        return self.result

    def predict_with_confidence(self, fingerprint):
        raise AssertionError("identify() should use classify_result()")


def _fingerprinter_with_result(contract_result):
    classifier = _ContractClassifier(contract_result)
    fingerprinter = LLMFingerprinter(
        endpoint="test",
        ollama_client=None,
        prompt_suite=_FakeSuite(),
        feature_extractor=_FakeExtractor(),
        classifier=classifier,
    )
    fingerprint = {
        "vector": np.array([1.0, 2.0], dtype=np.float32),
        "metadata": {
            "early_stopped": False,
            "layers_completed": ["discriminative", "behavioral", "stylistic"],
            "queries_executed": 31,
            "queries_total": 31,
        },
    }
    fingerprinter.fingerprint_model = lambda *_, **__: fingerprint
    return fingerprinter, classifier, fingerprint


def test_identify_maps_contract_result_to_legacy_report_dict():
    contract_result = MultiClassResult(
        top_label="gpt",
        top_score=0.87654,
        ranking=[
            ClassificationResult(label="gpt", score=0.87654),
            ClassificationResult(label="qwen", score=0.12345),
        ],
        metadata={
            "predicted_label": "gpt",
            "is_ood": False,
            "ood_info": {"agreement_ratio": 1.0},
            "probabilities": {"gpt": 0.87654, "qwen": 0.12345},
        },
    )
    fingerprinter, classifier, fingerprint = _fingerprinter_with_result(contract_result)

    result = fingerprinter.identify("model-under-test")

    assert classifier.seen_fingerprint is fingerprint["vector"]
    assert result["family"] == "gpt"
    assert result["predicted_family"] == "gpt"
    assert result["confidence"] == 0.8765
    assert result["all_probabilities"] == {"gpt": 0.8765, "qwen": 0.1235}
    assert result["ood_detected"] is False
    assert result["ood_details"] == {"agreement_ratio": 1.0}
    assert result["queries_executed"] == 31


def test_identify_preserves_ood_best_guess_from_contract_metadata():
    contract_result = MultiClassResult(
        top_label="unknown",
        top_score=0.2,
        ranking=[
            ClassificationResult(label="alpha", score=0.2),
            ClassificationResult(label="beta", score=0.1),
        ],
        metadata={
            "predicted_label": "alpha",
            "is_ood": True,
            "ood_info": {"agreement_ratio": 0.34},
            "probabilities": {"alpha": 0.2, "beta": 0.1},
        },
    )
    fingerprinter, _, _ = _fingerprinter_with_result(contract_result)

    result = fingerprinter.identify("model-under-test")

    assert result["family"] == "unknown"
    assert result["predicted_family"] == "alpha"
    assert result["confidence"] == 0.2
    assert result["ood_detected"] is True
    assert result["ood_details"] == {"agreement_ratio": 0.34}
