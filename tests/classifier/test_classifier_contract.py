import numpy as np

from llm_fingerprinter.classifier import EnsembleClassifier
from llm_fingerprinter.contracts.classify import MultiClassResult
from llm_fingerprinter.template_classifier import TemplateClassifier


def test_ensemble_classifier_exposes_contract_result(monkeypatch):
    classifier = EnsembleClassifier(model_families={"alpha": 0, "beta": 1})

    def fake_predict_with_confidence(fingerprint):
        return (
            "beta",
            0.75,
            {"alpha": 0.25, "beta": 0.75},
            {"is_ood": False, "agreement_ratio": 1.0},
        )

    monkeypatch.setattr(
        classifier,
        "predict_with_confidence",
        fake_predict_with_confidence,
    )

    result = classifier.classify_result(np.array([1.0, 2.0], dtype=np.float32))

    assert isinstance(result, MultiClassResult)
    assert result.top_label == "beta"
    assert result.top_score == 0.75
    assert [item.label for item in result.ranking] == ["beta", "alpha"]
    assert [item.score for item in result.ranking] == [0.75, 0.25]
    assert result.metadata["classifier_type"] == "ensemble"
    assert result.metadata["predicted_label"] == "beta"
    assert result.metadata["is_ood"] is False


def test_ensemble_classifier_marks_ood_top_label_unknown(monkeypatch):
    classifier = EnsembleClassifier(model_families={"alpha": 0, "beta": 1})

    def fake_predict_with_confidence(fingerprint):
        return (
            "alpha",
            0.2,
            {"alpha": 0.2, "beta": 0.1},
            {"is_ood": True, "agreement_ratio": 0.34},
        )

    monkeypatch.setattr(
        classifier,
        "predict_with_confidence",
        fake_predict_with_confidence,
    )

    result = classifier.classify_result(np.array([1.0, 2.0], dtype=np.float32))

    assert result.top_label == "unknown"
    assert result.top_score == 0.2
    assert result.metadata["predicted_label"] == "alpha"
    assert result.metadata["is_ood"] is True


def test_template_classifier_exposes_contract_result_and_keeps_legacy_dict():
    classifier = TemplateClassifier()
    classifier.templates = {
        "alpha": np.array([1.0, 0.0], dtype=np.float32),
        "beta": np.array([0.0, 1.0], dtype=np.float32),
    }
    classifier.is_built = True

    fingerprint = np.array([1.0, 0.0], dtype=np.float32)
    legacy = classifier.classify(fingerprint, top_k=2)
    result = classifier.classify_result(fingerprint, top_k=2)

    assert legacy["family"] == "alpha"
    assert isinstance(result, MultiClassResult)
    assert result.top_label == "alpha"
    assert result.top_score == legacy["confidence"]
    assert [item.label for item in result.ranking] == ["alpha", "beta"]
    assert result.ranking[0].raw["distance"] == 0.0
    assert result.metadata["classifier_type"] == "template"
    assert result.metadata["predicted_label"] == "alpha"
    assert result.metadata["is_ood"] is False
