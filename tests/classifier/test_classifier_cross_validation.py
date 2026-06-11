import numpy as np

from llm_fingerprinter.classifier import EnsembleClassifier


def test_cross_validate_ignores_configured_classes_without_samples(monkeypatch):
    classifier = EnsembleClassifier(
        model_families={"alpha": 0, "missing": 1, "beta": 2},
        augment_data=False,
    )
    X = np.array(
        [[0.0], [0.1], [0.2], [1.0], [1.1], [1.2]],
        dtype=np.float32,
    )
    y = np.array([0, 0, 0, 2, 2, 2])

    monkeypatch.setattr(EnsembleClassifier, "train", lambda self, X, y: True)
    monkeypatch.setattr(
        EnsembleClassifier,
        "predict_with_confidence",
        lambda self, fingerprint: (
            "alpha" if fingerprint[0] < 0.5 else "beta",
            1.0,
            {},
            {},
        ),
    )

    results = classifier.cross_validate(X, y, n_folds=5)

    assert results is not None
    assert results["n_folds"] == 3
    assert results["mean_accuracy"] == 1.0
