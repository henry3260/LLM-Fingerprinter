"""LLM Fingerprinter - Black-box LLM model identification system.
"""

__version__ = "0.3.0"
__author__ = "litemars"

__all__ = [
    "LLMFingerprinter",
    "EnsembleClassifier",
    "create_classifier",
    "FeatureExtractor",
    "PromptSuite",
    "FingerprintStore",
    "__version__",
]


def __getattr__(name):
    """Lazily import public classes so lightweight CLI commands stay fast."""
    if name == "LLMFingerprinter":
        from llm_fingerprinter.fingerprinter import LLMFingerprinter
        return LLMFingerprinter
    if name in {"EnsembleClassifier", "create_classifier"}:
        from llm_fingerprinter.classifier import EnsembleClassifier, create_classifier
        return {
            "EnsembleClassifier": EnsembleClassifier,
            "create_classifier": create_classifier,
        }[name]
    if name == "FeatureExtractor":
        from llm_fingerprinter.feature_extractor import FeatureExtractor
        return FeatureExtractor
    if name == "PromptSuite":
        from llm_fingerprinter.promptgen import PromptSuite
        return PromptSuite
    if name == "FingerprintStore":
        from llm_fingerprinter.fingerprint_store import FingerprintStore
        return FingerprintStore
    raise AttributeError(f"module 'llm_fingerprinter' has no attribute {name!r}")
