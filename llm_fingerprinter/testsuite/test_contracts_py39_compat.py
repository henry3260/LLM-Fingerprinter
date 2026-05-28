from llm_fingerprinter.contracts import (
    ClassificationResult,
    Feature,
    FingerprintInput,
    LLMRequest,
    Message,
    MultiClassResult,
    ProviderScore,
    TokenUsage,
)


def test_contracts_public_imports_work():
    msg = Message(role="user", content="hello")
    req = LLMRequest(provider="grok", model="grok-1", messages=[msg])
    assert req.messages[0].content == "hello"

    usage = TokenUsage(input_tokens=1, output_tokens=2, total_tokens=3)
    assert usage.total_tokens == 3

    feat = Feature(name="len", value=12)
    assert feat.value == 12

    fp = FingerprintInput(target_text="x", candidate_providers=["grok"], candidate_models=["grok-1"])
    assert fp.n_trials == 1

    score = ProviderScore(provider="grok", model="grok-1", score=0.9)
    assert score.score == 0.9

    cls = ClassificationResult(label="grok", score=0.9)
    out = MultiClassResult(top_label="grok", top_score=0.9, ranking=[cls])
    assert out.ranking[0].label == "grok"
