import pytest

from llm_fingerprinter import config
from llm_fingerprinter.promptgen import PromptSuite


def test_prompt_suite_defaults_to_configured_chinese(monkeypatch):
    monkeypatch.setattr(config, "PROMPT_LANGUAGE", "zh")

    suite = PromptSuite()
    prompts = suite.get_prompts()

    assert suite.language == "zh"
    assert len(prompts) == 31
    assert prompts[0]["text"].startswith("你到底是哪個 AI 模型")
    assert len(suite.get_prompts(layer=PromptSuite.LAYER_DISCRIMINATIVE)) == 11
    assert len(suite.get_prompts(layer=PromptSuite.LAYER_BEHAVIORAL)) == 7
    assert len(suite.get_prompts(layer=PromptSuite.LAYER_STYLISTIC)) == 13


def test_prompt_suite_can_load_english_prompts():
    suite = PromptSuite(language="en")
    prompts = suite.get_prompts()

    assert suite.language == "en"
    assert len(prompts) == 31
    assert prompts[0]["text"].startswith("What AI model")


def test_prompt_suite_rejects_unknown_language():
    with pytest.raises(ValueError, match="Unsupported prompt language"):
        PromptSuite(language="fr")
