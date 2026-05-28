"""Backward-compatible import path for PromptSuite."""

from llm_fingerprinter.promptgen import (
    PromptItem,
    PromptPackage,
    PromptSuite,
    build_prompt_package,
)

__all__ = ["PromptItem", "PromptPackage", "PromptSuite", "build_prompt_package"]
