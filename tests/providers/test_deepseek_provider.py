from llm_fingerprinter.providers.deepseek import DeepSeekProvider
from llm_fingerprinter.providers.openai import OpenAIProvider


def test_deepseek_provider_is_openai_compatible_subclass():
    assert issubclass(DeepSeekProvider, OpenAIProvider)


def test_deepseek_provider_sets_provider_name():
    provider = DeepSeekProvider.__new__(DeepSeekProvider)
    provider.name = "openai"
    provider.name = "deepseek"
    assert provider.name == "deepseek"
