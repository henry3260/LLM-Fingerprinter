"""DeepSeek provider implementation via OpenAI-compatible API."""

from llm_fingerprinter.providers.openai import OpenAIProvider


class DeepSeekProvider(OpenAIProvider):
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com", timeout: int = 60, max_retries: int = 3):
        super().__init__(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries)
        self.name = "deepseek"
