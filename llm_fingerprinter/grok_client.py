import logging
import time
from typing import List

from llm_fingerprinter.base_client import BaseClient, ClientError

logger = logging.getLogger(__name__)


class GrokError(ClientError):
    """Base exception for Grok client errors."""
    pass


class GrokConnectionError(GrokError):
    """Raised when connection to Grok API fails."""
    pass


class GrokGenerationError(GrokError):
    """Raised when generation fails."""
    pass


class GrokAuthError(GrokError):
    """Raised when authentication fails."""
    pass


class GrokClient(BaseClient):
    def __init__(self,
                 api_key: str,
                 endpoint: str = "https://api.x.ai/v1",
                 timeout: int = 60,
                 max_retries: int = 3):
        super().__init__(timeout=timeout, max_retries=max_retries)

        try:
            from openai import OpenAI
            import openai as openai_module
        except ImportError as e:
            raise ImportError(
                "openai package is required. Install with: pip install openai"
            ) from e

        self._openai_module = openai_module
        self.api_key = api_key
        self.endpoint = endpoint.rstrip("/")

        self.client = OpenAI(
            api_key=api_key,
            base_url=self.endpoint,
            timeout=timeout,
            max_retries=max_retries,
        )

        logger.info(f"Initialized GrokClient for {self.endpoint}")

    def _perform_health_check(self) -> bool:
        try:
            list(self.client.models.list())
            return True
        except self._openai_module.AuthenticationError:
            logger.warning("API reachable but authentication failed - check API key")
            return False
        except self._openai_module.PermissionDeniedError:
            logger.warning("API reachable but access forbidden - check permissions")
            return False
        except self._openai_module.APIConnectionError as e:
            logger.error(f"Connection error to API at {self.endpoint}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error checking API connectivity: {e}")
            return False

    def generate(self, model, prompt, system=None, temperature=0.7, max_tokens=512):
        start = time.time()

        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})

            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            elapsed = time.time() - start

            if not response.choices:
                raise GrokGenerationError("No choices in response")

            text = response.choices[0].message.content
            text = text.strip() if text else ""

            usage = response.usage
            completion_tokens = usage.completion_tokens if usage else 0

            logger.debug(f"Generated {len(text)} chars, {completion_tokens} tokens in {elapsed:.2f}s")
            return text

        except self._openai_module.AuthenticationError:
            raise GrokAuthError("Invalid API key")
        except self._openai_module.PermissionDeniedError:
            raise GrokAuthError("Access forbidden - check API key permissions")
        except self._openai_module.NotFoundError:
            raise GrokGenerationError(f"Model '{model}' not found")
        except self._openai_module.RateLimitError:
            raise GrokGenerationError("Rate limit exceeded - please wait and retry")
        except self._openai_module.APIConnectionError as e:
            logger.error(f"Connection error to API: {e}")
            raise GrokConnectionError(f"Cannot connect to API at {self.endpoint}")
        except self._openai_module.APITimeoutError:
            logger.warning(f"Timeout querying {model} after {self.timeout}s")
            raise GrokConnectionError(f"Request timeout after {self.timeout}s")
        except GrokError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error generating from {model}: {e}")
            raise GrokGenerationError(f"Generation failed: {e}")

    def list_models(self) -> List[str]:
        try:
            models = [model.id for model in self.client.models.list()]
            models.sort()
            logger.info(f"Found {len(models)} models on Grok API")
            return models
        except self._openai_module.AuthenticationError:
            logger.error("Authentication failed - check API key")
            return []
        except self._openai_module.APIConnectionError:
            logger.error(f"Cannot connect to API at {self.endpoint}")
            return []
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []

    def model_info(self, model):
        try:
            model_obj = self.client.models.retrieve(model)
            return {
                "id": model_obj.id,
                "object": model_obj.object,
                "created": getattr(model_obj, 'created', None),
                "owned_by": getattr(model_obj, 'owned_by', 'xai'),
            }
        except self._openai_module.NotFoundError:
            logger.warning(f"Model '{model}' not found")
            return None
        except Exception as e:
            logger.error(f"Error getting model info for {model}: {e}")
            return None

    def close(self):
        self.client.close()
        logger.debug("Closed GrokClient")
