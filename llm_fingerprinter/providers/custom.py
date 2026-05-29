"""Custom provider backed by a template-driven HTTP request file."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from llm_fingerprinter.contracts.llm import LLMRequest, LLMResponse, TokenUsage
from llm_fingerprinter.providers.base import BaseProvider, ProviderCapabilities, validate_request

logger = logging.getLogger(__name__)


class CustomProviderError(Exception):
    """Base exception for custom provider errors."""


class CustomConnectionError(CustomProviderError):
    """Raised when connection to API fails."""


class CustomGenerationError(CustomProviderError):
    """Raised when generation fails."""


class CustomAuthError(CustomProviderError):
    """Raised when authentication fails."""


class CustomProvider(BaseProvider):
    """Provider adapter for arbitrary HTTP APIs through a request template file."""

    def __init__(
        self,
        request_file: str,
        api_key: str | None = None,
        timeout: int = 120,
        auth_header: str = "Authorization",
        auth_prefix: str = "Bearer",
        default_model: str | None = None,
        default_temperature: float = 0.7,
        default_max_tokens: int = 512,
        default_system: str | None = None,
        response_path: list | None = None,
    ):
        super().__init__(
            name="custom",
            capabilities=ProviderCapabilities(
                supports_system_role=True,
                supports_tools=False,
                supports_json_mode=False,
            ),
        )
        self.api_key = api_key
        self.auth_header = auth_header
        self.auth_prefix = auth_prefix
        self.timeout = timeout
        self.default_model = default_model
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.default_system = default_system or ""
        self.response_path = response_path
        self.url: str | None = None
        self.payload_template: str | None = None

        self._parse_request_file(request_file)

        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=10,
            max_retries=0,
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.session.headers.update({"Content-Type": "application/json"})

        if api_key:
            if auth_prefix:
                self.session.headers[auth_header] = f"{auth_prefix} {api_key}"
            else:
                self.session.headers[auth_header] = api_key

        logger.info("Initialized CustomProvider for %s", self.url)

    def _parse_request_file(self, request_file: str) -> None:
        path = Path(request_file)
        if not path.exists():
            raise CustomProviderError(f"Request file not found: {request_file}")

        content = path.read_text().strip()
        lines = content.split("\n")
        if not lines:
            raise CustomProviderError(f"Request file is empty: {request_file}")

        self.url = lines[0].strip()
        if not self.url.startswith(("http://", "https://")):
            raise CustomProviderError(f"Invalid URL in request file: {self.url}")

        if len(lines) <= 1:
            raise CustomProviderError("Request file must contain JSON payload after URL")

        json_content = "\n".join(lines[1:]).strip()
        test_json = json_content
        test_json = test_json.replace("$PROMPT$", "test")
        test_json = test_json.replace("$SYSTEM$", "test")
        test_json = test_json.replace("$MODEL$", "test")
        test_json = test_json.replace("$TEMPERATURE$", "0.7")
        test_json = test_json.replace("$MAX_TOKENS$", "512")

        try:
            json.loads(test_json)
        except json.JSONDecodeError as exc:
            raise CustomProviderError(f"Invalid JSON in request file: {exc}") from exc

        if "$PROMPT$" not in json_content:
            raise CustomProviderError("Request file must contain $PROMPT$ placeholder")

        self.payload_template = json_content
        logger.debug("Parsed custom request file: URL=%s", self.url)

    def _build_payload(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system: str | None = None,
    ) -> dict:
        if not self.payload_template:
            raise CustomProviderError("No payload template configured")

        payload_str = self.payload_template
        escaped_prompt = json.dumps(prompt)[1:-1]
        escaped_system = json.dumps(system or self.default_system)[1:-1]

        payload_str = payload_str.replace("$PROMPT$", escaped_prompt)
        payload_str = payload_str.replace("$SYSTEM$", escaped_system)
        payload_str = payload_str.replace("$MODEL$", model or self.default_model or "")
        payload_str = payload_str.replace(
            "$TEMPERATURE$",
            str(temperature if temperature is not None else self.default_temperature),
        )
        payload_str = payload_str.replace(
            "$MAX_TOKENS$",
            str(max_tokens if max_tokens is not None else self.default_max_tokens),
        )

        try:
            return json.loads(payload_str)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse custom payload: %s", payload_str[:200])
            raise CustomGenerationError(f"Invalid payload after substitution: {exc}") from exc

    def _extract_response_text(self, data) -> str:
        if isinstance(data, dict):
            if data.get("done_reason", "") == "load":
                return ""
            if "response" in data and data["response"] == "" and data.get("done"):
                return ""
            if "error" in data:
                logger.error("API returned error: %s", data["error"])
                return ""

        if self.response_path:
            result = self._get_path(data, self.response_path)
            if isinstance(result, str) and result.strip():
                return result.strip()

        fallback_paths = [
            ["choices", 0, "message", "content"],
            ["choices", 0, "text"],
            ["choices", 0, "delta", "content"],
            ["response"],
            ["content"],
            ["text"],
            ["output"],
            ["message"],
            ["result"],
            ["answer"],
            ["completion"],
            ["generated_text"],
            ["data", "content"],
            ["data", "text"],
            ["message", "content"],
            ["content", 0, "text"],
        ]

        for path in fallback_paths:
            result = self._get_path(data, path)
            if isinstance(result, str) and result.strip():
                return result.strip()

        return ""

    def _get_path(self, data, path: list):
        result = data
        for key in path:
            if result is None:
                return None
            if isinstance(key, int):
                if isinstance(result, (list, tuple)) and len(result) > key:
                    result = result[key]
                else:
                    return None
            elif isinstance(result, dict):
                result = result.get(key)
            else:
                return None
        return result

    def _parse_streaming_response(self, response_text: str) -> str:
        full_text = []
        for line in response_text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            for obj_str in self._split_json_objects(line):
                try:
                    text = self._extract_response_text(json.loads(obj_str))
                except json.JSONDecodeError:
                    continue
                if text:
                    full_text.append(text)
        return "".join(full_text)

    def _split_json_objects(self, text: str) -> list[str]:
        objects = []
        depth = 0
        start = 0
        in_string = False
        escape_next = False

        for index, char in enumerate(text):
            if escape_next:
                escape_next = False
                continue
            if char == "\\" and in_string:
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                if depth == 0:
                    start = index
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    objects.append(text[start:index + 1])
        return objects

    def health_check(self) -> bool:
        if not self.url:
            logger.error("No custom URL configured")
            return False

        try:
            base_url = "/".join(self.url.split("/")[:3])
            response = self.session.head(base_url, timeout=10)
            return response.status_code < 500
        except requests.exceptions.ConnectionError as exc:
            logger.error("Connection error: %s", exc)
            return False
        except requests.exceptions.Timeout:
            logger.error("Timeout connecting to %s", self.url)
            return False
        except Exception as exc:
            logger.error("Error checking connectivity: %s", exc)
            return False

    def generate(self, request: LLMRequest) -> LLMResponse:
        validate_request(self.name, request)

        messages = getattr(request, "messages", [])
        system_prompt = next((m.content for m in messages if m.role == "system"), None)
        prompt = next((m.content for m in reversed(messages) if m.role == "user"), "")

        text = self._generate_text(
            prompt=prompt,
            model=getattr(request, "model", None),
            temperature=getattr(request, "temperature", None),
            max_tokens=getattr(request, "max_tokens", None),
            system=system_prompt,
        )

        return LLMResponse(
            provider=self.name,
            model=getattr(request, "model", ""),
            text=text or "",
            finish_reason=None,
            usage=TokenUsage(),
            raw={},
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.Timeout, requests.ConnectionError)),
    )
    def _generate_text(
        self,
        prompt: str = "",
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system: str | None = None,
    ) -> str:
        logger.debug("Generating with custom model %s", model)
        if not self.url:
            raise CustomProviderError("No URL configured. Provide request_file.")

        payload = self._build_payload(prompt, model, temperature, max_tokens, system)
        try:
            start = time.time()
            response = self.session.post(self.url, json=payload, timeout=self.timeout)
            elapsed = time.time() - start

            if response.status_code == 200:
                response_text = response.text
                try:
                    result = response.json()
                    text = self._extract_response_text(result)
                    if text:
                        logger.debug("Generated %d chars in %.2fs", len(text), elapsed)
                        return text
                except ValueError:
                    pass

                text = self._parse_streaming_response(response_text)
                if text:
                    logger.debug("Generated %d chars (streaming) in %.2fs", len(text), elapsed)
                    return text

                if response_text.strip() and not response_text.strip().startswith("{"):
                    return response_text.strip()

                logger.error("Could not extract text from custom response")
                logger.debug("Response: %s", response_text[:500])
                return ""

            if response.status_code == 401:
                raise CustomAuthError("Invalid API key")
            if response.status_code == 403:
                raise CustomAuthError("Access forbidden - check API key permissions")
            if response.status_code == 404:
                raise CustomGenerationError(f"Endpoint not found: {self.url}")
            if response.status_code == 429:
                raise CustomGenerationError("Rate limit exceeded - please wait and retry")

            error_msg = self._extract_error_message(response)
            raise CustomGenerationError(f"API error {response.status_code}: {error_msg}")

        except requests.Timeout:
            logger.warning("Timeout after %ss", self.timeout)
            raise
        except requests.ConnectionError as exc:
            logger.error("Connection error: %s", exc)
            raise CustomConnectionError(f"Cannot connect to {self.url}") from exc
        except CustomProviderError:
            raise
        except Exception as exc:
            logger.error("Unexpected custom provider error: %s", exc)
            raise CustomGenerationError(f"Generation failed: {exc}") from exc

    def _extract_error_message(self, response: requests.Response) -> str:
        try:
            error_json = response.json()
            if "error" in error_json:
                err = error_json["error"]
                if isinstance(err, dict):
                    return err.get("message", str(err))
                return str(err)
            if "message" in error_json:
                return error_json["message"]
            if "detail" in error_json:
                return error_json["detail"]
            return str(error_json)[:200]
        except (ValueError, KeyError):
            return response.text[:200] if response.text else "Unknown error"

    def list_models(self) -> list[str]:
        logger.warning("list_models not supported for template-based custom provider")
        return []

    def close(self) -> None:
        self.session.close()
        logger.debug("Closed CustomProvider session")

    def __repr__(self) -> str:
        if self.url:
            return f"CustomProvider(url='{self.url}')"
        return "CustomProvider(not configured)"
