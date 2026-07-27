"""
Multi-model LLM client supporting OpenAI, Anthropic Claude, and DeepSeek.

Configuration via environment variables or .env file:
    OPENAI_API_KEY      — for GPT-4o and other OpenAI models
    ANTHROPIC_API_KEY   — for Claude models
    DEEPSEEK_API_KEY    — for DeepSeek V4
"""

import os
import time
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=False)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Supported model registry
# ---------------------------------------------------------------------------

class LLMConfigurationError(ValueError):
    """Raised when an LLM provider is selected without usable credentials."""


@dataclass(frozen=True)
class ModelSpec:
    provider: str       # "openai", "anthropic", "deepseek"
    model_id: str       # actual API model name


# Map user-facing names to provider + model_id
MODEL_MAP: Dict[str, ModelSpec] = {
    # OpenAI models
    "gpt-4o":               ModelSpec("openai", "gpt-4o"),
    "gpt-4o-mini":          ModelSpec("openai", "gpt-4o-mini"),
    "gpt-4-turbo":          ModelSpec("openai", "gpt-4-turbo"),
    "gpt-4":                ModelSpec("openai", "gpt-4"),
    "openai":               ModelSpec("openai", "gpt-4o"),          # default alias
    "chatgpt":              ModelSpec("openai", "gpt-4o"),

    # Anthropic Claude models
    "claude-sonnet-5":      ModelSpec("anthropic", "claude-sonnet-5"),
    "claude-sonnet-4":      ModelSpec("anthropic", "claude-sonnet-4-20250514"),
    "claude-opus-5":        ModelSpec("anthropic", "claude-opus-5"),
    "claude-opus-4":        ModelSpec("anthropic", "claude-opus-4-20250514"),
    "claude-haiku-4.5":     ModelSpec("anthropic", "claude-haiku-4-5-20251001"),
    "claude":               ModelSpec("anthropic", "claude-sonnet-5"),
    "anthropic":            ModelSpec("anthropic", "claude-sonnet-5"),

    # DeepSeek models
    "deepseek-v4-flash":    ModelSpec("deepseek", "deepseek-v4-flash"),
    "deepseek-v4-pro":      ModelSpec("deepseek", "deepseek-v4-pro"),
    "deepseek":             ModelSpec("deepseek", "deepseek-v4-flash"),
    # Compatibility aliases now point to supported V4 endpoints.
    "deepseek-v3":          ModelSpec("deepseek", "deepseek-v4-flash"),
    "deepseek-r1":          ModelSpec("deepseek", "deepseek-v4-pro"),
}


def _is_usable_key(value: Optional[str]) -> bool:
    if not value or not value.strip():
        return False
    lowered = value.strip().lower()
    return not any(marker in lowered for marker in ("your-", "replace-me", "changeme"))


def get_default_model_name() -> str:
    """Return an explicit default, or select a provider with configured credentials."""
    configured = os.environ.get("AST_CBOR_MODEL", "").strip()
    if configured:
        return configured
    if _is_usable_key(os.environ.get("DEEPSEEK_API_KEY")):
        return "deepseek"
    if _is_usable_key(os.environ.get("OPENAI_API_KEY")):
        return "gpt-4o-mini"
    if _is_usable_key(os.environ.get("ANTHROPIC_API_KEY")):
        return "claude-sonnet-5"
    return "deepseek"


DEFAULT_MODEL = get_default_model_name()


def resolve_model(name: str) -> ModelSpec:
    """Map a user-facing model name to a ModelSpec."""
    key = name.lower().strip()
    if key in MODEL_MAP:
        return MODEL_MAP[key]
    # Allow pass-through for raw model IDs: "openai:gpt-4o", "anthropic:claude-sonnet-5"
    if ":" in key:
        provider, model_id = key.split(":", 1)
        if provider in ("openai", "anthropic", "deepseek"):
            return ModelSpec(provider, model_id)
    raise ValueError(
        f"Unknown model: '{name}'. "
        f"Use a known alias ({', '.join(sorted(MODEL_MAP.keys()))}) "
        f"or 'provider:model_id' format."
    )


# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------

class LLMClient:
    """Unified client for OpenAI, Anthropic, and DeepSeek APIs."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 3,
        timeout: int = 120,
    ):
        """
        Args:
            model: Model name alias (see MODEL_MAP) or 'provider:model_id'.
            api_key: Override API key. If None, reads from env var.
            base_url: Override API base URL.
            max_retries: Max retry attempts on timeout/connection errors.
            timeout: Request timeout in seconds.
        """
        self.model_name = model or DEFAULT_MODEL
        self.spec = resolve_model(self.model_name)
        self.max_retries = max_retries
        self.timeout = timeout
        self._client = None  # lazy init

        # API KEY INJECTION POINT:
        # Set the environment variable for the selected provider before running,
        # or pass api_key=... explicitly to LLMClient. Never commit a real key.
        #   OpenAI:   OPENAI_API_KEY
        #   Anthropic: ANTHROPIC_API_KEY
        #   DeepSeek: DEEPSEEK_API_KEY
        env_key_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY",
        }
        self.api_key = api_key or os.environ.get(
            env_key_map.get(self.spec.provider, ""), ""
        )
        if not _is_usable_key(self.api_key):
            env_name = env_key_map[self.spec.provider]
            raise LLMConfigurationError(
                f"Missing API key for {self.spec.provider}. Set {env_name} in the "
                ".env file/environment, or pass api_key=... to LLMClient."
            )

        # Resolve base URL
        if base_url:
            self.base_url = base_url
        elif self.spec.provider == "deepseek":
            self.base_url = "https://api.deepseek.com"
        else:
            self.base_url = None  # SDK defaults

    def _get_client(self):
        """Lazy-initialize the appropriate SDK client."""
        if self._client is not None:
            return self._client

        if self.spec.provider in ("openai", "deepseek"):
            import openai
            kwargs = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url + (
                    "" if self.base_url.endswith("/v1") else "/v1"
                )
            self._client = openai.OpenAI(**kwargs)
            return self._client

        elif self.spec.provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
            return self._client

        raise RuntimeError(f"Unknown provider: {self.spec.provider}")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a completion.

        Returns dict with keys: output, model, usage (token counts if available).
        """
        attempt = 0
        while True:
            try:
                return self._call_api(prompt, max_tokens, temperature, system_prompt)
            except (ConnectionError, TimeoutError) as e:
                attempt += 1
                if attempt > self.max_retries:
                    raise
                wait = min(attempt * 2, 30)
                logger.warning(
                    "Retry %d/%d (timeout/connection, waiting %ds): %s",
                    attempt, self.max_retries, wait, e,
                )
                time.sleep(wait)
            except Exception as e:
                attempt += 1
                error_str = str(e)
                # Retry on rate limits and server errors
                if "rate_limit" in error_str.lower() or "429" in error_str:
                    if attempt > self.max_retries:
                        raise
                    wait = min(attempt * 5, 30)
                    logger.warning(
                        "Rate limited, retry %d/%d (waiting %ds)",
                        attempt, self.max_retries, wait,
                    )
                    time.sleep(wait)
                    continue
                if "server_error" in error_str.lower() or "500" in error_str or "503" in error_str:
                    if attempt > self.max_retries:
                        raise
                    wait = min(attempt * 2, 15)
                    logger.warning(
                        "Server error, retry %d/%d (waiting %ds): %s",
                        attempt, self.max_retries, wait, e,
                    )
                    time.sleep(wait)
                    continue
                raise

    def _call_api(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute the actual API call based on provider."""
        provider = self.spec.provider
        model_id = self.spec.model_id

        if provider in ("openai", "deepseek"):
            client = self._get_client()
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 0.0,
                timeout=self.timeout,
            )
            content = response.choices[0].message.content or ""
            usage = {}
            if hasattr(response, "usage") and response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            return {"output": content, "model": model_id, "usage": usage}

        elif provider == "anthropic":
            client = self._get_client()
            kwargs = {
                "model": model_id,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}],
            }
            if system_prompt:
                kwargs["system"] = system_prompt
            if temperature > 0:
                kwargs["temperature"] = temperature

            response = client.messages.create(**kwargs)
            # Extract text from the first content block
            content = ""
            for block in response.content:
                if block.type == "text":
                    content += block.text
            usage = {}
            if hasattr(response, "usage") and response.usage:
                usage = {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                }
            return {"output": content, "model": model_id, "usage": usage}

        raise RuntimeError(f"Unknown provider: {provider}")

    def generate_with_retry(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        system_prompt: Optional[str] = None,
        min_output_len: int = 1,
    ) -> str:
        """Generate and return just the output string, retrying if empty.

        This is a convenience wrapper that calls generate() in a loop,
        retrying when the response is empty, up to max_retries times.
        Each attempt increases max_tokens by token_increment.
        """
        current_max = max_tokens
        for attempt in range(self.max_retries):
            result = self.generate(prompt, current_max, temperature, system_prompt)
            output = result.get("output", "").strip()
            if len(output) >= min_output_len:
                return output
            current_max += max_tokens
            logger.warning(
                "Empty/short output on attempt %d, retrying with %d tokens...",
                attempt + 1, current_max,
            )
            time.sleep(1)
        raise ValueError(f"Empty output after {self.max_retries} attempts")


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_default_client: Optional[LLMClient] = None


def get_client(model: Optional[str] = None) -> LLMClient:
    """Get or create the default global LLM client."""
    global _default_client
    selected_model = model or DEFAULT_MODEL
    if _default_client is None or _default_client.spec != resolve_model(selected_model):
        _default_client = LLMClient(model=selected_model)
    return _default_client


def generate(
    model: str,
    prompt: str,
    max_tokens: int = 2048,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """Quick one-shot generation using the global client."""
    client = get_client(model)
    return client.generate(prompt, max_tokens, temperature)
