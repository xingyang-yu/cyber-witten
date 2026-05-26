"""Pluggable LLM backends for the Cyber-Witten generation step.

Retrieval (BGE + FAISS) is provider-agnostic; only the final "answer with
citations" call needs an LLM. This module isolates the per-provider details
behind a small `LLMBackend` protocol so `ask.py` stays clean and the project
isn't locked into one vendor.

Built-in backends:
    anthropic   Claude (default; sponsor model, best citation discipline)
    openai      GPT-4o family
    ollama      Local llama.cpp via Ollama (no API key, free)

The OpenAI-compatible backend can also reach Together/Groq/Fireworks/vLLM
by setting `base_url`; see OpenAICompatibleBackend.

All third-party SDKs are lazy-imported so users only need to install the
ones they actually use:
    pip install openai      # for openai or ollama backends
    pip install anthropic   # already in requirements.txt (default backend)
"""
from __future__ import annotations

import os
from typing import Protocol


class LLMBackend(Protocol):
    """Minimal interface every backend implements."""

    name: str
    model: str

    def generate(self, system: str, user: str, max_tokens: int = 2048) -> str: ...


class AnthropicBackend:
    name = "anthropic"
    DEFAULT_MODEL = "claude-sonnet-4-6"

    def __init__(self, model: str | None = None):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Set it in .env, or pick another "
                "backend: --provider openai (needs OPENAI_API_KEY) or "
                "--provider ollama (local, no key). For passages only, use "
                "--retrieve-only."
            )
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ImportError(
                "Anthropic backend requires `pip install anthropic`"
            ) from exc
        self._client = Anthropic()
        self.model = model or self.DEFAULT_MODEL

    def generate(self, system: str, user: str, max_tokens: int = 2048) -> str:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text


class OpenAICompatibleBackend:
    """Works against any OpenAI-compatible Chat Completions endpoint.

    Concrete providers (OpenAI, Ollama, Together, Groq, Fireworks, vLLM, ...)
    just supply a `base_url` and the appropriate env var for the API key.
    """

    name = "openai-compatible"
    DEFAULT_MODEL = "gpt-4o"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        api_key_required: bool = True,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                f"The '{self.name}' backend uses the OpenAI Python SDK as a "
                "transport (works for OpenAI, Ollama, Together, Groq, vLLM, ...). "
                "Install it with: pip install openai"
            ) from exc
        api_key = os.environ.get(api_key_env)
        if api_key_required and not api_key:
            raise RuntimeError(
                f"{api_key_env} not set. Set it in .env, or pick another "
                "backend: --provider anthropic (needs ANTHROPIC_API_KEY) or "
                "--provider ollama (local, no key). For passages only, use "
                "--retrieve-only."
            )
        self._client = OpenAI(
            api_key=api_key or "not-required",
            base_url=base_url,
        )
        self.model = model or self.DEFAULT_MODEL

    def generate(self, system: str, user: str, max_tokens: int = 2048) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content


class OpenAIBackend(OpenAICompatibleBackend):
    name = "openai"
    DEFAULT_MODEL = "gpt-4o"

    def __init__(self, model: str | None = None):
        super().__init__(model=model)  # default base_url + OPENAI_API_KEY


class OllamaBackend(OpenAICompatibleBackend):
    """Local Ollama server (https://ollama.com) — no API key, no cost.

    Prereq: `ollama serve` running, and `ollama pull <model>` for whatever
    model you'll use (e.g. `ollama pull llama3.1:8b`).
    """

    name = "ollama"
    DEFAULT_MODEL = "llama3.1:8b"
    DEFAULT_BASE_URL = "http://localhost:11434/v1"

    def __init__(self, model: str | None = None):
        super().__init__(
            model=model,
            base_url=self.DEFAULT_BASE_URL,
            api_key_env="OLLAMA_API_KEY",  # not actually used; allow override
            api_key_required=False,
        )


BACKENDS: dict[str, type[LLMBackend]] = {
    "anthropic": AnthropicBackend,
    "openai": OpenAIBackend,
    "ollama": OllamaBackend,
}


def get_backend(provider: str, model: str | None = None) -> LLMBackend:
    """Factory: instantiate the backend identified by `provider`."""
    if provider not in BACKENDS:
        raise ValueError(
            f"Unknown provider {provider!r}. Available: {sorted(BACKENDS)}"
        )
    return BACKENDS[provider](model=model)


def available_providers() -> list[str]:
    return sorted(BACKENDS.keys())
