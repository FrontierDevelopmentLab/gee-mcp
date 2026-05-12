import os
from pathlib import Path

from .base import _LLM_REGISTRY, BaseLLM, LLMProvider
from .cache import JSONFileCache


def init_llm_client(
    provider: str | None = os.getenv("LLM_PROVIDER"),
    model: str | None = os.getenv("LLM_NAME"),
    cache_dir: str | Path | None = None,
) -> BaseLLM:
    if provider is None:
        raise ValueError("LLM provider not found.")

    try:
        llm_provider = LLMProvider(provider)
    except ValueError:
        raise ValueError(
            f"LLM provider must be one of the following: "
            f"{[prov.value for prov in LLMProvider]}"
        )

    if model is None:
        raise ValueError("LLM identity not configured.")

    cache = JSONFileCache(cache_dir) if cache_dir else None
    return _LLM_REGISTRY[llm_provider].from_env(model=model, cache=cache)
