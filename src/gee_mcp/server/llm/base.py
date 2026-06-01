from abc import ABC, abstractmethod
from enum import Enum

from .cache import NullCache, ResponseCache
from .types import LLMCallReturn


class LLMProvider(Enum):
    GOOGLE = "google"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


# provider -> implementing class, populated by the @register_llm decorator
_LLM_REGISTRY: dict[LLMProvider, type["BaseLLM"]] = {}


def register_llm(provider: LLMProvider):
    """Class decorator: tag a `BaseLLM` subclass with its provider and register it."""

    def deco(cls: type["BaseLLM"]) -> type["BaseLLM"]:
        cls._provider = provider
        _LLM_REGISTRY[provider] = cls
        return cls

    return deco


class BaseLLM(ABC):
    _provider: LLMProvider | None = None

    def __init__(
        self,
        api_key: str | None,
        model: str,
        cache: ResponseCache | None = None,
    ):
        if self._provider is None:
            raise RuntimeError(
                f"{type(self).__name__} must set a class-level `_provider`"
            )

        self.api_key = api_key
        self.model = model
        self.cache: ResponseCache = cache if cache is not None else NullCache()

    @classmethod
    @abstractmethod
    def from_env(
        cls, model: str, cache: ResponseCache | None = None
    ) -> "BaseLLM":
        """Build an instance using credentials/config from environment variables."""

    def _cache_key(self, text: str, include_thinking: bool) -> str:
        return f"{self.model}::{include_thinking}::{text}"

    def call(self, text: str, include_thinking: bool = True) -> LLMCallReturn:
        key = self._cache_key(text, include_thinking)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        call_return = self._call(text=text, include_thinking=include_thinking)
        self.cache.put(key, call_return)
        return call_return

    @abstractmethod
    def _call(self, text: str, include_thinking: bool = True) -> LLMCallReturn:
        pass
