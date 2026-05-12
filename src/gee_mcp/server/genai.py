import hashlib
import json
import os
from abc import abstractmethod, ABC
import anthropic
import openai
from openai import OpenAI
from google import genai as google_genai
from google.genai import types
from loguru import logger
from typing import Any, Protocol, TypedDict
from pathlib import Path
from enum import Enum

# ----- File Base

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


class LLMCallReturn(TypedDict):
    """
    Structure of return from `call`:
    {"answer": answer, "thought": thought, "response": response}
    """
    answer: str
    thought: str | None
    response: Any  # raw provider SDK response object, or None for cache hits


class ResponseCache(Protocol):
    """Caches `LLMCallReturn`s keyed by an opaque string."""

    def get(self, key: str) -> LLMCallReturn | None: ...

    def put(self, key: str, value: LLMCallReturn) -> None: ...


class NullCache:
    """No-op cache — the default when caching isn't configured."""

    def get(self, key: str) -> LLMCallReturn | None:
        return None

    def put(self, key: str, value: LLMCallReturn) -> None:
        pass


class JSONFileCache:
    """Persists each `LLMCallReturn` as a JSON file named by a hash of the key.

    The raw provider `response` object isn't JSON-serializable, so only the
    extracted text is persisted; re-reads return `response: None`.
    """

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        logger.debug(f"response caching enabled at {self.directory}")

    def _path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode()).hexdigest()[:16]
        return self.directory / f"{digest}.json"

    def get(self, key: str) -> LLMCallReturn | None:
        path = self._path(key)
        if not path.exists():
            logger.debug(f"cache miss: {path}")
            return None
        logger.debug(f"cache hit: {path}")
        cached = json.loads(path.read_text())
        return {
            "answer": cached["answer"],
            "thought": cached.get("thought"),
            "response": None,
        }

    def put(self, key: str, value: LLMCallReturn) -> None:
        path = self._path(key)
        path.write_text(
            json.dumps(
                {"answer": value["answer"], "thought": value.get("thought")}, indent=2
            )
        )
        logger.debug(f"cached response: {path}")


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

    def call(self, text: str, include_thinking: bool=True) -> LLMCallReturn:
        key = self._cache_key(text, include_thinking)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        call_return = self._call(text=text, include_thinking=include_thinking)
        self.cache.put(key, call_return)
        return call_return

    @abstractmethod
    def _call(self, text: str, include_thinking: bool=True) -> LLMCallReturn:
        pass


# ------ File OpenAI -------------------


@register_llm(LLMProvider.OPENAI)
class OpenAILLM(BaseLLM):

    def __init__(self,
        api_key,
        model,
        cache: ResponseCache | None = None,
    ):
        super().__init__(api_key=api_key, model=model, cache=cache)
        self.client = OpenAI(api_key=api_key)

    @classmethod
    def from_env(cls, model: str, cache: ResponseCache | None = None) -> "OpenAILLM":
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("API key for OpenAI not found.")
        return cls(api_key=api_key, model=model, cache=cache)

    def _call(self, text: str, include_thinking: bool=True) -> LLMCallReturn:
        kwargs: dict = {"model": self.model, "input": text}
        if include_thinking:
            # Only takes effect on reasoning models (o-series, gpt-5, ...); the
            # raw chain-of-thought is never returned, just this summary. Some
            # orgs require verification before summaries are permitted, hence
            # the fallback below.
            kwargs["reasoning"] = {"summary": "auto"}

        try:
            model_response = self.client.responses.create(**kwargs)
        except openai.BadRequestError:
            if "reasoning" not in kwargs:
                raise
            kwargs.pop("reasoning")
            model_response = self.client.responses.create(**kwargs)

        thought = None
        for item in model_response.output:
            if item.type == "reasoning":
                parts = [
                    p.text for p in (item.summary or []) if getattr(p, "text", None)
                ]
                thought = "\n".join(parts) or None
                break

        return {
            "answer": model_response.output_text,
            "thought": thought,
            "response": model_response,
        }


# -------- File Anthropic -----------------------


@register_llm(LLMProvider.ANTHROPIC)
class AnthropicLLM(BaseLLM):

    # Anthropic requires an explicit max output token count; adaptive thinking
    # plus tool-free prose answers fit comfortably under this.
    MAX_TOKENS = 16000

    def __init__(self,
        api_key,
        model,
        cache: ResponseCache | None = None,
    ):

        super().__init__(api_key=api_key, model=model, cache=cache)
        self.client = anthropic.Anthropic(api_key=api_key)

    @classmethod
    def from_env(cls, model: str, cache: ResponseCache | None = None) -> "AnthropicLLM":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key is None:
            raise ValueError("API key for Anthropic not found.")
        return cls(api_key=api_key, model=model, cache=cache)

    def _call(self, text: str, include_thinking: bool=True) -> LLMCallReturn:

        if include_thinking:
            # Adaptive thinking is the only supported "on" mode on Opus 4.7;
            # "summarized" surfaces the reasoning text instead of omitting it.
            thinking_conf = {"type": "adaptive", "display": "summarized"}
        else:
            thinking_conf = {"type": "disabled"}

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.MAX_TOKENS,
            thinking=thinking_conf,
            messages=[{"role": "user", "content": text}],
        )

        thought = None
        answer = None

        for block in response.content:
            if block.type == "thinking":
                thought = block.thinking
            elif block.type == "text":
                answer = block.text

        return {"answer": answer, "thought": thought, "response": response}


# ------ File Google -----------------------------------


@register_llm(LLMProvider.GOOGLE)
class GoogleLLM(BaseLLM):

    def __init__(
        self,
        api_key=None,
        project=None,
        location="global",
        model="gemini-3.1-pro-preview",
        cache: ResponseCache | None = None,
    ):
        """
        use either api_key (for Gemini Developer API), or project + location (for VertexAI API)
        """

        super().__init__(api_key=api_key, model=model, cache=cache)

        if api_key is None and project is None:
            raise ValueError(
                "must provide either api_key or project+location for Gemini client initialization"
            )

        if api_key is not None:
            logger.debug("google genai client using developer api")
            self.client = google_genai.Client(vertexai=False, api_key=api_key)
        else:
            logger.debug(
                f"google genai client using vertexai api project {project} location {location}"
            )
            self.client = google_genai.Client(
                vertexai=True, project=project, location=location
            )

        logger.debug(f"genai client initialized with model {self.model}")

    @classmethod
    def from_env(
        cls, model: str = "gemini-3.1-pro-preview", cache: ResponseCache | None = None
    ) -> "GoogleLLM":
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if api_key:
            return cls(api_key=api_key, model=model, cache=cache)

        project = os.getenv("VERTEXAI_PROJECT", False)
        if not project:
            raise ValueError(
                "you must specify either an api key using the GEMINI_API_KEY or "
                "GOOGLE_API_KEY environment variable, or a vertexai project using "
                "the VERTEXAI_PROJECT environment variable"
            )
        location = os.getenv("VERTEXAI_LOCATION", "global")
        return cls(project=project, location=location, model=model, cache=cache)

    def _call(self, text, include_thinking=True) -> LLMCallReturn:


        if include_thinking:
            thinking_conf = types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(include_thoughts=True)
            )
        else:
            thinking_conf = None

        # Create the multimodal content parts
        response = self.client.models.generate_content(
            model=self.model, contents=[text], config=thinking_conf
        )

        thought = None
        answer = None

        for part in response.candidates[0].content.parts:
            if not part.text:
                continue
            if part.thought:
                thought = part.text
            else:
                answer = part.text

        return {"answer": answer, "thought": thought, "response": response}


# --------------- File genai ----------------------


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
