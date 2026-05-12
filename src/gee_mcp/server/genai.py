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
from typing import Any, TypedDict
from pathlib import Path
from enum import Enum

# ----- File Base

class LLMProvider(Enum):
    GOOGLE = "google"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class BaseLLM(ABC):

    _provider: LLMProvider | None = None 

    class LLMCallReturn(TypedDict):
        """
        Structure of return from `call`:
        {"answer": answer, "thought": thought, "response": response}
        """
        answer: str
        thought: str | None
        response: Any  # raw provider SDK response object, or None for cache hits


    def __init__(
        self,
        api_key: str | None,
        model: str,
        cache_dir: Path | None,
    ):
        
        if self._provider is None:
            raise RuntimeError(
                f"{type(self).__name__} must set a class-level `_provider`"
            )

        self.api_key = api_key
        self.model = model
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            logger.debug(f"response caching enabled at {cache_dir}")

    def _cache_path(self, text: str, include_thinking: bool) -> str:
        content = f"{self.model}::{include_thinking}::{text}"
        key = hashlib.sha256(content.encode()).hexdigest()[:16]
        return os.path.join(self.cache_dir, f"{key}.json")

    def _check_cache(self, path: str) -> LLMCallReturn | None:
        if os.path.exists(path):
            logger.debug(f"cache hit: {path}")
            with open(path) as f:
                cached = json.load(f)
            return {
                "answer": cached["answer"],
                "thought": cached.get("thought"),
                "response": None,  # raw provider response isn't persisted
            }
        logger.debug(f"cache miss: {path}")
        return None

    def _save_to_cache(self, path: str, call_return: LLMCallReturn):
        # `response` is a provider SDK object and isn't JSON-serializable; only
        # the extracted text is persisted (re-reads get `response: None`).
        with open(path, "w") as f:
            json.dump(
                {"answer": call_return["answer"], "thought": call_return.get("thought")},
                f,
                indent=2,
            )
        logger.debug("cached response")

    def call(self, text: str, include_thinking: bool=True) -> LLMCallReturn:
        cache_path = (
            self._cache_path(text, include_thinking) if self.cache_dir else None
        )
        if cache_path is not None:
            cached = self._check_cache(cache_path)
            if cached is not None:
                return cached
        call_return = self._call(text=text, include_thinking=include_thinking)
        if cache_path is not None:
            self._save_to_cache(cache_path, call_return)
        return call_return

    @abstractmethod
    def _call(self, text: str, include_thinking: bool=True) -> LLMCallReturn:
        pass


# ------ File OpenAI -------------------


class OpenAILLM(BaseLLM):

    _provider = LLMProvider.OPENAI

    def __init__(self,
        api_key, 
        model,
        cache_dir: str | None = None,
    ):
        super().__init__(api_key=api_key, model=model, cache_dir=cache_dir)
        self.client = OpenAI(api_key=api_key)

    def _call(self, text: str, include_thinking: bool=True) -> BaseLLM.LLMCallReturn:
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


def init_openai_genai_client(model: str) -> OpenAILLM:
    api_key = os.getenv("OPENAI_API_KEY")

    if api_key is None:
        raise ValueError("API key for OpenAI not found.")

    return OpenAILLM(api_key=api_key, model=model)


# -------- File Anthropic -----------------------


class AnthropicLLM(BaseLLM):

    _provider = LLMProvider.ANTHROPIC

    # Anthropic requires an explicit max output token count; adaptive thinking
    # plus tool-free prose answers fit comfortably under this.
    MAX_TOKENS = 16000

    def __init__(self,
        api_key,
        model,
        cache_dir: str | None = None,
    ):

        super().__init__(api_key=api_key, model=model, cache_dir=cache_dir)
        self.client = anthropic.Anthropic(api_key=api_key)

    def _call(self, text: str, include_thinking: bool=True) -> BaseLLM.LLMCallReturn:

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


def init_anthropic_genai_client(model: str) -> AnthropicLLM:
    api_key = os.getenv("ANTHROPIC_API_KEY")

    if api_key is None:
        raise ValueError("API key for Anthropic not found.")

    return AnthropicLLM(api_key=api_key, model=model)


# ------ File Google -----------------------------------


class GoogleLLM(BaseLLM):

    _provider = LLMProvider.GOOGLE

    def __init__(
        self,
        api_key=None,
        project=None,
        location="global",
        model="gemini-3.1-pro-preview",
        cache_dir=None,
    ):
        """
        use either api_key (for Gemini Developer API), or project + location (for VertexAI API)

        :param cache_dir: if set, cache responses to this directory to avoid redundant API calls
        """

        super().__init__(api_key=api_key, model=model, cache_dir=cache_dir)

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

    def _call(self, text, include_thinking=True) -> BaseLLM.LLMCallReturn:


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


def init_google_genai_client(model: str="gemini-3.1-pro-preview") -> GoogleLLM:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    vertexai_project = os.getenv("VERTEXAI_PROJECT", False)
    vertexai_location = os.getenv("VERTEXAI_LOCATION", "global")

    if api_key:
        genai_client = GoogleLLM(api_key=api_key, model=model)

    else:
        if not vertexai_project:
            raise ValueError(
                "you must specified either an api key using GEMINI_API_KEY or GOOGLE_API_KEY environment variable or a vertexai project using VERTEXAI_PROJECT environment variable"
            )
        genai_client = GoogleLLM(
            project=vertexai_project, location=vertexai_location
        )

    return genai_client


# --------------- File genai ----------------------

LLM_INIT_DICT = {
    LLMProvider.GOOGLE: init_google_genai_client,
    LLMProvider.ANTHROPIC: init_anthropic_genai_client,
    LLMProvider.OPENAI: init_openai_genai_client,
}


def init_llm_client(
    provider: str | None = os.getenv("LLM_PROVIDER"), 
    model: str | None = os.getenv("LLM_NAME"),
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

    return LLM_INIT_DICT[llm_provider](model=model)
