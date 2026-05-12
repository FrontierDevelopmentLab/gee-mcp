import os

import openai
from openai import OpenAI

from .base import BaseLLM, LLMProvider, register_llm
from .cache import ResponseCache
from .types import LLMCallReturn


@register_llm(LLMProvider.OPENAI)
class OpenAILLM(BaseLLM):
    def __init__(
        self,
        api_key,
        model,
        cache: ResponseCache | None = None,
    ):
        super().__init__(api_key=api_key, model=model, cache=cache)
        self.client = OpenAI(api_key=api_key)

    @classmethod
    def from_env(
        cls, model: str, cache: ResponseCache | None = None
    ) -> "OpenAILLM":
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("API key for OpenAI not found.")
        return cls(api_key=api_key, model=model, cache=cache)

    def _call(self, text: str, include_thinking: bool = True) -> LLMCallReturn:
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
                    p.text
                    for p in (item.summary or [])
                    if getattr(p, "text", None)
                ]
                thought = "\n".join(parts) or None
                break

        return {
            "answer": model_response.output_text,
            "thought": thought,
            "response": model_response,
        }
