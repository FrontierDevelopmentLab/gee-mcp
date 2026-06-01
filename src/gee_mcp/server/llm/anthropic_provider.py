import os

import anthropic

from .base import BaseLLM, LLMProvider, register_llm
from .cache import ResponseCache
from .types import LLMCallReturn


@register_llm(LLMProvider.ANTHROPIC)
class AnthropicLLM(BaseLLM):
    # Anthropic requires an explicit max output token count; adaptive thinking
    # plus tool-free prose answers fit comfortably under this.
    MAX_TOKENS = 16000

    def __init__(
        self,
        api_key,
        model,
        cache: ResponseCache | None = None,
    ):
        super().__init__(api_key=api_key, model=model, cache=cache)
        self.client = anthropic.Anthropic(api_key=api_key)

    @classmethod
    def from_env(
        cls, model: str, cache: ResponseCache | None = None
    ) -> "AnthropicLLM":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if api_key is None:
            raise ValueError("API key for Anthropic not found.")
        return cls(api_key=api_key, model=model, cache=cache)

    def _call(self, text: str, include_thinking: bool = True) -> LLMCallReturn:
        thinking_conf: anthropic.types.ThinkingConfigParam
        if include_thinking:
            # Adaptive thinking is the only supported "on" mode on Opus 4.7;
            # "summarized" surfaces the reasoning text instead of omitting it.
            thinking_conf = {"type": "adaptive", "display": "summarized"}
        else:
            thinking_conf = {"type": "disabled"}

        messages: list[anthropic.types.MessageParam] = [
            {"role": "user", "content": text}
        ]
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.MAX_TOKENS,
            thinking=thinking_conf,
            messages=messages,
        )

        thought = None
        answer = None

        for block in response.content:
            if block.type == "thinking":
                thought = block.thinking
            elif block.type == "text":
                answer = block.text

        return {"answer": answer, "thought": thought, "response": response}
