import os

from google import genai as google_genai
from google.genai import types
from loguru import logger

from .base import BaseLLM, LLMProvider, register_llm
from .cache import ResponseCache
from .types import LLMCallReturn


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
        cls,
        model: str = "gemini-3.1-pro-preview",
        cache: ResponseCache | None = None,
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
        return cls(
            project=project, location=location, model=model, cache=cache
        )

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

        candidates = response.candidates or []
        content = candidates[0].content if candidates else None
        for part in (content.parts or []) if content else []:
            if not part.text:
                continue
            if part.thought:
                thought = part.text
            else:
                answer = part.text

        return {"answer": answer, "thought": thought, "response": response}
