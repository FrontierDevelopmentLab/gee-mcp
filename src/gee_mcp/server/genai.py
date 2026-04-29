import hashlib
import json
import os

from google import genai as google_genai
from google.genai import types
from loguru import logger


def init_genai_client(model="gemini-3.1-pro-preview"):
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    vertexai_project = os.getenv("VERTEXAI_PROJECT", False)
    vertexai_location = os.getenv("VERTEXAI_LOCATION", "global")

    if api_key:
        genai_client = Gemini(api_key=api_key, model=model)

    else:
        if not vertexai_project:
            raise ValueError(
                "you must specified either an api key using GEMINI_API_KEY or GOOGLE_API_KEY environment variable or a vertexai project using VERTEXAI_PROJECT environment variable"
            )
        genai_client = Gemini(
            project=vertexai_project, location=vertexai_location
        )

    return genai_client


class Gemini:
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

        self.model = model
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            logger.debug(f"response caching enabled at {cache_dir}")
        logger.debug(f"genai client initialized with model {self.model}")

    def _cache_key(self, text, include_thinking):
        content = f"{self.model}::{include_thinking}::{text}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def call(self, text, include_thinking=True):
        # check cache
        if self.cache_dir:
            key = self._cache_key(text, include_thinking)
            cache_path = os.path.join(self.cache_dir, f"{key}.json")
            if os.path.exists(cache_path):
                logger.debug(f"cache hit: {key}")
                with open(cache_path) as f:
                    cached = json.load(f)
                return {
                    "answer": cached["answer"],
                    "thought": cached.get("thought"),
                    "response": None,
                }

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

        # save to cache
        if self.cache_dir:
            with open(cache_path, "w") as f:
                json.dump({"answer": answer, "thought": thought}, f, indent=2)
            logger.debug(f"cached response: {key}")

        return {"answer": answer, "thought": thought, "response": response}
