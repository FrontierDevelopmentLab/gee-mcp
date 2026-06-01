import hashlib
import json
from pathlib import Path
from typing import Protocol

from loguru import logger

from .types import LLMCallReturn


class ResponseCache(Protocol):
    """Caches `LLMCallReturn`s keyed by an opaque string."""

    def get(self, key: str) -> LLMCallReturn | None:
        ...

    def put(self, key: str, value: LLMCallReturn) -> None:
        ...


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
                {"answer": value["answer"], "thought": value.get("thought")},
                indent=2,
            )
        )
        logger.debug(f"cached response: {path}")
