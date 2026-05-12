"""Tests for the ``gee_mcp.server.llm`` package: registry, factory, cache."""


import pytest

from gee_mcp.server.llm import (
    AnthropicLLM,
    BaseLLM,
    GoogleLLM,
    JSONFileCache,
    LLMProvider,
    NullCache,
    OpenAILLM,
    init_llm_client,
    register_llm,
)
from gee_mcp.server.llm.base import _LLM_REGISTRY

# Env vars that any provider's ``from_env`` might read; cleared per-test so a
# developer's real credentials don't leak into the assertions.
_PROVIDER_ENV_VARS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "VERTEXAI_PROJECT",
    "VERTEXAI_LOCATION",
    "LLM_PROVIDER",
    "LLM_NAME",
)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch):
    """Remove all provider-related env vars for the duration of a test."""
    for var in _PROVIDER_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


class _FakeLLM(BaseLLM):
    """Minimal concrete ``BaseLLM`` whose ``_call`` records invocations."""

    def __init__(self, *, model="fake-model", cache=None):
        self.calls = 0
        super().__init__(api_key=None, model=model, cache=cache)

    @classmethod
    def from_env(cls, model="fake-model", cache=None):
        return cls(model=model, cache=cache)

    def _call(self, text, include_thinking=True):
        self.calls += 1
        return {
            "answer": f"answer-{self.calls}",
            "thought": None,
            "response": object(),
        }


# Give _FakeLLM a provider so BaseLLM.__init__ accepts it, without polluting
# the real registry that init_llm_client consults.
_FakeLLM._provider = LLMProvider.OPENAI


class TestRegistry:
    """The ``@register_llm`` decorator and ``_LLM_REGISTRY``."""

    @staticmethod
    def test_all_builtin_providers_registered():
        """Each ``LLMProvider`` maps to its implementing class."""
        assert _LLM_REGISTRY[LLMProvider.OPENAI] is OpenAILLM
        assert _LLM_REGISTRY[LLMProvider.ANTHROPIC] is AnthropicLLM
        assert _LLM_REGISTRY[LLMProvider.GOOGLE] is GoogleLLM

    @staticmethod
    def test_register_llm_sets_provider_attr():
        """The decorator stamps ``_provider`` onto the class."""
        assert OpenAILLM._provider is LLMProvider.OPENAI
        assert AnthropicLLM._provider is LLMProvider.ANTHROPIC
        assert GoogleLLM._provider is LLMProvider.GOOGLE

    @staticmethod
    def test_register_llm_is_idempotent_and_returns_class():
        """Re-registering a class is harmless and ``register_llm`` returns it."""

        class Dummy(BaseLLM):
            def _call(self, text, include_thinking=True):
                ...

            @classmethod
            def from_env(cls, model, cache=None):
                return cls(api_key=None, model=model, cache=cache)

        try:
            returned = register_llm(LLMProvider.OPENAI)(Dummy)
            assert returned is Dummy
            assert _LLM_REGISTRY[LLMProvider.OPENAI] is Dummy
        finally:
            # Restore so other tests (and the rest of the suite) see the real class.
            _LLM_REGISTRY[LLMProvider.OPENAI] = OpenAILLM

    @staticmethod
    def test_undecorated_subclass_raises():
        """A ``BaseLLM`` subclass without ``_provider`` cannot be instantiated."""

        class Rogue(BaseLLM):
            def _call(self, text, include_thinking=True):
                ...

            @classmethod
            def from_env(cls, model, cache=None):
                return cls(api_key=None, model=model, cache=cache)

        with pytest.raises(RuntimeError, match="_provider"):
            Rogue(api_key=None, model="x")


class TestInitLLMClient:
    """``init_llm_client`` dispatch and validation."""

    @staticmethod
    def test_unknown_provider_raises(clean_env):
        """An unrecognised provider string is rejected with the valid options."""
        with pytest.raises(ValueError, match="must be one of"):
            init_llm_client(provider="bogus", model="x")

    @staticmethod
    def test_missing_provider_raises(clean_env):
        """A ``None`` provider is rejected."""
        with pytest.raises(ValueError, match="provider not found"):
            init_llm_client(provider=None, model="x")

    @staticmethod
    def test_missing_model_raises(clean_env):
        """A ``None`` model is rejected (provider validated first)."""
        with pytest.raises(ValueError, match="identity not configured"):
            init_llm_client(provider="openai", model=None)

    @staticmethod
    def test_missing_api_key_raises(clean_env):
        """A valid provider with no credentials surfaces a provider-specific error."""
        with pytest.raises(ValueError, match="API key for OpenAI"):
            init_llm_client(provider="openai", model="gpt-5")

    @staticmethod
    def test_builds_requested_provider(clean_env, monkeypatch):
        """With credentials present, the right class is constructed."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        client = init_llm_client(provider="openai", model="gpt-5")
        assert isinstance(client, OpenAILLM)
        assert client.model == "gpt-5"
        assert isinstance(client.cache, NullCache)

    @staticmethod
    def test_cache_dir_attaches_json_file_cache(
        clean_env, monkeypatch, tmp_path
    ):
        """Passing ``cache_dir`` wires a ``JSONFileCache`` onto the client."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        client = init_llm_client(
            provider="openai", model="gpt-5", cache_dir=tmp_path
        )
        assert isinstance(client.cache, JSONFileCache)
        assert client.cache.directory == tmp_path


class TestFromEnv:
    """Per-provider ``from_env`` credential resolution."""

    @staticmethod
    def test_openai_from_env(clean_env, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        assert isinstance(OpenAILLM.from_env(model="gpt-5"), OpenAILLM)

    @staticmethod
    def test_openai_from_env_missing_key(clean_env):
        with pytest.raises(ValueError, match="API key for OpenAI"):
            OpenAILLM.from_env(model="gpt-5")

    @staticmethod
    def test_anthropic_from_env_missing_key(clean_env):
        with pytest.raises(ValueError, match="API key for Anthropic"):
            AnthropicLLM.from_env(model="claude-opus-4-7")

    @staticmethod
    def test_google_from_env_developer_api(clean_env, monkeypatch):
        """``GEMINI_API_KEY`` selects the developer-API construction path."""
        monkeypatch.setenv("GEMINI_API_KEY", "g-test")
        assert isinstance(GoogleLLM.from_env(), GoogleLLM)

    @staticmethod
    def test_google_from_env_falls_back_to_google_api_key(
        clean_env, monkeypatch
    ):
        monkeypatch.setenv("GOOGLE_API_KEY", "g-test")
        assert isinstance(GoogleLLM.from_env(), GoogleLLM)

    @staticmethod
    def test_google_from_env_no_credentials(clean_env):
        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            GoogleLLM.from_env()


class TestCaching:
    """Cache implementations and ``BaseLLM.call`` integration."""

    @staticmethod
    def test_null_cache_never_caches():
        """``NullCache`` always misses, so every call hits ``_call``."""
        llm = _FakeLLM()
        assert isinstance(llm.cache, NullCache)
        llm.call("hello")
        llm.call("hello")
        assert llm.calls == 2

    @staticmethod
    def test_default_cache_is_null_cache():
        assert isinstance(_FakeLLM().cache, NullCache)

    @staticmethod
    def test_call_uses_cache_on_hit(tmp_path):
        """A second identical call is served from the cache without re-invoking ``_call``."""
        llm = _FakeLLM(cache=JSONFileCache(tmp_path))
        first = llm.call("hello")
        second = llm.call("hello")
        assert llm.calls == 1
        assert second["answer"] == first["answer"] == "answer-1"
        assert second["response"] is None  # not persisted

    @staticmethod
    def test_call_distinguishes_keys(tmp_path):
        """Different prompts (and ``include_thinking``) are cached separately."""
        llm = _FakeLLM(cache=JSONFileCache(tmp_path))
        llm.call("a")
        llm.call("b")
        llm.call("a", include_thinking=False)
        assert llm.calls == 3
        assert len(list(tmp_path.glob("*.json"))) == 3

    @staticmethod
    def test_call_writes_through_to_cache(tmp_path):
        """A fresh client sharing the cache dir reads what a previous one wrote."""
        JSONFileCache(tmp_path)  # ensure dir exists
        first = _FakeLLM(cache=JSONFileCache(tmp_path))
        first.call("persist me")
        second = _FakeLLM(cache=JSONFileCache(tmp_path))
        result = second.call("persist me")
        assert second.calls == 0
        assert result["answer"] == "answer-1"

    @staticmethod
    def test_in_memory_cache_protocol(tmp_path):
        """Any object satisfying ``ResponseCache`` works — incl. an empty dict-backed one."""

        class MemCache(dict):
            def get(self, key):
                return dict.get(self, key)

            def put(self, key, value):
                self[key] = value

        cache = MemCache()
        llm = _FakeLLM(cache=cache)
        llm.call("z")
        llm.call("z")
        assert llm.calls == 1
        assert llm._cache_key("z", True) in cache

    @staticmethod
    def test_json_file_cache_persists_only_text(tmp_path):
        """The on-disk JSON contains ``answer``/``thought`` but not ``response``."""
        import json

        cache = JSONFileCache(tmp_path)
        cache.put(
            "k", {"answer": "hi", "thought": "thinking", "response": object()}
        )
        (path,) = tmp_path.glob("*.json")
        stored = json.loads(path.read_text())
        assert stored == {"answer": "hi", "thought": "thinking"}
        # And reading it back yields response=None.
        assert cache.get("k") == {
            "answer": "hi",
            "thought": "thinking",
            "response": None,
        }

    @staticmethod
    def test_json_file_cache_get_miss_returns_none(tmp_path):
        assert JSONFileCache(tmp_path).get("absent") is None
