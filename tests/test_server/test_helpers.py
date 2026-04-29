"""Tests for ``helpers`` utilities (no live GEE / Gemini)."""

import importlib

import pytest

from gee_mcp.config import SERVER_MODULE as PKG


@pytest.fixture()
def helpers_mod():
    """Return the imported helpers module."""
    return importlib.import_module(f"{PKG}.helpers")


class TestExtractXmlTag:
    """Test ``helpers.extract_xml_tag``."""

    @staticmethod
    def test_extracts_simple_content(helpers_mod):
        """Returns content between matching tags."""
        text = "prefix <FOO>hello</FOO> suffix"
        assert helpers_mod.extract_xml_tag(text, "FOO") == "hello"

    @staticmethod
    def test_raises_on_missing_tag(helpers_mod):
        """Raises ``NoTagFoundError`` when tag absent."""
        with pytest.raises(helpers_mod.NoTagFoundError):
            helpers_mod.extract_xml_tag("no tags here", "FOO")

    @staticmethod
    def test_raises_on_only_open(helpers_mod):
        """Raises when opening tag has no closing partner."""
        with pytest.raises(helpers_mod.NoTagFoundError):
            helpers_mod.extract_xml_tag("<FOO>no closer", "FOO")


class TestExtractTag:
    """Test ``helpers.extract_tag`` (markdown fenced blocks)."""

    @staticmethod
    def test_extracts_fenced_block(helpers_mod):
        """Returns content of the first fenced block."""
        text = "intro\n```json\n[1, 2]\n```\nouter"
        out = helpers_mod.extract_tag(text, "json")
        assert "[1, 2]" in out

    @staticmethod
    def test_non_greedy_with_multiple_fences(helpers_mod):
        """Does not consume across subsequent fences."""
        text = "```json\nfirst\n```\nmiddle\n```json\nsecond\n```"
        out = helpers_mod.extract_tag(text, "json")
        assert "first" in out
        assert "second" not in out

    @staticmethod
    def test_raises_when_missing(helpers_mod):
        """Raises when no fence is present."""
        with pytest.raises(helpers_mod.NoTagFoundError):
            helpers_mod.extract_tag("plain text", "json")


class TestRemoveLeadingSpaces:
    """Test ``helpers.remove_leading_spaces`` lambda."""

    @staticmethod
    def test_strips_each_line(helpers_mod):
        """Each line is independently stripped of leading whitespace."""
        text = "  hello\n    world\n\tfoo"
        out = helpers_mod.remove_leading_spaces(text)
        assert out == "hello\nworld\nfoo"
